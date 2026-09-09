// dit_graph.cpp — 阶段3：GPU 常驻整图（分块合并计算图）实现
// 把 32 层 DiT 合并为 8 块 × 4 层的 param/bin 整图（export_dit_graph.py 生成），
// 块内 input(VkMat) -> 连续 extract(VkMat) -> download 一次，块间 vid/txt 残差
// download/upload，消除原有逐层 Net 的 2 upload + 2 download CPU 往返。
#include "dit_vk.h"
#include "awa_layer.h"
#include "ada_compose.h"
#include "cast_bf16_layer.h"
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/mat.h>
#include <ncnn/command.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdlib>

// 与 dit_vk.cpp 一致的架构常量
static const int DIM = 2560;
static const int NUM_LAYERS = 32, MM_LAYERS = 10;

// 块图前缀：默认按 precision_ 选择（fp32/fp16/bf16 三套块图）；
// 可用 SEEDVR_BLOCK_PREFIX 环境变量覆盖，用于测试不同 chunk 导出的块图
// （如 dit_block_bf16_c8_* = 8 层/块，验证「块数越少 per-Net 固定开销越低」）。
static const char* block_prefix(int precision)
{
    const char* ov = getenv("SEEDVR_BLOCK_PREFIX");
    if (ov && ov[0]) return ov;
    return (precision == 0) ? "dit_block_" : (precision == 1 ? "dit_block_f16_" : "dit_block_bf16_");
}

static std::vector<unsigned char> graph_read_bin(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(0, std::ios::end);
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    std::vector<unsigned char> d(n);
    if (n) f.read((char*)d.data(), n);
    return d;
}

// 加载单个块 Net 到 gblocks_[b]（未加载时）；返回 false 表示文件缺失/加载失败
bool DitVk::graph_load_block(int b)
{
    if (gblocks_[b]) return true;
    const char* prefix = block_prefix(precision_);
    char pname[64], bname[64];
    snprintf(pname, sizeof(pname), "%s%d.param", prefix, b);
    snprintf(bname, sizeof(bname), "%s%d.bin", prefix, b);
    std::string pfull = graph_dir_ + pname, bfull = graph_dir_ + bname;
    std::ifstream fp(pfull);
    if (!fp) { fprintf(stderr, "[graph] 缺 %s\n", pfull.c_str()); return false; }
    std::vector<unsigned char> bbin = graph_read_bin(bfull);
    if (bbin.empty()) { fprintf(stderr, "[graph] 缺 %s\n", bfull.c_str()); return false; }
    std::unique_ptr<ncnn::Net> bn(new ncnn::Net);
    bn->set_vulkan_device(vkdev);
    bn->opt.use_vulkan_compute = true;
    // 低精度模式：标准算子 低16位 存储 + fp32 累加（tensor core）；AWA 前后 Cast 转 fp32（flat shader）
    // ⚠️ 必须区分 fp16/bf16：--fp16 走 fp16 存储（AwaLayer 内 cast_f16 管线），--bf16 走 bf16 存储。
    //    之前统一用 bf16（precision_!=0）导致 --fp16 实际跑 bf16（PSNR 15dB 级）。
    bool bf16 = (precision_ == 2);
    bool f16  = (precision_ == 1);
    if (getenv("SEEDVR_GRAPH_FP32STORAGE")) { bf16 = false; f16 = false; }  // 诊断：强制 fp32 存储（对齐 test_dit_graph）
    bn->opt.use_bf16_storage = bf16;
    bn->opt.use_bf16_packed  = bf16;
    bn->opt.use_fp16_storage = f16;
    bn->opt.use_fp16_packed  = f16;
    bn->opt.use_fp16_arithmetic = false;   // fp32 累加（fp16/bf16 累加器会溢出）
    bn->opt.lightmode = false;   // lightmode=true 在 extract 多 blob 时会递归重算崩
    // DIAG: SEEDVR_NO_REUSE=1 时用 VkWeightAllocator 作为 blob allocator（每次独立分配，不复用中间 blob），
    // 验证 ncnn 的 blob 复用是否是 vid_in/ada 错误根因
    ncnn::VkAllocator* b_alloc = blob_alloc;
    if (getenv("SEEDVR_NO_REUSE")) {
        if (!noreuse_alloc_) noreuse_alloc_ = new ncnn::VkWeightAllocator(vkdev);
        b_alloc = noreuse_alloc_;
    }
    bn->opt.blob_vkallocator = b_alloc;
    bn->opt.workspace_vkallocator = b_alloc;
    bn->opt.staging_vkallocator = staging_alloc;
    bn->register_custom_layer("AWA", AwaLayer::creator);
    bn->register_custom_layer("AdaCompose", AdaComposeLayer::creator);
    bn->register_custom_layer("Bf16Cast", Bf16CastLayer::creator);
    if (bn->load_param(pfull.c_str()) != 0) { fprintf(stderr, "[graph] block %d load_param FAIL\n", b); return false; }
    if (bn->load_model(bbin.data()) < 0) { fprintf(stderr, "[graph] block %d load_model FAIL\n", b); return false; }
    gblocks_[b] = std::move(bn);
    return true;
}

// 加载单 32 层整图 Net（dit_graph.param/bin）。
// 权重 bin 是 fp16（type=1），但按 Net opt 决定的中间激活精度运行：
//   --bf16（precision=2）：激活 bf16（防 1080p 大激活 fp16 溢出），权重仍 fp16 读（权重不会 >65504）
//   --fp16（precision=1）：权重+激活均 fp16（360p 小激活可用，1080p 可能溢出）
//   fp32（precision=0）：单 Net fp32 权重 20GB 超 16GB 显存，不支持。
bool DitVk::load_single_net()
{
    if (single_net_) return true;
    // fp32 单 Net 仅禁止「全 32 层」（20GB 权重超 16GB 显存）；小单 Net（如 4 层 1.3GB）
    // 用 fp32 可做精确数值对拍（与逐层参考路径同精度）。全层请改用 --bf16/--fp16。
    if (precision_ == 0 && single_nlayers_ >= NUM_LAYERS) {
        fprintf(stderr, "[graph] 单 Net fp32 仅支持 < 全层（32 层 fp32 权重 20GB 超 16GB 显存）；请用 --bf16/--fp16 或更小单 Net\n");
        return false;
    }
    std::string pfull = graph_dir_ + single_prefix_ + ".param";
    std::ifstream fp(pfull);
    if (!fp) { fprintf(stderr, "[graph] 缺 %s\n", pfull.c_str()); return false; }
    std::vector<unsigned char> bbin = graph_read_bin(graph_dir_ + single_prefix_ + ".bin");
    if (bbin.empty()) { fprintf(stderr, "[graph] 缺 %s\n", (graph_dir_ + single_prefix_ + ".bin").c_str()); return false; }
    single_net_.reset(new ncnn::Net);
    single_net_->set_vulkan_device(vkdev);
    single_net_->opt.use_vulkan_compute = true;
    bool bf16 = (precision_ == 2);
    bool f16  = (precision_ == 1);
    single_net_->opt.use_bf16_storage = bf16;
    single_net_->opt.use_bf16_packed  = bf16;
    single_net_->opt.use_fp16_storage = f16;
    single_net_->opt.use_fp16_packed  = f16;
    single_net_->opt.use_fp16_arithmetic = false;   // fp32 累加（fp16/bf16 累加器会溢出）
    single_net_->opt.lightmode = false;
    single_net_->opt.blob_vkallocator = blob_alloc;
    single_net_->opt.workspace_vkallocator = blob_alloc;
    single_net_->opt.staging_vkallocator = staging_alloc;
    single_net_->register_custom_layer("AWA", AwaLayer::creator);
    single_net_->register_custom_layer("AdaCompose", AdaComposeLayer::creator);
    single_net_->register_custom_layer("Bf16Cast", Bf16CastLayer::creator);
    if (single_net_->load_param(pfull.c_str()) != 0) { fprintf(stderr, "[graph] single load_param FAIL\n"); return false; }
    if (single_net_->load_model(bbin.data()) < 0) { fprintf(stderr, "[graph] single load_model FAIL\n"); return false; }
    fprintf(stderr, "[graph] single Net 加载 OK (prefix=%s, %d层, precision=%d, 权重常驻 + %s激活)\n",
            single_prefix_.c_str(), single_nlayers_, precision_, bf16 ? "bf16" : "fp16");
    return true;
}

bool DitVk::load_graph(const std::string& graph_dir, int chunk)
{
    if (graph_loaded_) release_graph();
    graph_dir_ = graph_dir;
    if (graph_dir_.empty() || graph_dir_.back() != '/') graph_dir_ += '/';
    if (chunk < 1) chunk = 4;
    if (NUM_LAYERS % chunk != 0) { fprintf(stderr, "[graph] chunk=%d 不整除 %d\n", chunk, NUM_LAYERS); return false; }
    gblock_chunk_ = chunk;
    gblock_count_ = NUM_LAYERS / chunk;
    gblocks_.clear();
    gblocks_.resize(gblock_count_);
    // 只校验文件存在（块 Net 按需加载+用完释放：8 块 fp32 权重 20GB 超 16GB 显存，不能全常驻）
    const char* prefix = block_prefix(precision_);
    for (int b = 0; b < gblock_count_; b++) {
        char pname[64];
        snprintf(pname, sizeof(pname), "%s%d.param", prefix, b);
        std::ifstream fp(graph_dir_ + pname);
        if (!fp) { fprintf(stderr, "[graph] 缺 %s（需先跑 export_dit_graph.py 生成块图；fp16 需 f16 变体）\n", (graph_dir_ + pname).c_str()); return false; }
    }
    graph_loaded_ = true;
    gblock_run_ = gblock_count_;   // 默认跑全部块；set_run_blocks(n) 可限制只跑前 n 块（用于小单 Net 同层对比）
    win_injected_ = false;
    fprintf(stderr, "[graph] ready (%d blocks, chunk=%d; 块 Net 按需加载+用完释放)\n", gblock_count_, chunk);
    return true;
}

void DitVk::release_graph()
{
    gblocks_.clear();
    single_net_.reset();
    for (auto& v : up_alloc_) for (auto* a : v) delete a;
    up_alloc_.clear();
    single_loaded_ = false;
    graph_loaded_ = false;
    win_injected_ = false;
}

// emb(15360,) + branch -> 每层每流 6 个 (DIM,) ada 向量（与 test/engine 同公式）
std::vector<std::vector<float>> DitVk::compute_ada(const std::vector<float>& emb, int n_layers) const
{
    std::vector<std::vector<float>> ada((size_t)n_layers * 2 * 6, std::vector<float>(DIM, 0.f));
    for (int i = 0; i < n_layers; i++) {
        bool dual = (i < MM_LAYERS);
        const RawBranch* br[2] = { &bvid_raw[i], &btxt_raw[i] };
        for (int flow = 0; flow < 2; flow++) {
            const RawBranch& R = *br[flow];
            int idx0 = (i * 2 + flow) * 6;
            for (int d = 0; d < DIM; d++) {
                ada[idx0 + 0][d] = emb[d * 6 + 1] + R.attn_scale[d];   // a_sc
                ada[idx0 + 1][d] = emb[d * 6 + 0] + R.attn_shift[d];   // a_sh
                ada[idx0 + 2][d] = emb[d * 6 + 2] + R.attn_gate[d];    // a_g
                ada[idx0 + 3][d] = emb[d * 6 + 4] + R.mlp_scale[d];    // m_sc
                ada[idx0 + 4][d] = emb[d * 6 + 3] + R.mlp_shift[d];    // m_sh
                ada[idx0 + 5][d] = emb[d * 6 + 5] + R.mlp_gate[d];     // m_g
            }
        }
    }
    return ada;
}

// 为块 b 的 AWA 层注入窗口几何（每块独立跟踪签名；窗口变化时重新注入）
// ★ 必须在块 Net 加载之后调用（graph_load_block 之后）——否则 mutable_layers() 是空列表，
//   注入无效，AWA 会 fallback 到 win bin（nwin=8/3200 网格）导致窗口错乱 -> 黑图
void DitVk::inject_graph_geometry(int b)
{
    ncnn::Net* net = single_loaded_ ? single_net_.get()
                                     : (b >= 0 && b < (int)gblocks_.size() ? gblocks_[b].get() : nullptr);
    if (!net) return;
    int sig = win_ns.t * 1000000 + win_ns.h * 10000 + win_ns.w * 100 + win_ns.nwin;
    sig = sig * 1000 + win_ns.txt_len + (win_sh.nwin << 8);
    if ((int)gblock_geom_sig_.size() <= b) gblock_geom_sig_.resize(b + 1, -1);
    if (gblock_geom_sig_[b] == sig) return;   // 该块已用当前窗口注入过
    // Win -> 几何（与 awa_vk.cpp / test 同公式）
    auto win_to_geom = [&](const Win& w, std::vector<float>& vidx, std::vector<float>& cumf,
                           std::vector<float>& vfreq, std::vector<float>& tfreq) {
        cumf.assign(w.nwin + 1, 0.f);
        vidx.clear();
        for (int wi = 0; wi < w.nwin; wi++) {
            int f_i = (w.en[wi] - w.st[wi]) * (w.eh[wi] - w.sh[wi]) * (w.ew[wi] - w.sw[wi]);
            cumf[wi + 1] = cumf[wi] + (float)f_i;
            for (int lt = w.st[wi]; lt < w.en[wi]; lt++)
                for (int lh = w.sh[wi]; lh < w.eh[wi]; lh++)
                    for (int lw = w.sw[wi]; lw < w.ew[wi]; lw++)
                        vidx.push_back((float)(((lt * w.h) + lh) * w.w + lw));
        }
        vfreq = w.vid_freq;
        tfreq = w.txt_freq;
    };
    std::vector<float> gv1, gc1, gf1, gt1, gv2, gc2, gf2, gt2;
    win_to_geom(win_ns, gv1, gc1, gf1, gt1);
    win_to_geom(win_sh, gv2, gc2, gf2, gt2);
        int Lv = (int)(win_ns.h * win_ns.w);   // 单帧 token 数（t=1）
    for (auto* l : net->mutable_layers()) {
        if (l->type != "AWA") continue;
        AwaLayer* a = (AwaLayer*)l;
        if (a->win_type() == 1)
            a->set_window_geometry(gv2, gc2, gf2, gt2, win_sh.nwin, Lv, win_sh.txt_len);
        else
            a->set_window_geometry(gv1, gc1, gf1, gt1, win_ns.nwin, Lv, win_ns.txt_len);
    }
    gblock_geom_sig_[b] = sig;
    win_injected_ = true;
    win_sig_ = sig;
    fprintf(stderr, "[graph] block %d AWA 几何注入 OK (Lv=%d, ns.nwin=%d sh.nwin=%d)\n", b, Lv, win_ns.nwin, win_sh.nwin);
}

bool DitVk::forward_graph(const std::vector<float>& vid_patch, int Lv,
                          const std::vector<float>& txt, int TXT, float timestep,
                          std::vector<float>& out_sr)
{
    if (!graph_loaded_) { fprintf(stderr, "[graph] 未加载\n"); return false; }

    std::vector<float> emb = time_embedding(timestep);
    // ada/final 在 AdaCompose 图中由 GPU shader 算，无需 CPU 预计算
    // 旧图路径仍需 CPU compute_ada + 逐个 upload
    std::vector<std::vector<float>> ada;  // 延迟计算
    std::vector<float> fin_sc(DIM), fin_sh(DIM);
    bool ada_computed = false;
    auto ensure_ada = [&]() {
        if (ada_computed) return;
        int nlayer_ada = single_loaded_ ? single_nlayers_ : NUM_LAYERS;
        ada = compute_ada(emb, nlayer_ada);
        for (int d = 0; d < DIM; d++) {
            // [FIX 2026-09-10] emb 6 槽布局：out 层(attn 组) scale=emb[6d+1], shift=emb[6d+0]。
            // 旧代码 emb[d*3+1/0] 6 槽下错位（奇 d 取到 mlp 组）→ 64 维输出 rel≈40% → 图像糊。
            fin_sc[d] = emb[d * 6 + 1] + Wvoa_sc[d];
            fin_sh[d] = emb[d * 6 + 0] + Wvoa_s[d];
        }
        ada_computed = true;
    };

    std::vector<float> vid_cur = vid_patch, txt_cur = txt;
    int nb, chunk;
    if (single_loaded_) { nb = 1; chunk = single_nlayers_; }   // 单 Net：整图 1 个 Net，无块间
    else { nb = gblock_run_; chunk = gblock_chunk_; }
    bool persistent = graph_persistent_ && (precision_ != 0) && !getenv("SEEDVR_GRAPH_RELEASE");
    if (single_loaded_) persistent = true;   // 单 Net 10GB 权重必须常驻（本任务核心：消除块间 CPU 往返）
    for (int b = 0; b < nb; b++)
    {
        int l0 = b * chunk, l1 = l0 + chunk;
        auto t_blk0 = std::chrono::high_resolution_clock::now();
        ncnn::Net* bn;
        if (single_loaded_) {
            bn = single_net_.get();
            if (b == 0) inject_graph_geometry(0);   // 单 Net 含 32 个 AWA，一次注入全部
        } else {
            // ⚠️ 上一块 Net 在「其 ex/cmd 已析构后」才释放：Extractor 持有 Net 内部对象引用，
            // 若 Net 先析构（块末 reset）而 ex 后析构（块作用域末）-> 野指针 -> pool destroyed too early。
            // 故 reset 挪到下一轮开头（上一轮 ex/cmd 已在块末析构）。
            if (b > 0 && !persistent) gblocks_[b - 1].reset();
            if (!graph_load_block(b)) { fprintf(stderr, "[graph] block %d 加载失败\n", b); return false; }
            inject_graph_geometry(b);   // 块加载后必须注入（否则 fallback win bin 窗口错乱 -> 黑图）
            bn = gblocks_[b].get();
        }
        ncnn::VkCompute cmd(vkdev);
        ncnn::Extractor ex = bn->create_extractor();
        // FIX: ncnn 的 blob allocator 会把分块图的所有 Input blob（vid/txt/ada 向量）复用同一 buffer，
        // 导致后 upload 的覆盖先 upload 的（vid_patch 被 txt/ada 覆盖）-> 第 0 层就错。
        // 解法：每个 upload 用【独立的 VkBlobAllocator 实例】，从物理上隔离不同 blob 的 buffer，
        // 使 ncnn 无法跨 Input 复用。缓存按 (block, upload序号) 索引，跨帧复用。
        if ((int)up_alloc_.size() <= b) up_alloc_.resize(b + 1);
        auto& blk_alloc = up_alloc_[b];
        if (!blk_alloc.empty()) {
            for (auto* a : blk_alloc) a->clear();   // 释放上一帧 buffer，复用 allocator 实例
        }
        up_seq_ = 0;
        auto upload = [&](const char* name, const std::vector<float>& data, int dim, int h) {
            ncnn::Mat m; m.create(dim, h, (size_t)4u, 1);
            memcpy(m.data, data.data(), data.size() * sizeof(float));
            ncnn::VkMat vk;
            // 独立 allocator（每 (block,seq) 一个实例），绝不与其它 Input 复用 buffer
            if (blk_alloc.size() <= (size_t)up_seq_) blk_alloc.resize(up_seq_ + 1, nullptr);
            if (!blk_alloc[up_seq_]) blk_alloc[up_seq_] = new ncnn::VkBlobAllocator(vkdev);
            ncnn::VkAllocator* ia = blk_alloc[up_seq_];
            up_seq_++;
            ncnn::Option uopt = bn->opt;
            uopt.blob_vkallocator = ia;
            cmd.record_upload(m, vk, uopt);
            ex.input(name, vk);
            if (getenv("SEEDVR_DUMP_ADABUF") && vk.buffer()) {
                fprintf(stderr, "[adabuf] %s -> ptr=%p off=%zu\n", name, (void*)vk.buffer(), (size_t)vk.offset);
            }
            // DIAG: 验证 in_vid0 上传到 GPU 后的值（检查是否被转 bf16 或丢失）
            if (getenv("SEEDVR_DIAG_IN0") && strcmp(name, "in_vid0") == 0) {
                ncnn::VkCompute dcmd(vkdev);
                ncnn::Mat dv;
                ncnn::Option do2 = bn->opt; do2.use_packing_layout = false;
                dcmd.record_download(vk, dv, do2);
                dcmd.submit_and_wait(); dcmd.reset();
                float mn=1e30f,mx=-1e30f; const float* hp=(const float*)dv.data;
                for (size_t q=0;q<(size_t)dv.w*dv.h;q++){ float v=hp[q]; if(v<mn)mn=v; if(v>mx)mx=v; }
                fprintf(stderr, "[diag_in0] elemsize=%zu w=%d h=%d range[%.4f,%.4f] data0=%.4f\n",
                        (size_t)dv.elemsize, dv.w, dv.h, mn, mx, hp[0]);
            }
        };
        if (b == 0) {
            if (getenv("SEEDVR_DUMP_VPATCH")) {
                FILE* vf = fopen("vidpatch_engine.f32", "wb");
                if (vf) { fwrite(vid_patch.data(), sizeof(float), vid_patch.size(), vf); fclose(vf); }
            }
            upload("in_vid0", vid_patch, 132, Lv);
            upload("in_txt0", txt, 5120, TXT);
        } else {
            char inn[64];
            snprintf(inn, sizeof(inn), "v_cur_%d", l0);
            upload(inn, vid_cur, DIM, Lv);
            snprintf(inn, sizeof(inn), "t_cur_%d", l0);
            upload(inn, txt_cur, DIM, TXT);
        }
        // AdaCompose 图：只 upload emb（1 次），GPU shader 算 384+2 个 ada 向量
        // 旧图：逐个 upload 384 ada + 2 final（CPU compute_ada + 384 次 record_upload）
        bool use_adacompose = false;
        for (const char* nm : bn->input_names())
            if (nm && strcmp(nm, "emb") == 0) { use_adacompose = true; break; }
        if (use_adacompose) {
            // AdaCompose shader 固定按 float[] 读取 emb；低精度 Net 也必须保留 fp32 上传布局。
            ncnn::Mat m_emb; m_emb.create(15360, 1, (size_t)4u, 1);
            memcpy(m_emb.data, emb.data(), emb.size() * sizeof(float));
            ncnn::VkMat vk_emb;
            ncnn::Option emb_opt = bn->opt;
            emb_opt.use_fp16_storage = emb_opt.use_fp16_packed = false;
            emb_opt.use_bf16_storage = emb_opt.use_bf16_packed = false;
            cmd.record_upload(m_emb, vk_emb, emb_opt);
            ex.input("emb", vk_emb);
        } else {
            ensure_ada();
            if (getenv("SEEDVR_DUMP_ADA_CPU") && b == 0) {
                // ada[0] = 层0 vid 流 a_sc；ada[1] = 层0 vid 流 a_sh...
                fprintf(stderr, "[ada_cpu] ada[0..5][0:4] = ");
                for (int k = 0; k < 6; k++) {
                    fprintf(stderr, "[");
                    for (int j = 0; j < 4; j++) fprintf(stderr, "%.4f ", ada[k][j]);
                    fprintf(stderr, "] ");
                }
                fprintf(stderr, "\n");
                // 层0 vid a_sc (index (0*2+0)*6+0 = 0), a_sh (=1), a_g (=2)
                float mn=1e30f, mx=-1e30f; for (auto v : ada[0]) { if(v<mn)mn=v; if(v>mx)mx=v; }
                fprintf(stderr, "[ada_cpu] ada[0](v a_sc) range[%.4f,%.4f]\n", mn, mx);
            }
            for (int i = l0; i < l1; i++)
                for (int f = 0; f < 2; f++) {
                    const char* fname = (f == 0) ? "v" : "t";
                    for (int k = 0; k < 6; k++) {
                        static const char* kn[] = {"a_sc", "a_sh", "a_g", "m_sc", "m_sh", "m_g"};
                        char nm[64]; snprintf(nm, sizeof(nm), "ada_%d_%s_%s", i, fname, kn[k]);
                        upload(nm, ada[(i * 2 + f) * 6 + k], DIM, 1);
                    }
                }
            if (b == nb - 1) {
                upload("ada_fin_sc", fin_sc, DIM, 1);
                upload("ada_fin_sh", fin_sh, DIM, 1);
            }
        }
        if (b == nb - 1) {
            const char* outn = "out0";
            ncnn::Mat m_out;
            // 诊断：SEEDVR_DUMP=blob1:blob2 dump 中间/输出 blob（单 Net 分支）
            const char* dumpenv = getenv("SEEDVR_DUMP");
            if (dumpenv && single_loaded_) {
                std::string dls = dumpenv;
                std::string tok;
                std::stringstream ss(dls);
                while (std::getline(ss, tok, ',')) {
                    ncnn::Mat dm;
                    if (precision_ == 0) {
                        ncnn::VkMat dvk;
                        if (ex.extract(tok.c_str(), dvk, cmd) != 0) { fprintf(stderr, "[dump] extract %s FAIL\n", tok.c_str()); break; }
                        ncnn::VkMat dvk1;
                        vkdev->convert_packing(dvk, dvk1, 1, cmd, bn->opt);
                        { ncnn::Option od = bn->opt; od.use_packing_layout = false; cmd.record_download(dvk1, dm, od); }
                        cmd.submit_and_wait();
                        cmd.reset();
                    } else {
                        if (ex.extract(tok.c_str(), dm) != 0) { fprintf(stderr, "[dump] extract %s FAIL\n", tok.c_str()); break; }
                    }
                    FILE* df = fopen(("dump_" + tok + ".f32").c_str(), "wb");
                    if (df) { fwrite(dm.data, 1, dm.total() * (size_t)dm.elemsize, df); fclose(df);
                        fprintf(stderr, "[dump] %s w=%d h=%d c=%d ep=%d elemsize=%zu total=%zu\n",
                                tok.c_str(), dm.w, dm.h, dm.c, dm.elempack, (size_t)dm.elemsize, dm.total());
                    }
                }
            }
            if (precision_ == 0) {
                // fp32：VkMat 连续 + convert_packing + download
                ncnn::VkMat vk_out;
                ex.extract(outn, vk_out, cmd);
                ncnn::VkMat vk_p1;
                vkdev->convert_packing(vk_out, vk_p1, 1, cmd, bn->opt);
                { ncnn::Option od = bn->opt; od.use_packing_layout = false; cmd.record_download(vk_p1, m_out, od); }
                cmd.submit_and_wait();
                cmd.reset();
            } else {
                // bf16/fp16：GPU extract 主路径（与 fp32 同 cmd 连续 record + download；
                // record_download 内置 fp16/bf16 -> fp32 cast）
                ncnn::VkMat vk_out, vk_p1;
                if (ex.extract(outn, vk_out, cmd) != 0) { fprintf(stderr, "[graph] block %d extract %s FAIL\n", b, outn); return false; }
                vkdev->convert_packing(vk_out, vk_p1, 1, cmd, bn->opt);
                { ncnn::Option od = bn->opt; od.use_packing_layout = false; cmd.record_download(vk_p1, m_out, od); }
                cmd.submit_and_wait();
                cmd.reset();
            }
            if (m_out.elempack != 1 || m_out.w != 64 || m_out.h != Lv) {
                fprintf(stderr, "[graph] block %d out dims w=%d h=%d ep=%d\n", b, m_out.w, m_out.h, m_out.elempack);
                return false;
            }
            out_sr.assign((const float*)m_out.data, (const float*)m_out.data + (size_t)Lv * 64);
        } else {
            char outn[64];
            snprintf(outn, sizeof(outn), "v_cur_%d", l1);
            char outn2[64];
            snprintf(outn2, sizeof(outn2), "t_cur_%d", l1);
            ncnn::Mat mv, mt;
            if (precision_ == 0) {
                ncnn::VkMat vk_out, vk_out2;
                ex.extract(outn, vk_out, cmd);
                ex.extract(outn2, vk_out2, cmd);
                ncnn::VkMat vp1, tp1;
                vkdev->convert_packing(vk_out, vp1, 1, cmd, bn->opt);
                vkdev->convert_packing(vk_out2, tp1, 1, cmd, bn->opt);
                { ncnn::Option od = bn->opt; od.use_packing_layout = false; cmd.record_download(vp1, mv, od);
                  cmd.record_download(tp1, mt, od); }
                cmd.submit_and_wait();
                cmd.reset();
            } else {
                auto t_pre = std::chrono::high_resolution_clock::now();
                // 低精度：走与 fp32 相同的 GPU extract 主路径（cmd 连续 record，避免 CPU extract
                // 每次自建 cmd 重 forward 导致 extractor 状态错乱/块间 vid 坏）。
                // record_download 内置 fp16/bf16 -> fp32 cast（convert_packing 的 cast_type_to）。
                ncnn::VkMat vk_out, vk_out2;
                if (ex.extract(outn, vk_out, cmd) != 0) { fprintf(stderr, "[graph] block %d extract %s FAIL\n", b, outn); return false; }
                if (ex.extract(outn2, vk_out2, cmd) != 0) { fprintf(stderr, "[graph] block %d extract %s FAIL\n", b, outn2); return false; }
                { ncnn::Option od = bn->opt; od.use_packing_layout = false; cmd.record_download(vk_out, mv, od);
                  cmd.record_download(vk_out2, mt, od); }
                cmd.submit_and_wait();
                cmd.reset();
                auto t_post = std::chrono::high_resolution_clock::now();
                double extr_ms = std::chrono::duration<double, std::milli>(t_post - t_pre).count();
                double blk_ms = std::chrono::duration<double, std::milli>(t_post - t_blk0).count();
                fprintf(stderr, "[perf] 图块[%d,%d) chunk=%d 总=%7.1fms  extract(GPU+下载)=%7.1fms(%.0f%%)\n",
                        l0, l1, chunk, blk_ms, extr_ms, (extr_ms / blk_ms) * 100.0);
            }
            if (mv.w != DIM || mv.h != Lv) { fprintf(stderr, "[graph] block %d vid out w=%d h=%d\n", b, mv.w, mv.h); return false; }
            vid_cur.assign((const float*)mv.data, (const float*)mv.data + (size_t)Lv * DIM);
            txt_cur.assign((const float*)mt.data, (const float*)mt.data + (size_t)TXT * DIM);
            // DIAG: dump 每块输出 vid_cur 到 block{b}_vid.f32，用于逐层对比 numpy 参考
            if (getenv("SEEDVR_DUMP_BLOCKOUT")) {
                char bfn[64]; snprintf(bfn, sizeof(bfn), "blkout_%d_vid.f32", b);
                FILE* bf = fopen(bfn, "wb");
                if (bf) { fwrite(vid_cur.data(), sizeof(float), vid_cur.size(), bf); fclose(bf); }
            }
            // DIAG: block0 内部中间 blob（定位 ada 调制/qkv/attn/mlp 哪一步错）
            if (b == 0 && getenv("SEEDVR_DUMP_B0MID")) {
                const char* mids[] = {"in_vid0", "v_cur_0", "v_rn_0", "v_m2_0", "v_qkv_0", "v_attn_0", "v_cur_0_a",
                                      "v_rn2_0", "v_m4_0", "v_hs_0", "v_cur_1", "v_cur_1_a", "v_cur_2", "v_cur_2_a", "v_cur_3",
                                      "ada_0_v_a_sc", "ada_0_v_a_sh", "ada_0_v_m_sc", "ada_0_v_m_sh", "ada_0_v_m_g"};
                for (auto* mn2 : mids) {
                    ncnn::Mat dm2;
                    if (ex.extract(mn2, dm2) != 0) { fprintf(stderr, "[b0mid] %s FAIL\n", mn2); continue; }
                    char mf[64]; snprintf(mf, sizeof(mf), "b0mid_%s.f32", mn2);
                    FILE* mfp = fopen(mf, "wb");
                    if (mfp) { fwrite(dm2.data, 1, dm2.total()*(size_t)dm2.elemsize, mfp); fclose(mfp);
                        float mn=1e30f,mx=-1e30f; const float* dd=(const float*)dm2.data;
                        for (size_t q=0;q<(size_t)dm2.w*dm2.h*dm2.c;q++){float v=dd[q]; if(v<mn)mn=v; if(v>mx)mx=v;}
                        fprintf(stderr, "[b0mid] %s w=%d h=%d range[%.4f,%.4f]\n", mn2, dm2.w, dm2.h, mn, mx); }
                    // DIAG: 同 blob GPU extract 对照（与 fp32 主路径一致的 convert_packing+download）
                    if (getenv("SEEDVR_DUMP_B0MID_GPU")) {
                        ncnn::VkCompute gcmd(vkdev);
                        ncnn::VkMat gvk, gvk1;
                        ncnn::Mat gm;
                        bool gok = (ex.extract(mn2, gvk, gcmd) == 0);
                        if (gok) {
                            vkdev->convert_packing(gvk, gvk1, 1, gcmd, bn->opt);
                            { ncnn::Option od = bn->opt; od.use_packing_layout = false; gcmd.record_download(gvk1, gm, od); }
                            gcmd.submit_and_wait(); gcmd.reset();
                            char gf[64]; snprintf(gf, sizeof(gf), "b0mid_g_%s.f32", mn2);
                            FILE* gfp = fopen(gf, "wb");
                            if (gfp) { fwrite(gm.data, 1, gm.total()*(size_t)gm.elemsize, gfp); fclose(gfp);
                                float gmn=1e30f,gmx=-1e30f; const float* gd=(const float*)gm.data;
                                for (size_t q=0;q<(size_t)gm.w*gm.h*gm.c;q++){float v=gd[q]; if(v<gmn)gmn=v; if(v>gmx)gmx=v;}
                                fprintf(stderr, "[b0mid_g] %s w=%d h=%d ep=%d es=%zu range[%.4f,%.4f]\n", mn2, gm.w, gm.h, gm.elempack, (size_t)gm.elemsize, gmn, gmx); }
                        } else fprintf(stderr, "[b0mid_g] %s extract FAIL\n", mn2);
                    }
                }
            }
        }
        // 常驻仅限低精度（bf16/fp16 权重 <16GB 显存，帧间零加载）；fp32 权重 20GB 超显存必须逐块释放。
        // 显存紧张时（1080p 大激活）可设 SEEDVR_GRAPH_RELEASE=1 强制逐块释放。
        // 块释放时机：下一轮循环开头（本块的 ex/cmd 已析构）——见循环头注释。
    }
    if (!single_loaded_ && !persistent) gblocks_[nb - 1].reset();
    return true;
}

// 显式加载单整图 Net（prefix.param/bin，默认 dit_graph=32 层）。引擎与验证程序调用，
// 与 load_graph（分块路径）互斥，便于 A/B 对比。prefix/nlayers 可用于加载小单 Net（如 dit_4l_bf16）
// 验证「合并成单 Net」机制本身数值正确，而不触发 32 层 10GB 权重的 16GB 显存 OOM。
bool DitVk::load_single(const std::string& graph_dir, const std::string& prefix, int nlayers)
{
    if (graph_loaded_) release_graph();
    single_prefix_ = prefix;
    single_nlayers_ = nlayers;
    // 重建干净 Vulkan allocator：前面分块推理（或旧路径）释放的 blob 在 free-list 留下大量
    // 碎片，单 Net 权重需「连续」device 分配，碎片会导致分配失败 -> OOM/SIGSEGV。
    // reset 销毁并重建 VkDevice，allocator 回到初始干净状态（CPU 侧权重/分支常数保留在内存）。
    reset_vulkan_device();
    graph_dir_ = graph_dir;
    if (graph_dir_.empty() || graph_dir_.back() != '/') graph_dir_ += '/';
    if (!load_single_net()) return false;
    graph_loaded_ = true;
    single_loaded_ = true;
    win_injected_ = false;
    fprintf(stderr, "[graph] single-net ready (%d 层合并 1 个 Net, 整图 1 次 download, prefix=%s)\n",
            single_nlayers_, single_prefix_.c_str());
    return true;
}

// dit_graph.cpp — 阶段3：GPU 常驻整图（分块合并计算图）实现
// 把 32 层 DiT 合并为 8 块 × 4 层的 param/bin 整图（export_dit_graph.py 生成），
// 块内 input(VkMat) -> 连续 extract(VkMat) -> download 一次，块间 vid/txt 残差
// download/upload，消除原有逐层 Net 的 2 upload + 2 download CPU 往返。
#include "dit_vk.h"
#include "awa_layer.h"
#include "cast_bf16_layer.h"
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/mat.h>
#include <ncnn/command.h>
#include <cstdio>
#include <cstring>
#include <fstream>

// 与 dit_vk.cpp 一致的架构常量
static const int DIM = 2560;
static const int NUM_LAYERS = 32, MM_LAYERS = 10;

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
    // 低精度按精度选块前缀：fp16 -> dit_block_f16_*，bf16 -> dit_block_bf16_*（内容均为 fp32 权重 bin）
    const char* prefix = (precision_ == 0) ? "dit_block_" : (precision_ == 1 ? "dit_block_f16_" : "dit_block_bf16_");
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
    bn->opt.use_bf16_storage = bf16;
    bn->opt.use_bf16_packed  = bf16;
    bn->opt.use_fp16_storage = f16;
    bn->opt.use_fp16_packed  = f16;
    bn->opt.use_fp16_arithmetic = false;   // fp32 累加（fp16/bf16 累加器会溢出）
    bn->opt.lightmode = false;   // lightmode=true 在 extract 多 blob 时会递归重算崩
    bn->opt.blob_vkallocator = blob_alloc;
    bn->opt.workspace_vkallocator = blob_alloc;
    bn->opt.staging_vkallocator = staging_alloc;
    bn->register_custom_layer("AWA", AwaLayer::creator);
    bn->register_custom_layer("Bf16Cast", Bf16CastLayer::creator);
    if (bn->load_param(pfull.c_str()) != 0) { fprintf(stderr, "[graph] block %d load_param FAIL\n", b); return false; }
    if (bn->load_model(bbin.data()) < 0) { fprintf(stderr, "[graph] block %d load_model FAIL\n", b); return false; }
    gblocks_[b] = std::move(bn);
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
    const char* prefix = (precision_ == 0) ? "dit_block_" : (precision_ == 1 ? "dit_block_f16_" : "dit_block_bf16_");
    for (int b = 0; b < gblock_count_; b++) {
        char pname[64];
        snprintf(pname, sizeof(pname), "%s%d.param", prefix, b);
        std::ifstream fp(graph_dir_ + pname);
        if (!fp) { fprintf(stderr, "[graph] 缺 %s（需先跑 export_dit_graph.py 生成块图；fp16 需 f16 变体）\n", (graph_dir_ + pname).c_str()); return false; }
    }
    graph_loaded_ = true;
    win_injected_ = false;
    fprintf(stderr, "[graph] ready (%d blocks, chunk=%d; 块 Net 按需加载+用完释放)\n", gblock_count_, chunk);
    return true;
}

void DitVk::release_graph()
{
    gblocks_.clear();
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
    if (b < 0 || b >= (int)gblocks_.size() || !gblocks_[b]) return;
    int sig = win_ns.t * 1000000 + win_ns.h * 10000 + win_ns.w * 100 + win_ns.nwin;
    sig = sig * 1000 + win_ns.txt_len + (win_sh.nwin << 8);
    if ((int)gblock_geom_sig_.size() <= b) gblock_geom_sig_.resize(gblock_count_, -1);
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
    for (auto* l : gblocks_[b]->mutable_layers()) {
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
    std::vector<std::vector<float>> ada = compute_ada(emb, NUM_LAYERS);
    std::vector<float> fin_sc(DIM), fin_sh(DIM);
    for (int d = 0; d < DIM; d++) {
        fin_sc[d] = emb[d * 3 + 1] + Wvoa_sc[d];
        fin_sh[d] = emb[d * 3 + 0] + Wvoa_s[d];
    }

    std::vector<float> vid_cur = vid_patch, txt_cur = txt;
    int nb = gblock_count_, chunk = gblock_chunk_;
    // 常驻仅限低精度（bf16/fp16 权重 <16GB 显存，帧间零加载）；fp32 权重 20GB 超显存必须逐块释放。
    // 显存紧张时（1080p 大激活/单图）可设 SEEDVR_GRAPH_RELEASE=1 或 set_graph_persistent(false) 逐块释放。
    bool persistent = graph_persistent_ && (precision_ != 0) && !getenv("SEEDVR_GRAPH_RELEASE");
    for (int b = 0; b < nb; b++)
    {
        int l0 = b * chunk, l1 = l0 + chunk;
        // ⚠️ 上一块的 Net 在「其 ex/cmd 已析构后」才释放：Extractor 持有 Net 内部对象引用，
        // 若 Net 先析构（块末 reset）而 ex 后析构（块作用域末）-> 野指针 -> pool allocator destroyed too early。
        // 故 reset 挪到下一轮开头（上一轮的 ex/cmd 已在块末析构）。
        if (b > 0 && !persistent) gblocks_[b - 1].reset();
        if (!graph_load_block(b)) { fprintf(stderr, "[graph] block %d 加载失败\n", b); return false; }
        inject_graph_geometry(b);   // 块加载后必须注入（否则 fallback win bin 窗口错乱 -> 黑图）
        ncnn::Net* bn = gblocks_[b].get();
        ncnn::VkCompute cmd(vkdev);
        ncnn::Extractor ex = bn->create_extractor();
        auto upload = [&](const char* name, const std::vector<float>& data, int dim, int h) {
            ncnn::Mat m; m.create(dim, h, (size_t)4u, 1);
            memcpy(m.data, data.data(), data.size() * sizeof(float));
            ncnn::VkMat vk;
            cmd.record_upload(m, vk, bn->opt);
            ex.input(name, vk);
        };
        if (b == 0) {
            upload("in_vid0", vid_patch, 132, Lv);
            upload("in_txt0", txt, 5120, TXT);
        } else {
            char inn[64];
            snprintf(inn, sizeof(inn), "v_cur_%d", l0);
            upload(inn, vid_cur, DIM, Lv);
            snprintf(inn, sizeof(inn), "t_cur_%d", l0);
            upload(inn, txt_cur, DIM, TXT);
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
        if (b == nb - 1) {
            const char* outn = "out0";
            ncnn::Mat m_out;
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
                // bf16/fp16：CPU extract（ncnn 自动低16->fp32；VkMat convert_packing 不支持 bf16）
                if (ex.extract(outn, m_out) != 0) { fprintf(stderr, "[graph] block %d extract %s FAIL\n", b, outn); return false; }
                // ⚠️ 必须像 fp32 分支一样清空外部 cmd：否则块末 gblocks_[b].reset() 销毁 Net 时，
                // cmd 里还挂着 record_upload 的 VkMat 引用 -> "pool allocator destroyed too early" 崩
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
                if (ex.extract(outn, mv) != 0) { fprintf(stderr, "[graph] block %d extract %s FAIL\n", b, outn); return false; }
                if (ex.extract(outn2, mt) != 0) { fprintf(stderr, "[graph] block %d extract %s FAIL\n", b, outn2); return false; }
                // 同最后块：清空外部 cmd，避免 reset Net 时 cmd 残留引用 -> allocator 过早销毁崩
                cmd.submit_and_wait();
                cmd.reset();
            }
            if (mv.w != DIM || mv.h != Lv) { fprintf(stderr, "[graph] block %d vid out w=%d h=%d\n", b, mv.w, mv.h); return false; }
            vid_cur.assign((const float*)mv.data, (const float*)mv.data + (size_t)Lv * DIM);
            txt_cur.assign((const float*)mt.data, (const float*)mt.data + (size_t)TXT * DIM);
        }
        // 常驻仅限低精度（bf16/fp16 权重 <16GB 显存，帧间零加载）；fp32 权重 20GB 超显存必须逐块释放。
        // 显存紧张时（1080p 大激活）可设 SEEDVR_GRAPH_RELEASE=1 强制逐块释放。
        // 块释放时机：下一轮循环开头（本块的 ex/cmd 已析构）——见循环头注释。
    }
    if (!persistent) gblocks_[nb - 1].reset();
    return true;
}

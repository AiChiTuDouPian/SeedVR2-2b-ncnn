// test_dit_graph.cpp — 阶段2 验证：整图 dit_graph.param/bin（32 层）vs 现有 forward_latent
// 流程：
//   1. 构造随机 vid patch (Lv,132) / txt (TXT,5120)，timestep
//   2. 算 emb（DitVk::time_embedding）-> ada 参数（emb + branch，与 forward_latent 同公式）
//   3. GPU 整图：注册 AWA 自定义层，input 所有 blob（vid/txt/384 个 ada/2 fin）-> extract out0
//   4. CPU 参考：DitVk::forward_latent 全 32 层 -> sr_out
//   5. cos(out0, sr_out)，期望 1.0
// 用法：seedvr2_test_dit_graph [model_dir] [graph_dir] [Lv] [timestep]
#include "dit_vk.h"
#include "awa_window.h"
#include "awa_layer.h"
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/mat.h>
#include <ncnn/command.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

static const int DIM = 2560, QKV = 7680, MLP = 6912;
static const int NUM_LAYERS = 32, MM_LAYERS = 10;
static const int TXT_N = 58;
static const float EPS = 1e-5f;

static unsigned int g_seed = 2026;
static float frand() { g_seed = g_seed * 1664525u + 1013904223u; return ((g_seed >> 8) & 0xFFFFFF) / (float)0x1000000; }

static double cos_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return dot / sqrt(na * nb);
}

static std::string read_text(const std::string& p) { std::ifstream f(p, std::ios::binary); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }
static std::vector<unsigned char> read_bin(const std::string& p) {
    std::ifstream f(p, std::ios::binary); f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg(); f.seekg(0);
    std::vector<unsigned char> b(n); f.read((char*)b.data(), n); return b;
}
static std::vector<float> read_raw_f(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    int64_t ndim; f.read((char*)&ndim, 8);
    std::vector<int64_t> sh(ndim); for (auto& d : sh) f.read((char*)&d, 8);
    int64_t total = 1; for (auto d : sh) total *= d;
    std::vector<float> v(total); f.read((char*)v.data(), total * 4); return v;
}

static void silu_vec(std::vector<float>& x) { for (auto& v : x) v = v / (1.f + expf(-v)); }
// 与 DitVk::time_embedding 完全一致（lin 是 public）
static std::vector<float> time_embedding(DitVk& dit, float t) {
    auto sinusoidal = [](float tt, int dim) {
        std::vector<float> e(dim); int half = dim / 2;
        for (int j = 0; j < dim; j++) {
            float freq = (j < half) ? (float)exp(-logf(10000.f) * j / half) : (float)exp(-logf(10000.f) * (j - half) / half);
            e[j] = (j < half) ? sinf(tt * freq) : cosf(tt * freq);
        }
        return e;
    };
    std::vector<float> e = sinusoidal(t, 256);
    e = dit.lin("emb_proj_in", e.data(), 1, 256, 2560); silu_vec(e);
    e = dit.lin("emb_proj_hid", e.data(), 1, 2560, 2560); silu_vec(e);
    e = dit.lin("emb_proj_out", e.data(), 1, 2560, 15360);
    return e;
}

int main(int argc, char** argv)
{
    std::string model_dir = argc > 1 ? argv[1] : "models/m5/";
    std::string graph_dir = argc > 2 ? argv[2] : "models/m5_graph/";
    int Lv = argc > 3 ? atoi(argv[3]) : 64;
    if (getenv("SEEDVR_TEST_REALWIN")) {
        int th = getenv("SEEDVR_TEST_TH") ? atoi(getenv("SEEDVR_TEST_TH")) : 23;
        int tw = getenv("SEEDVR_TEST_TW") ? atoi(getenv("SEEDVR_TEST_TW")) : 34;
        Lv = th * tw;   // 真实窗口：token 网格（360p=782 / 1080p=6936）
    }
    float timestep = argc > 4 ? (float)atof(argv[4]) : 0.5f;
    int n_layers = argc > 5 ? atoi(argv[5]) : 1;   // 验证层数（阶段2 标准：1 层）
    if (model_dir.empty() || model_dir.back() != '/') model_dir += '/';
    if (graph_dir.empty() || graph_dir.back() != '/') graph_dir += '/';

    bool ok = false;
    {
    ncnn::VulkanDevice* vkdev = nullptr;
    ncnn::VkAllocator* blob_alloc = nullptr;
    ncnn::VkAllocator* staging_alloc = nullptr;

    // ---- CPU 参考引擎（也负责 create_gpu_instance / emb / branch 权重）----
    DitVk dit;
    if (!dit.init(model_dir, false, 0)) { fprintf(stderr, "FAIL DitVk::init\n"); return 1; }
    vkdev = ncnn::get_gpu_device(0);
    blob_alloc = vkdev->acquire_blob_allocator();
    staging_alloc = vkdev->acquire_staging_allocator();

    // ---- 窗口：合成小窗口（默认）或真实 make_win（SEEDVR_TEST_REALWIN=1，与 engine 同款）----
    const int ROPE_ROT = 126;
    Win wns, wsh;
    bool realwin = (getenv("SEEDVR_TEST_REALWIN") != nullptr);
    if (realwin) {
        // engine 同款：token 网格来自 H8/2 × W8/2（360p=23×34；1080p=68×102），num_windows={4,3,3}
        int th = getenv("SEEDVR_TEST_TH") ? atoi(getenv("SEEDVR_TEST_TH")) : 23;
        int tw = getenv("SEEDVR_TEST_TW") ? atoi(getenv("SEEDVR_TEST_TW")) : 34;
        int nw[3] = {4, 3, 3};
        wns = awa::make_win(1, th, tw, nw, TXT_N, false);
        wsh = awa::make_win(1, th, tw, nw, TXT_N, true);
        fprintf(stderr, "[win] 真实窗口 ns.nwin=%d sh.nwin=%d (Lv=%d)\n", wns.nwin, wsh.nwin, th * tw);
    } else {
        auto make_win = [&](Win& w, bool shifted) {
            w.t = 1; w.h = 8; w.w = 8; w.nwin = 2; w.txt_len = TXT_N;
            w.st = {0, 0}; w.en = {4, 4}; w.sh = {0, 0}; w.eh = {4, 4};
            if (!shifted) { w.sw = {0, 2}; w.ew = {2, 4}; }
            else          { w.sw = {1, 3}; w.ew = {3, 5}; }
            int sumf = 0; for (int wi = 0; wi < 2; wi++) sumf += (w.en[wi]-w.st[wi])*(w.eh[wi]-w.sh[wi])*(w.ew[wi]-w.sw[wi]);
            w.vid_freq.resize((size_t)sumf * ROPE_ROT); for (auto& v : w.vid_freq) v = frand() * 6.2831853f;
            w.txt_freq.resize((size_t)2 * TXT_N * ROPE_ROT); for (auto& v : w.txt_freq) v = frand() * 6.2831853f;
        };
        make_win(wns, false);
        make_win(wsh, true);
        fprintf(stderr, "[win] 注入合成窗口 (ns.nwin=%d sh.nwin=%d)\n", wns.nwin, wsh.nwin);
    }
    dit.set_windows(wns, wsh);   // CPU 参考用注入窗口（覆盖 init 时 load_win 的 3200 网格）

    // Win -> 几何（vidx/cumf/vfreq/tfreq，与 awa_vk.cpp 同公式）
    auto win_to_geom = [&](const Win& w, std::vector<float>& vidx, std::vector<float>& cumf,
                           std::vector<float>& vfreq, std::vector<float>& tfreq) {
        cumf.assign(w.nwin + 1, 0.f);
        vidx.clear();
        for (int wi = 0; wi < w.nwin; wi++) {
            int f_i = (w.en[wi]-w.st[wi]) * (w.eh[wi]-w.sh[wi]) * (w.ew[wi]-w.sw[wi]);
            cumf[wi + 1] = cumf[wi] + f_i;
            for (int lt = w.st[wi]; lt < w.en[wi]; lt++)
                for (int lh = w.sh[wi]; lh < w.eh[wi]; lh++)
                    for (int lw = w.sw[wi]; lw < w.ew[wi]; lw++)
                        vidx.push_back((float)(((lt * w.h) + lh) * w.w + lw));
        }
        vfreq = w.vid_freq;
        tfreq = w.txt_freq;
    };
    std::vector<float> g_vidx, g_cumf, g_vfreq, g_tfreq, g_vidx2, g_cumf2, g_vfreq2, g_tfreq2;
    win_to_geom(wns, g_vidx, g_cumf, g_vfreq, g_tfreq);
    win_to_geom(wsh, g_vidx2, g_cumf2, g_vfreq2, g_tfreq2);

    // ---- 随机输入 ----
    std::vector<float> vid_patch((size_t)Lv * 132), txt_in((size_t)TXT_N * 5120);
    for (auto& v : vid_patch) v = frand() * 2.f - 1.f;
    for (auto& v : txt_in) v = frand() * 2.f - 1.f;

    // ---- CPU 参考：全 32 层 ----
    std::vector<float> vl, tl, sr_ref;
    dit.forward_latent(vid_patch, Lv, txt_in, TXT_N, timestep, 0, n_layers, true, true, vl, tl, sr_ref);
    fprintf(stderr, "[ref] sr_out=%zu\n", sr_ref.size());

    // ---- ada 参数（emb + branch，与 forward_latent 完全同公式）----
    std::vector<float> emb = time_embedding(dit, timestep);   // (15360,)
    std::vector<std::vector<float>> ada(n_layers * 2 * 6, std::vector<float>(DIM, 0.f));
    auto ada_idx = [&](int i, int flow, int k) { return (i * 2 + flow) * 6 + k; };
    for (int i = 0; i < n_layers; i++) {
        bool dual = (i < MM_LAYERS);
        std::string bv = dual ? ("b" + std::to_string(i) + "_vid") : ("b" + std::to_string(i) + "_all");
        std::string bt = dual ? ("b" + std::to_string(i) + "_txt") : ("b" + std::to_string(i) + "_all");
        for (int flow = 0; flow < 2; flow++) {
            const std::string& b = (flow == 0) ? bv : bt;
            std::vector<float> sc_a = read_raw_f(model_dir + b + "_attn_scale.bin");
            std::vector<float> sh_a = read_raw_f(model_dir + b + "_attn_shift.bin");
            std::vector<float> g_a  = read_raw_f(model_dir + b + "_attn_gate.bin");
            std::vector<float> sc_m = read_raw_f(model_dir + b + "_mlp_scale.bin");
            std::vector<float> sh_m = read_raw_f(model_dir + b + "_mlp_shift.bin");
            std::vector<float> g_m  = read_raw_f(model_dir + b + "_mlp_gate.bin");
            for (int d = 0; d < DIM; d++) {
                ada[ada_idx(i, flow, 0)][d] = emb[d * 6 + 1] + sc_a[d];   // a_sc
                ada[ada_idx(i, flow, 1)][d] = emb[d * 6 + 0] + sh_a[d];   // a_sh
                ada[ada_idx(i, flow, 2)][d] = emb[d * 6 + 2] + g_a[d];    // a_g
                ada[ada_idx(i, flow, 3)][d] = emb[d * 6 + 4] + sc_m[d];   // m_sc
                ada[ada_idx(i, flow, 4)][d] = emb[d * 6 + 3] + sh_m[d];   // m_sh
                ada[ada_idx(i, flow, 5)][d] = emb[d * 6 + 5] + g_m[d];    // m_g
            }
        }
    }
    std::vector<float> fin_sc(DIM), fin_sh(DIM);
    {
        std::vector<float> wvo_sc = read_raw_f(model_dir + "vid_out_ada_scale.bin");
        std::vector<float> wvo_sh = read_raw_f(model_dir + "vid_out_ada_shift.bin");
        for (int d = 0; d < DIM; d++) { fin_sc[d] = emb[d * 3 + 1] + wvo_sc[d]; fin_sh[d] = emb[d * 3 + 0] + wvo_sh[d]; }
    }

    // ---- GPU 整图：n_layers==32 走分块连续执行（阶段3），否则单 Net（阶段2 1 层验证）----
    std::vector<float> sr_gpu;
    if (n_layers == NUM_LAYERS) {
        // ============ 分块 GPU 常驻连续执行：8 块 × 4 层 ============
        // 块内 input(VkMat) -> 连续 extract(VkMat) -> download 一次；块间 CPU 残差（vid+txt）
        // CH=4：系统 RAM 17GB，ncnn load_model 每层持有 weight_data+weight_data_packed 双份 fp32，
        // 8 层块 (~5GB×2) 会 OOM 崩（InnerProduct create_pipeline 的 weight_data_packed 分配失败）
        const int CH = getenv("SEEDVR_TEST_CH") ? atoi(getenv("SEEDVR_TEST_CH")) : 4;
        const int NB = NUM_LAYERS / CH;   // 8
        std::vector<float> vid_cur = vid_patch, txt_cur = txt_in;
        int dbg_only = getenv("SEEDVR_DEBUG_BLOCK0") ? 1 : 0;   // 调试：只跑块 0
        for (int b = 0; b < NB; b++) {
            if (dbg_only && b > 0) break;
            int l0 = b * CH, l1 = l0 + CH;
            const char* gprefix = getenv("SEEDVR_GRAPH_BF16") ? "dit_block_bf16_" : (getenv("SEEDVR_GRAPH_F16") ? "dit_block_f16_" : "dit_block_");
            char pname[64], bname[64];
            snprintf(pname, sizeof(pname), "%s%d.param", gprefix, b);
            snprintf(bname, sizeof(bname), "%s%d.bin", gprefix, b);
            ncnn::Net bn;
            bn.set_vulkan_device(vkdev);
            bn.opt.use_vulkan_compute = true;
            bool gf16 = (getenv("SEEDVR_GRAPH_F16") != nullptr);
            bool gbf16 = (getenv("SEEDVR_GRAPH_BF16") != nullptr);
            bn.opt.use_fp16_storage = bn.opt.use_fp16_packed = gf16;
            bn.opt.use_bf16_storage = bn.opt.use_bf16_packed = gbf16;
            bn.opt.use_fp16_arithmetic = false;   // fp16 存储 + fp32 累加
            bn.opt.use_bf16_storage = bn.opt.use_bf16_packed = false;
            bn.opt.lightmode = false;
            bn.opt.blob_vkallocator = blob_alloc;
            bn.opt.workspace_vkallocator = blob_alloc;
            bn.opt.staging_vkallocator = staging_alloc;
            bn.register_custom_layer("AWA", AwaLayer::creator);
            fprintf(stderr, "[chunks] loading block %d (layers %d..%d)...\n", b, l0, l1); fflush(stderr);
            if (bn.load_param((graph_dir + pname).c_str()) != 0) { fprintf(stderr, "FAIL block %d load_param\n", b); return 1; }
            std::vector<unsigned char> bbin = read_bin(graph_dir + bname);
            fprintf(stderr, "[chunks] block %d load_model...\n", b); fflush(stderr);
            if (bn.load_model(bbin.data()) < 0) { fprintf(stderr, "FAIL block %d load_model\n", b); return 1; }
            fprintf(stderr, "[chunks] block %d loaded OK\n", b); fflush(stderr);
            for (auto* l : bn.mutable_layers()) {
                if (l->type == "AWA") {
                    AwaLayer* a = (AwaLayer*)l;
                    if (a->win_type() == 1)
                        a->set_window_geometry(g_vidx2, g_cumf2, g_vfreq2, g_tfreq2, wsh.nwin, Lv, TXT_N);
                    else
                        a->set_window_geometry(g_vidx, g_cumf, g_vfreq, g_tfreq, wns.nwin, Lv, TXT_N);
                }
            }
            ncnn::VkCompute cmd(vkdev);
            ncnn::Extractor ex = bn.create_extractor();
            auto upload = [&](const char* name, const std::vector<float>& data, int dim, int h) {
                ncnn::Mat m; m.create(dim, h, (size_t)4u, 1);
                memcpy(m.data, data.data(), data.size() * sizeof(float));
                ncnn::VkMat vk;
                cmd.record_upload(m, vk, bn.opt);
                ex.input(name, vk);
            };
            if (b == 0) {
                upload("in_vid0", vid_patch, 132, Lv);
                upload("in_txt0", txt_in, 5120, TXT_N);
            } else {
                char inn[64];
                snprintf(inn, sizeof(inn), "v_cur_%d", l0);
                upload(inn, vid_cur, DIM, Lv);
                snprintf(inn, sizeof(inn), "t_cur_%d", l0);
                upload(inn, txt_cur, DIM, TXT_N);
            }
            for (int i = l0; i < l1; i++)
                for (int f = 0; f < 2; f++) {
                    const char* fname = (f == 0) ? "v" : "t";
                    for (int k = 0; k < 6; k++) {
                        static const char* kn[] = {"a_sc", "a_sh", "a_g", "m_sc", "m_sh", "m_g"};
                        char nm[64]; snprintf(nm, sizeof(nm), "ada_%d_%s_%s", i, fname, kn[k]);
                        upload(nm, ada[ada_idx(i, f, k)], DIM, 1);
                    }
                }
            if (b == NB - 1) {
                upload("ada_fin_sc", fin_sc, DIM, 1);
                upload("ada_fin_sh", fin_sh, DIM, 1);
            }
            // 输出：块 0..2 提取 vid/txt 残差（v_cur_{l1}, t_cur_{l1}）；块 3 提取 out0
            ncnn::VkMat vk_out, vk_out2;
            char outn[64];
            if (b == NB - 1) {
                snprintf(outn, sizeof(outn), "out0");
                ex.extract(outn, vk_out, cmd);
                ncnn::VkMat vk_p1;
                vkdev->convert_packing(vk_out, vk_p1, 1, cmd, bn.opt);
                ncnn::Mat m_out;
                { ncnn::Option od = bn.opt; od.use_packing_layout = false; cmd.record_download(vk_p1, m_out, od); }
                cmd.submit_and_wait();
                cmd.reset();
                if (m_out.elempack == 1 && m_out.w == 64 && m_out.h == Lv)
                    sr_gpu.assign((const float*)m_out.data, (const float*)m_out.data + (size_t)Lv * 64);
                else { fprintf(stderr, "FAIL block3 out dims w=%d h=%d ep=%d\n", m_out.w, m_out.h, m_out.elempack); return 1; }
            } else {
                snprintf(outn, sizeof(outn), "v_cur_%d", l1);
                ex.extract(outn, vk_out, cmd);
                char outn2[64];
                snprintf(outn2, sizeof(outn2), "t_cur_%d", l1);
                ex.extract(outn2, vk_out2, cmd);
                ncnn::VkMat vp1, tp1;
                vkdev->convert_packing(vk_out, vp1, 1, cmd, bn.opt);
                vkdev->convert_packing(vk_out2, tp1, 1, cmd, bn.opt);
                ncnn::Mat mv, mt;
                { ncnn::Option od = bn.opt; od.use_packing_layout = false; cmd.record_download(vp1, mv, od);
                  cmd.record_download(tp1, mt, od); }
                cmd.submit_and_wait();
                cmd.reset();
                if (mv.w != DIM || mv.h != Lv) { fprintf(stderr, "FAIL block %d vid out dims w=%d h=%d\n", b, mv.w, mv.h); return 1; }
                vid_cur.assign((const float*)mv.data, (const float*)mv.data + (size_t)Lv * DIM);
                txt_cur.assign((const float*)mt.data, (const float*)mt.data + (size_t)TXT_N * DIM);
            }
            fprintf(stderr, "[chunks] block %d (layers %d..%d) done\n", b, l0, l1);
            bn.clear();   // 释放块 Net 权重+激活显存（下一块再加载）
        }
    } else {
        // ============ 单 Net 整图（1 层验证） ============
        ncnn::Net net;
        net.set_vulkan_device(vkdev);
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_storage = net.opt.use_fp16_packed = (getenv("SEEDVR_GRAPH_F16") != nullptr);
        net.opt.use_bf16_storage = net.opt.use_bf16_packed = (getenv("SEEDVR_GRAPH_BF16") != nullptr);
        net.opt.use_fp16_arithmetic = false;   // fp16 存储 + fp32 累加（fp16 累加器溢出 -> NaN/爆炸）
        net.opt.use_bf16_storage = net.opt.use_bf16_packed = false;
        net.opt.lightmode = false;   // 调试：保留中间 blob（排查递归重算问题）
        net.opt.blob_vkallocator = blob_alloc;
        net.opt.workspace_vkallocator = blob_alloc;   // VkMat extract 无兜底
        net.opt.staging_vkallocator = staging_alloc;
        net.register_custom_layer("AWA", AwaLayer::creator);
        // 用文件加载 param（DataReaderFromStdio/fscanf），绕开 load_param_mem 的 sscanf 字符串解析问题
        const char* gbase = getenv("SEEDVR_GBASE") ? getenv("SEEDVR_GBASE") : ((n_layers == 1) ? "dit_graph_1l" : "dit_graph");
        if (net.load_param((graph_dir + std::string(gbase) + ".param").c_str()) != 0) { fprintf(stderr, "FAIL load_param\n"); return 1; }
        std::vector<unsigned char> bin = read_bin(graph_dir + std::string(gbase) + ".bin");
        if (net.load_model(bin.data()) < 0) { fprintf(stderr, "FAIL load_model\n"); return 1; }
        fprintf(stderr, "[graph] loaded OK\n");

        // ---- 注入窗口几何到每个 AWA 层（偶数层 nonshifted / 奇数层 shifted）----
        for (auto* l : net.mutable_layers()) {
            if (l->type == "AWA") {
                AwaLayer* a = (AwaLayer*)l;
                if (a->win_type() == 1)
                    a->set_window_geometry(g_vidx2, g_cumf2, g_vfreq2, g_tfreq2, wsh.nwin, Lv, TXT_N);
                else
                    a->set_window_geometry(g_vidx, g_cumf, g_vfreq, g_tfreq, wns.nwin, Lv, TXT_N);
            }
        }
        fprintf(stderr, "[graph] AWA 窗口几何注入 OK\n");

        ncnn::VkCompute cmd(vkdev);
        ncnn::Extractor ex = net.create_extractor();
        auto upload_input = [&](const char* name, const std::vector<float>& data, int dim, int h) {
            ncnn::Mat m; m.create(dim, h, (size_t)4u, 1);
            memcpy(m.data, data.data(), data.size() * sizeof(float));
            ncnn::VkMat vk;
            cmd.record_upload(m, vk, net.opt);
            ex.input(name, vk);
        };
        upload_input("in_vid0", vid_patch, 132, Lv);
        upload_input("in_txt0", txt_in, 5120, TXT_N);
        for (int i = 0; i < n_layers; i++) {
            for (int f = 0; f < 2; f++) {
                const char* fname = (f == 0) ? "v" : "t";
                for (int k = 0; k < 6; k++) {
                    static const char* kn[] = {"a_sc", "a_sh", "a_g", "m_sc", "m_sh", "m_g"};
                    char nm[64]; snprintf(nm, sizeof(nm), "ada_%d_%s_%s", i, fname, kn[k]);
                    upload_input(nm, ada[ada_idx(i, f, k)], DIM, 1);
                }
            }
        }
        upload_input("ada_fin_sc", fin_sc, DIM, 1);
        upload_input("ada_fin_sh", fin_sh, DIM, 1);

        ncnn::VkMat vk_out;
        // fp16 调试：独立 cmd 下载中间 blob 定位 NaN
        if (getenv("SEEDVR_GRAPH_F16")) {
            for (const char* bn : {"v_cur_0", "v_qkv_0_f32", "v_attn_0", "v_cur_0_a", "v_cur_1", "v_f2"}) {
                ncnn::VkCompute cm2(vkdev);
                ncnn::Extractor ex2 = net.create_extractor();
                auto up2 = [&](const char* name, const std::vector<float>& data, int dim, int h) {
                    ncnn::Mat m; m.create(dim, h, (size_t)4u, 1);
                    memcpy(m.data, data.data(), data.size() * sizeof(float));
                    ncnn::VkMat vk; cm2.record_upload(m, vk, net.opt); ex2.input(name, vk);
                };
                up2("in_vid0", vid_patch, 132, Lv);
                up2("in_txt0", txt_in, 5120, TXT_N);
                for (int i = 0; i < n_layers; i++) for (int f = 0; f < 2; f++) for (int k = 0; k < 6; k++) {
                    static const char* kn[] = {"a_sc","a_sh","a_g","m_sc","m_sh","m_g"};
                    char nm[64]; snprintf(nm, sizeof(nm), "ada_%d_%s_%s", i, f ? "t" : "v", kn[k]);
                    up2(nm, ada[ada_idx(i, f, k)], DIM, 1);
                }
                ncnn::VkMat vm; int er = ex2.extract(bn, vm, cm2);
                if (er != 0) { fprintf(stderr, "[mid] %s extract FAIL(%d)\n", bn, er); continue; }
                ncnn::VkMat vp; vkdev->convert_packing(vm, vp, 1, cm2, net.opt);
                ncnn::Mat mm; { ncnn::Option od = net.opt; od.use_packing_layout = false; cm2.record_download(vp, mm, od); }
                cm2.submit_and_wait(); cm2.reset();
                const float* d = (const float*)mm.data;
                float mn = 1e30f, mx = -1e30f; int nn = 0;
                size_t n = (size_t)mm.w * mm.h;
                for (size_t k = 0; k < n && k < 8192; k++) { float v = d[k]; if (v != v) nn++; if (v < mn) mn = v; if (v > mx) mx = v; }
                fprintf(stderr, "[mid] %s: w=%d h=%d [0..3]=%.4f %.4f %.4f %.4f nan=%d min=%.4f max=%.4f\n",
                        bn, mm.w, mm.h, d[0], d[1], d[2], d[3], nn, mn, mx);
            }
        }
        ex.extract("out0", vk_out, cmd);
        ncnn::VkMat vk_out_p1;
        vkdev->convert_packing(vk_out, vk_out_p1, 1, cmd, net.opt);
        ncnn::Mat m_out;
        { ncnn::Option opt_dl = net.opt; opt_dl.use_packing_layout = false; cmd.record_download(vk_out_p1, m_out, opt_dl); }
        cmd.submit_and_wait();
        cmd.reset();
        if (m_out.elempack == 1 && m_out.w == 64 && m_out.h == Lv)
            sr_gpu.assign((const float*)m_out.data, (const float*)m_out.data + (size_t)Lv * 64);
        else { fprintf(stderr, "FAIL out dims w=%d h=%d ep=%d\n", m_out.w, m_out.h, m_out.elempack); return 1; }
    }
    // 下载中间值验证 GPU 执行（每个 blob 独立 cmd + submit）
#if 0
    {
        for (const char* bn : {"v_cur_0", "v_m2_0", "v_qkv_0", "v_attn_0", "v_cur_0_a", "v_f1"}) {
            ncnn::VkCompute cm2(vkdev);
            ncnn::Extractor ex2 = net.create_extractor();
            for (auto& kv : input_bufs) { ncnn::VkMat vk; cm2.record_upload(kv.second, vk, net.opt); ex2.input(kv.first.c_str(), vk); }
            ncnn::VkMat vm; ex2.extract(bn, vm, cm2);
            ncnn::VkMat vp; vkdev->convert_packing(vm, vp, 1, cm2, net.opt);
            ncnn::Mat mm; { ncnn::Option od = net.opt; od.use_packing_layout = false; cm2.record_download(vp, mm, od); }
            cm2.submit_and_wait(); cm2.reset();
            fprintf(stderr, "[val] %s: w=%d h=%d [0..3]=%.4f %.4f %.4f %.4f\n", bn, mm.w, mm.h,
                    ((const float*)mm.data)[0], ((const float*)mm.data)[1], ((const float*)mm.data)[2], ((const float*)mm.data)[3]);
            if (std::string(bn) == "v_attn_0") {   // 检查未覆盖 token 是否清零
                const float* d = (const float*)mm.data;
                fprintf(stderr, "[val] v_attn_0 token4[0..3]=%.4f %.4f %.4f %.4f token8[0]=%.4f\n",
                        d[4 * 2560 + 0], d[4 * 2560 + 1], d[4 * 2560 + 2], d[4 * 2560 + 3], d[8 * 2560 + 0]);
            }
        }
    }
#endif

    double c = cos_sim(sr_gpu, sr_ref);
    fprintf(stderr, "\n========== 阶段2/3 整图验证 ==========\n");
    fprintf(stderr, "  整图 %d 层 vs forward_latent  cos = %.10f  (期望 1.0)\n", n_layers, c);
    fprintf(stderr, "  sr_gpu[0..3] = %.6f %.6f %.6f %.6f | sr_ref[0..3] = %.6f %.6f %.6f %.6f\n",
            sr_gpu[0], sr_gpu[1], sr_gpu[2], sr_gpu[3], sr_ref[0], sr_ref[1], sr_ref[2], sr_ref[3]);
    {   // 差异分布：按 token 分段统计 maxdiff
        int ntok = (int)sr_gpu.size() / 64;
        for (int t = 0; t < ntok; t++) {
            float md = 0; size_t mi = 0;
            for (int k = 0; k < 64; k++) {
                float d = fabsf(sr_gpu[t*64+k] - sr_ref[t*64+k]);
                if (d > md) { md = d; mi = (size_t)t*64+k; }
            }
            if (md > 0.02f)
                fprintf(stderr, "  [diff] token %d: maxdiff=%.5f @idx=%zu (gpu=%.5f ref=%.5f)\n",
                        t, md, mi, sr_gpu[mi], sr_ref[mi]);
        }
    }
    ok = (c > 1 - 1e-6);
    fprintf(stderr, "  ==> %s\n", ok ? "PASS ✅" : "FAIL ❌");
    fprintf(stderr, "======================================\n");
    // 作用域结束：net（后声明）先析构释放 AWA pipeline，dit 再析构 destroy_gpu_instance
    }
    return ok ? 0 : 1;
}

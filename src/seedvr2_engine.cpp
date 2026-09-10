// seedvr2_engine.cpp — SeedVR2 单帧超分引擎实现
// 单帧链路（与历史 main.cpp 逐位等价）：resize/pad/normalize -> VAE encode -> 采样
//   -> noise+cond+mask -> DiT(AWA) -> sr_latent -> upscaled=noise-sr -> VAE decode
//   -> LAB 色彩校正 -> 反 normalize + 裁剪
#include "seedvr2_engine.h"
#include "vae_vk.h"
#include "awa_window.h"
#include "image_io.h"
#include "rng.h"
#include "color_fix.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <fstream>
#include <cstdint>

static const float SCALING_FACTOR = 0.9152f;
static const int TXT_LEN = 58;
static const float TIMESTEP = 1000.0f;
static const int NUM_LAYERS = 32;

static bool read_raw(const std::string& path, std::vector<float>& data) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[FAIL] 读 %s\n", path.c_str()); return false; }
    int64_t ndim; f.read((char*)&ndim, 8);
    std::vector<int64_t> shape(ndim);
    for (auto& d : shape) f.read((char*)&d, 8);
    int64_t total = 1; for (auto d : shape) total *= d;
    data.resize(total); f.read((char*)data.data(), total * 4);
    return true;
}

bool SeedVR2Engine::init(const Config& cfg) {
    cfg_ = cfg;

    // ---- 0. 读文本条件 ----
    if (!read_raw(cfg_.modeldir + "/txt.bin", txt_)) return false;
    fprintf(stderr, "[engine] txt %zu 元素\n", txt_.size());

    // ---- DiT 引擎（常驻，跨帧复用）----
    dit_ = std::make_unique<DitVk>();
    if (!dit_->init(cfg_.modeldir, cfg_.fp16_arith, cfg_.precision, cfg_.use_cpu)) return false;
    dit_->set_graph_persistent(cfg_.graph_resident);   // 单图/显存紧张逐块释放；多帧常驻加速
    // SEEDVR_GRAPH_RESIDENT=1：强制块常驻（实验用）。注意单图模式下每块本来就只加载一次，
    // 常驻并不省时间、只多占显存；它的价值在帧序列（跨帧零重载）。此开关用于压测显存边界。
    if (const char* gr = getenv("SEEDVR_GRAPH_RESIDENT")) dit_->set_graph_persistent(atoi(gr) != 0);

    // ---- 阶段3：GPU 合并图 ----
    // RTX 5060 Ti 16GB 无法承载 32 层单 Net 的约 10GB 权重 + staging + 激活 + workspace，
    // 且驱动 OOM 可在 ncnn 返回失败前直接终止进程。因此默认走可用的分块图；
    // 仅显式 SEEDVR_SINGLE_NET=1 时尝试单 Net（用于更大显存卡或实验）。
    // CPU 模式：AWA 自定义 Layer 为 Vulkan-only，无法走合并图，强制 graphdir 为空 -> 走 forward_latent（全 C++ + ncnn CPU GEMM）。
    std::string gdir = cfg_.use_cpu ? std::string("") : cfg_.graphdir;
    if (!gdir.empty()) {
        bool try_single = getenv("SEEDVR_SINGLE_NET") && atoi(getenv("SEEDVR_SINGLE_NET")) != 0;
        if (try_single && dit_->load_single(gdir)) {
            fprintf(stderr, "[engine] 单 Net 加载成功\n");
        } else {
            if (try_single) fprintf(stderr, "[engine] 单 Net 加载失败，回退分块图\n");
            // 块图 chunk（每块层数）：默认 2。实测（1080p bf16，见 bench/PERFORMANCE_ANALYSIS.md）
            // 32 块 → 16 块使 DiT 105.6s → 96.2s（-8.9%，输出与 chunk=1 逐位一致）；
            // 再加大到 chunk=4 无进一步收益（块变大带来的显存/分配压力抵消）。
            // 收益来源只是摊薄「每块固定开销」（cmd/Extractor 创建、窗口几何注入），
            // 与权重加载字节数无关——后者才是 DiT 的最大单项（见 PERFORMANCE_ANALYSIS.md §3）。
            // 低精度若缺 c{N} 块图则自动回退 chunk=1（旧 32 块图）。
            // SEEDVR_GRAPH_CHUNK 可覆盖（配合 SEEDVR_BLOCK_PREFIX 做实验）。
            int gchunk = 2;
            if (const char* gc = getenv("SEEDVR_GRAPH_CHUNK")) gchunk = atoi(gc);
            if (gchunk > 0) fprintf(stderr, "[engine] 图 chunk=%d (prefix=%s)\n", gchunk,
                                    getenv("SEEDVR_BLOCK_PREFIX") ? getenv("SEEDVR_BLOCK_PREFIX") : "(default)");
            if (gchunk > 0 && !dit_->load_graph(gdir, gchunk)) {
                if (gchunk != 1) {
                    fprintf(stderr, "[engine] chunk=%d 块图不可用，回退 chunk=1\n", gchunk);
                    if (!dit_->load_graph(gdir, 1))
                        fprintf(stderr, "[engine] 图加载失败（继续用旧分块路径）\n");
                } else {
                    fprintf(stderr, "[engine] 图加载失败（继续用旧分块路径）\n");
                }
            }
        }
    }

    ready_ = true;
    return true;
}

bool SeedVR2Engine::process(const std::vector<float>& rgb, int W, int H, int frame_idx,
                            std::vector<float>& out_rgb, int& outW, int& outH) {
    if (!ready_) return false;
    prof_.reset();
    int seed = cfg_.seed + frame_idx;
    auto tp0 = std::chrono::high_resolution_clock::now();

    // ---- 1. 预处理：bicubic 短边 resize -> pad16 -> normalize[-1,1] ----
    std::vector<float> resized; int rW, rH;
    img::resize_shortest_edge(rgb.data(), W, H, 3, cfg_.resolution, resized, rW, rH);
    int padW = (rW + 15) / 16 * 16;
    int padH = (rH + 15) / 16 * 16;

    // 转 channel-first (3,padH,padW) + pad 补 0 + normalize [-1,1]
    std::vector<float> x((size_t)3 * padH * padW);
    for (int h = 0; h < padH; h++)
        for (int w = 0; w < padW; w++)
            for (int c = 0; c < 3; c++) {
                float v = (h < rH && w < rW) ? resized[((size_t)h * rW + w) * 3 + c] : 0.0f;
                if (v < 0) v = 0; if (v > 1) v = 1;  // clamp
                x[((size_t)c * padH + h) * padW + w] = (v - 0.5f) / 0.5f;  // [-1,1]
            }

    int H8 = padH / 8, W8 = padW / 8;
    auto tp1 = std::chrono::high_resolution_clock::now();
    prof_.add("1. 预处理 (resize/pad/normalize)", "CPU",
              std::chrono::duration<double>(tp1 - tp0).count() * 1000.0);
    fprintf(stderr, "[perf] 预处理(resize/pad/norm) = %.3f s\n",
            std::chrono::duration<double>(tp1 - tp0).count());

    // ---- 2. VAE encode -> mean/logvar -> 采样（独立作用域，encode 后释放 Vulkan 资源）----
    std::vector<float> latent((size_t)16 * H8 * W8);
    {
        VaeVk vae;
        // VAE 强制 fp32（低精度 bf16 模式下 VAE 析构会 pool allocator 崩；fp32 VAE 1080p 已验证稳定）
        if (!vae.init(cfg_.vaedir, H8, W8, 0, cfg_.use_cpu)) return false;
        ncnn::Mat img_mat(padW, padH, 3);
        memcpy(img_mat.data, x.data(), x.size() * 4);
        ncnn::Mat mean, logvar;
        if (!vae.encode(img_mat, mean, logvar)) return false;
        // 采样 latent = mean + exp(0.5*logvar) * randn（seed+1000000，批量 fill 对齐 PyTorch）
        // [RNG 对齐] 官方 VAE 重参数化 eps 也是 CUDA(Philox)：generation_phases.py set_seed(seed+1000000)
        //   后 VAE .latent 采样 randn_like → 必须用 PhiloxRandn(seed+1000000)，不能用 CPU mt19937！
        //   （mt19937 版本 cond 与官方 cos≈0.9993、残差 1~5% → 细粒度纹理错乱；对齐后应 ≈1）
        rng::PhiloxRandn rngv(seed + 1000000);
        std::vector<float> eps(latent.size());
        rngv.fill(eps.data(), eps.size());
        const float* mp = (const float*)mean.data;
        const float* lp = (const float*)logvar.data;
        const size_t n = latent.size();
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < (long long)n; i++) {
            float std = std::exp(0.5f * lp[i]);
            latent[(size_t)i] = mp[i] + std * eps[i];
        }
    }
    auto tp2 = std::chrono::high_resolution_clock::now();
    prof_.add("2. VAE encode (子图GPU / 采样CPU)", "GPU(Vulkan)+CPU",
              std::chrono::duration<double>(tp2 - tp1).count() * 1000.0);
    fprintf(stderr, "[perf] VAE encode = %.3f s\n",
            std::chrono::duration<double>(tp2 - tp1).count());

    // ---- 3. 构造 vid_grid (H8,W8,33) channel-last HWC访问更快----
    std::vector<float> noise((size_t)16 * H8 * W8);
    // [RNG 对齐] 官方 DiT 噪声 = torch.randn(CUDA, seed) = Philox4x32-10 + curand_normal4（非 mt19937）
    //   → PhiloxRandn 复刻 CUDA normal_kernel 布局，噪声与官方 cos≈0.999999999+（x0 从 0.379 → 0.99995）
    { rng::PhiloxRandn rngn(seed); rngn.fill(noise.data(), noise.size()); }
    std::vector<float> vid_grid((size_t)H8 * W8 * 33);
    for (int h = 0; h < H8; h++)
        for (int w = 0; w < W8; w++) {
            float* vp = &vid_grid[((size_t)h * W8 + w) * 33];
            for (int c = 0; c < 16; c++) {
                vp[c] = noise[((size_t)c * H8 + h) * W8 + w];                        // noise -> channel-last
                vp[16 + c] = latent[((size_t)c * H8 + h) * W8 + w] * SCALING_FACTOR; // cond -> channel-last
            }
            vp[32] = 1.0f;  // mask
        }
    // SEEDVR_DUMP_VIDGRID=1：dump DiT 输入 vid_grid（H8*W8*33 channel-last）到 vid_grid.bin，
    // 用于和 PyTorch 的 DiT 输入对比，确认输入是否一致。
    if (getenv("SEEDVR_DUMP_VIDGRID")) {
        FILE* fp = fopen("vid_grid.bin", "wb");
        if (fp) {
            size_t n = (size_t)H8 * W8 * 33;
            fwrite(&n, sizeof(size_t), 1, fp);
            fwrite(&H8, sizeof(int), 1, fp);
            fwrite(&W8, sizeof(int), 1, fp);
            fwrite(vid_grid.data(), sizeof(float), n, fp);
            fclose(fp);
            fprintf(stderr, "[dump] vid_grid -> vid_grid.bin (%d x %d x 33 = %zu floats)\n", H8, W8, n);
        }
    }

    // [DIAG] SEEDVR_LOAD_X0=<path>：跳过噪声/DiT，直接载入外部 x0 作为 VAE decode 输入
    //   （channel-first [16,H8,W8] 原始 float；容忍 0/16/20B 头）→ 隔离 decode 的模糊/纹理贡献。
    std::vector<float> sr;      // DiT 输出 v（channel-last）；LOAD_X0 时未用
    std::vector<float> z_dec;   // VAE decode 输入（channel-first）
    if (const char* lx = getenv("SEEDVR_LOAD_X0")) {
        size_t expect = (size_t)16 * H8 * W8;
        z_dec.resize(expect);
        FILE* fp = fopen(lx, "rb");
        if (!fp) { fprintf(stderr, "[engine] SEEDVR_LOAD_X0 无法打开 %s\n", lx); return false; }
        fseek(fp, 0, SEEK_END); long sz = ftell(fp); fseek(fp, 0, SEEK_SET);
        long full = (long)(expect * 4);
        if (sz == full) { /* raw */ }
        else if (sz == full + 16) { fseek(fp, 16, SEEK_SET); }
        else if (sz == full + 20) { fseek(fp, 20, SEEK_SET); }
        else {
            fprintf(stderr, "[engine] SEEDVR_LOAD_X0 大小不符: %ld (期望 %ld 或 +16/+20 头)\n", sz, full);
            fclose(fp); return false;
        }
        size_t rd = fread(z_dec.data(), 4, expect, fp); fclose(fp);
        if (rd != expect) { fprintf(stderr, "[engine] SEEDVR_LOAD_X0 读入不足 %zu/%zu\n", rd, expect); return false; }
        fprintf(stderr, "[engine] 载入外部 x0 (%s) [16x%d x %d], 跳过噪声+DiT\n", lx, H8, W8);
    }
    bool use_ext_x0 = (getenv("SEEDVR_LOAD_X0") != nullptr);
    std::chrono::high_resolution_clock::time_point tp3, tp3b;

    if (!use_ext_x0) {
    // ---- 4. DiT（常驻引擎：graph 整图路径 或 分块推理路径）----
    {
        int token_h = H8 / 2, token_w = W8 / 2;
        // SEEDVR_FULLWIN=1：退化 AWA 为全窗口注意力（nwin=1，每窗覆盖整个 latent），
        // 用于定位"格子内部细节错/网格"是否来自窗口划分；默认 0 用多窗口 {4,3,3}
        int num_windows[3] = {4, 3, 3};
        const char* fw = getenv("SEEDVR_FULLWIN");
        if (fw && atoi(fw) != 0) num_windows[0] = num_windows[1] = num_windows[2] = 1;
        Win wns = awa::make_win(1, token_h, token_w, num_windows, TXT_LEN, false);
        Win wsh = awa::make_win(1, token_h, token_w, num_windows, TXT_LEN, true);
        dit_->set_windows(wns, wsh);
        fprintf(stderr, "[engine] DiT: Lv=%d nwin(ns=%d,sh=%d)\n", H8 * W8, wns.nwin, wsh.nwin);

        int Lv = (H8 / 2) * (W8 / 2);  // token 数（2×2 patch 合并）
        std::vector<float> cur_vid = DitVk::patchify_grid(vid_grid, 1, H8, W8);
        std::vector<float> cur_txt = txt_;
        if (dit_->graph_ready()) {
            // 阶段3：GPU 常驻整图（8 块 × 4 层合并计算图），块内零 CPU 往返
            auto tg0 = std::chrono::high_resolution_clock::now();
            std::vector<float> sro_raw;
            if (!dit_->forward_graph(cur_vid, Lv, cur_txt, TXT_LEN, TIMESTEP, sro_raw)) {
                fprintf(stderr, "[engine] DiT graph 推理失败\n"); return false;
            }
            // ★ graph 返回 out0 原始输出 (Lv,64)（patch 空间）；必须 unpatchify 成
            //   (H8*W8,16) channel-last 才能与旧路径 sr 语义一致（否则布局错位 -> 图失真）
            sr = DitVk::unpatchify_latent(sro_raw, 1, H8, W8);
            auto tg1 = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "[perf] DiT graph(32层) = %.3f s\n",
                    std::chrono::duration<double>(tg1 - tg0).count());
        } else {
            int K = 8;
            if (const char* ck = getenv("SEEDVR_CHUNK")) K = atoi(ck);
            int done = 0; bool ok = true;
            while (done < NUM_LAYERS) {
                int l0 = done;
                int l1 = (done + K < NUM_LAYERS) ? done + K : NUM_LAYERS;
                bool do_init = (l0 == 0);
                bool is_final = (l1 >= NUM_LAYERS);
                std::vector<float> vout, tout, sro;
                if (!dit_->forward_latent(cur_vid, Lv, cur_txt, TXT_LEN, TIMESTEP, l0, l1, do_init, is_final, vout, tout, sro)) {
                    ok = false; break;
                }
                if (!sro.empty()) sr = DitVk::unpatchify_latent(sro, 1, H8, W8);
                else {
                    cur_vid = vout; cur_txt = tout;
                    // [DIAG] SEEDVR_DUMP_CHUNKOUT=1：dump chunk 边界 vid 隐藏态 (Lv,DIM) 残差流，
                    //   与官方 nadit 逐 block dump(pt_block{N}_vid.bin) 对拍 → 漂移曲线/结构性 bug 定位
                    if (getenv("SEEDVR_DUMP_CHUNKOUT")) {
                        char fn[64]; snprintf(fn, sizeof(fn), "chunkout_%d.bin", l1);
                        FILE* fp = fopen(fn, "wb");
                        if (fp) {
                            fwrite(&l1, sizeof(int), 1, fp);          // 已处理层数(8/16/24)
                            fwrite(&Lv, sizeof(int), 1, fp);
                            fwrite(vout.data(), sizeof(float), vout.size(), fp);
                            fclose(fp);
                            fprintf(stderr, "[dump] chunk %d vid hidden -> %s  (%d x %zu = %zu floats)\n",
                                    l1, fn, Lv, vout.size() / (size_t)Lv, vout.size());
                        }
                    }
                }
                done = l1;
                if (done < NUM_LAYERS) dit_->reset_vulkan_device();
            }
            if (!ok) { fprintf(stderr, "[engine] DiT 推理失败\n"); return false; }
        }
        if (sr.empty()) { fprintf(stderr, "[engine] 未产出 sr\n"); return false; }
        // SEEDVR_DUMP_SR=1：dump DiT 输出 latent（unpatchify 后，channel-last 16 通道）到 latent_sr.bin，
        // 用于和 PyTorch 的 DiT 输出逐值对比，定位"格子内部细节错"根因。
        if (getenv("SEEDVR_DUMP_SR")) {
            FILE* fp = fopen("latent_sr.bin", "wb");
            if (fp) {
                size_t n = (size_t)H8 * W8 * 16;
                fwrite(&n, sizeof(size_t), 1, fp);
                fwrite(&H8, sizeof(int), 1, fp);
                fwrite(&W8, sizeof(int), 1, fp);
                fwrite(sr.data(), sizeof(float), n, fp);
                fclose(fp);
                fprintf(stderr, "[dump] sr latent -> latent_sr.bin  (%d x %d x 16 = %zu floats)\n", H8, W8, n);
            }
        }
    }
    tp3 = std::chrono::high_resolution_clock::now();
    prof_.add("3. DiT (32层 NaDiT / AWA)", "GPU(Vulkan)",
              std::chrono::duration<double>(tp3 - tp2).count() * 1000.0);
    fprintf(stderr, "[perf] DiT(32层) = %.3f s\n",
            std::chrono::duration<double>(tp3 - tp2).count());

    // ---- 5. upscaled = noise - sr；转 channel-first 再 /0.9152 ----
    z_dec.resize((size_t)16 * H8 * W8);
    for (int h = 0; h < H8; h++)
        for (int w = 0; w < W8; w++)
            for (int c = 0; c < 16; c++) {
                float n = noise[((size_t)c * H8 + h) * W8 + w];
                float s = sr[((size_t)h * W8 + w) * 16 + c];   // sr channel-last
                z_dec[((size_t)c * H8 + h) * W8 + w] = (n - s) / SCALING_FACTOR;
            }
    }   // end if(!use_ext_x0)：噪声+DiT 计算 z_dec

    // ---- 6. VAE decode（独立作用域，decode 后立即拷贝输出再释放）----
    // 常驻模式（graph_resident=true，默认）：分块图跨帧复用，VAE 前不释放，避免每帧退化为慢速旧路径。
    // 仅显存紧张时（graph_resident=false）在 VAE 前主动释放 graph（DiT 权重 ~5.5GB 会挤爆 VAE decode 显存）。
    tp3b = std::chrono::high_resolution_clock::now();
    prof_.add("4. upscaled (z_dec = (noise-sr)/0.9152)", "CPU",
              std::chrono::duration<double>(tp3b - tp3).count() * 1000.0);
    if (dit_->graph_ready() && !cfg_.graph_resident) dit_->release_graph();
    std::vector<float> y_data((size_t)3 * padH * padW);
    {
        VaeVk vae;
        // VAE 强制 fp32（低精度 bf16 模式下 VAE 析构会 pool allocator 崩；fp32 VAE 1080p 已验证稳定）
        if (!vae.init(cfg_.vaedir, H8, W8, 0, cfg_.use_cpu)) return false;
        ncnn::Mat z_mat(W8, H8, 16);
        memcpy(z_mat.data, z_dec.data(), z_dec.size() * 4);
        ncnn::Mat y;
        if (!vae.decode(z_mat, y)) return false;
        memcpy(y_data.data(), y.data, y_data.size() * 4);
    }
    auto tp4 = std::chrono::high_resolution_clock::now();
    prof_.add("5. VAE decode (子图GPU)", "GPU(Vulkan)",
              std::chrono::duration<double>(tp4 - tp3b).count() * 1000.0);
    fprintf(stderr, "[perf] VAE decode = %.3f s\n",
            std::chrono::duration<double>(tp4 - tp3).count());

    // ---- 7. LAB 色彩校正（content=超分结果 style=原始LR）----
    auto tp4b = std::chrono::high_resolution_clock::now();
    if (cfg_.color_fix) {
        colorfix::lab_color_transfer(y_data.data(), x.data(), padH, padW, y_data.data(), 0.8f);
    }
    auto tp4c = std::chrono::high_resolution_clock::now();
    prof_.add("6. LAB色彩校正", cfg_.color_fix ? "CPU" : "CPU(跳过)",
              std::chrono::duration<double>(tp4c - tp4b).count() * 1000.0);

    // ---- 8. 反 normalize + 裁剪 pad ----
    std::vector<float> out_local((size_t)rH * rW * 3);
    for (int h = 0; h < rH; h++)
        for (int w = 0; w < rW; w++)
            for (int c = 0; c < 3; c++) {
                float v = y_data[((size_t)c * padH + h) * padW + w];
                v = v * 0.5f + 0.5f;  // [-1,1] -> [0,1]
                if (v < 0) v = 0; if (v > 1) v = 1;
                out_local[((size_t)h * rW + w) * 3 + c] = v;
            }
    auto tp5 = std::chrono::high_resolution_clock::now();
    prof_.add("7. 反归一化 + 裁剪", "CPU",
              std::chrono::duration<double>(tp5 - tp4c).count() * 1000.0);
    out_rgb = std::move(out_local);
    outW = rW; outH = rH;
    return true;
}

SeedVR2Engine::LoadModeInfo SeedVR2Engine::load_mode_info() const {
    LoadModeInfo info;
    if (!dit_) return info;
    info.single_net = dit_->is_single_net();
    info.graph_ready = dit_->graph_ready();
    info.num_blocks = dit_->get_num_blocks();
    info.block_chunk = dit_->get_block_chunk();
    info.graph_resident = dit_->is_graph_resident();
    return info;
}

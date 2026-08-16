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
#include <cstdio>
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
    if (!dit_->init(cfg_.modeldir, false, cfg_.precision)) return false;

    ready_ = true;
    return true;
}

bool SeedVR2Engine::process(const std::vector<float>& rgb, int W, int H, int frame_idx,
                            std::vector<float>& out_rgb, int& outW, int& outH) {
    if (!ready_) return false;
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
    fprintf(stderr, "[perf] 预处理(resize/pad/norm) = %.3f s\n",
            std::chrono::duration<double>(tp1 - tp0).count());

    // ---- 2. VAE encode -> mean/logvar -> 采样（独立作用域，encode 后释放 Vulkan 资源）----
    std::vector<float> latent((size_t)16 * H8 * W8);
    {
        VaeVk vae;
        if (!vae.init(cfg_.vaedir, H8, W8, cfg_.precision)) return false;
        ncnn::Mat img_mat(padW, padH, 3);
        memcpy(img_mat.data, x.data(), x.size() * 4);
        ncnn::Mat mean, logvar;
        if (!vae.encode(img_mat, mean, logvar)) return false;
        // 采样 latent = mean + exp(0.5*logvar) * randn（seed+1000000，批量 fill 对齐 PyTorch）
        rng::Randn rngv(seed + 1000000);
        std::vector<float> eps(latent.size());
        rngv.fill(eps.data(), eps.size());
        const float* mp = (const float*)mean.data;
        const float* lp = (const float*)logvar.data;
        for (size_t i = 0; i < latent.size(); i++) {
            float std = std::exp(0.5f * lp[i]);
            latent[i] = mp[i] + std * eps[i];
        }
    }
    auto tp2 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[perf] VAE encode = %.3f s\n",
            std::chrono::duration<double>(tp2 - tp1).count());

    // ---- 3. 构造 vid_grid (H8,W8,33) channel-last ----
    std::vector<float> noise((size_t)16 * H8 * W8);
    { rng::Randn rngn(seed); rngn.fill(noise.data(), noise.size()); }
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

    // ---- 4. DiT（常驻引擎，分块推理，块间 reset_vulkan_device）----
    std::vector<float> sr;
    {
        int token_h = H8 / 2, token_w = W8 / 2;
        int num_windows[3] = {4, 3, 3};
        Win wns = awa::make_win(1, token_h, token_w, num_windows, TXT_LEN, false);
        Win wsh = awa::make_win(1, token_h, token_w, num_windows, TXT_LEN, true);
        dit_->set_windows(wns, wsh);
        fprintf(stderr, "[engine] DiT: Lv=%d nwin(ns=%d,sh=%d)\n", H8 * W8, wns.nwin, wsh.nwin);

        int K = 8;
        if (const char* ck = getenv("SEEDVR_CHUNK")) K = atoi(ck);
        int Lv = (H8 / 2) * (W8 / 2);  // token 数（2×2 patch 合并）
        std::vector<float> cur_vid = DitVk::patchify_grid(vid_grid, 1, H8, W8);
        std::vector<float> cur_txt = txt_;
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
            else { cur_vid = vout; cur_txt = tout; }
            done = l1;
            if (done < NUM_LAYERS) dit_->reset_vulkan_device();
        }
        if (!ok) { fprintf(stderr, "[engine] DiT 推理失败\n"); return false; }
        if (sr.empty()) { fprintf(stderr, "[engine] 未产出 sr\n"); return false; }
    }
    auto tp3 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[perf] DiT(32层) = %.3f s\n",
            std::chrono::duration<double>(tp3 - tp2).count());

    // ---- 5. upscaled = noise - sr；转 channel-first 再 /0.9152 ----
    std::vector<float> z_dec((size_t)16 * H8 * W8);
    for (int h = 0; h < H8; h++)
        for (int w = 0; w < W8; w++)
            for (int c = 0; c < 16; c++) {
                float n = noise[((size_t)c * H8 + h) * W8 + w];
                float s = sr[((size_t)h * W8 + w) * 16 + c];   // sr channel-last
                z_dec[((size_t)c * H8 + h) * W8 + w] = (n - s) / SCALING_FACTOR;
            }

    // ---- 6. VAE decode（独立作用域，decode 后立即拷贝输出再释放）----
    std::vector<float> y_data((size_t)3 * padH * padW);
    {
        VaeVk vae;
        if (!vae.init(cfg_.vaedir, H8, W8, cfg_.precision)) return false;
        ncnn::Mat z_mat(W8, H8, 16);
        memcpy(z_mat.data, z_dec.data(), z_dec.size() * 4);
        ncnn::Mat y;
        if (!vae.decode(z_mat, y)) return false;
        memcpy(y_data.data(), y.data, y_data.size() * 4);
    }
    auto tp4 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[perf] VAE decode = %.3f s\n",
            std::chrono::duration<double>(tp4 - tp3).count());

    // ---- 7. LAB 色彩校正（content=超分结果 style=原始LR）----
    if (cfg_.color_fix) {
        colorfix::lab_color_transfer(y_data.data(), x.data(), padH, padW, y_data.data(), 0.8f);
    }

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
    out_rgb = std::move(out_local);
    outW = rW; outH = rH;
    return true;
}

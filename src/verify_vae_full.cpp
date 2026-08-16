// verify_vae_full.cpp — VAE 完整 encode/decode 验证（VaeVk vs PyTorch 2D 参考）
// 用法: seedvr2_verify_vae_full <modeldir> <refdir> [--cpu]
#include "vae_vk.h"
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

static std::vector<float> read_raw(const std::string& path, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[FAIL] 打开 %s\n", path.c_str()); exit(1); }
    int64_t ndim; f.read((char*)&ndim, 8); shape.resize(ndim);
    for (auto& d : shape) f.read((char*)&d, 8);
    int64_t total = 1; for (auto d : shape) total *= d;
    std::vector<float> data(total); f.read((char*)data.data(), total * 4);
    return data;
}

static void write_raw(const std::string& path, const std::vector<int64_t>& shape, const float* data) {
    std::ofstream f(path, std::ios::binary);
    int64_t ndim = (int64_t)shape.size(); f.write((char*)&ndim, 8);
    for (auto d : shape) f.write((char*)&d, 8);
    int64_t total = 1; for (auto d : shape) total *= d;
    f.write((char*)data, total * 4);
}

static ncnn::Mat to_mat(const std::vector<float>& data, int C, int H, int W) {
    ncnn::Mat m(W, H, C);
    memcpy(m.data, data.data(), data.size() * 4);
    return m;
}

static std::vector<float> from_mat(const ncnn::Mat& m) {
    std::vector<float> d(m.c * m.h * m.w);
    memcpy(d.data(), m.data, d.size() * 4);
    return d;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "用法: %s <modeldir> <refdir> [--cpu]\n", argv[0]); return 1; }
    std::string modeldir = argv[1], refdir = argv[2];
    bool cpu = false;
    for (int i = 3; i < argc; i++) if (std::string(argv[i]) == "--cpu") cpu = true;
    if (cpu) {
        // VaeVk 硬编码 use_vulkan_compute=true，这里仅 Vulkan 验证。
        fprintf(stderr, "[warn] VaeVk 仅 Vulkan 模式\n");
    }

    std::vector<int64_t> sh;
    auto in = read_raw(refdir + "/in.bin", sh);       // (3,H,W)
    int C = (int)sh[0], H = (int)sh[1], W = (int)sh[2];
    auto mean_ref = read_raw(refdir + "/mean.bin", sh);     // (16,H/8,W/8)
    auto logvar_ref = read_raw(refdir + "/logvar.bin", sh);
    auto latent_ref = read_raw(refdir + "/latent.bin", sh); // (16,H/8,W/8)
    auto out_ref = read_raw(refdir + "/out.bin", sh);       // (3,H,W)

    VaeVk vae;
    if (!vae.init(modeldir, H / 8, W / 8)) { fprintf(stderr, "[FAIL] VaeVk init\n"); return 1; }

    // ---- encode ----
    ncnn::Mat img = to_mat(in, C, H, W);
    ncnn::Mat mean, logvar;
    if (!vae.encode(img, mean, logvar)) { fprintf(stderr, "[FAIL] encode\n"); return 1; }
    write_raw(refdir + "/mean_vk.bin", {(int64_t)16, (int64_t)(H / 8), (int64_t)(W / 8)}, (const float*)mean.data);
    write_raw(refdir + "/logvar_vk.bin", {(int64_t)16, (int64_t)(H / 8), (int64_t)(W / 8)}, (const float*)logvar.data);
    fprintf(stderr, "[verify] encode: mean shape (%d,%d,%d)\n", mean.w, mean.h, mean.c);

    // ---- decode ----
    ncnn::Mat lat = to_mat(latent_ref, 16, H / 8, W / 8);
    ncnn::Mat out;
    if (!vae.decode(lat, out)) { fprintf(stderr, "[FAIL] decode\n"); return 1; }
    write_raw(refdir + "/out_vk.bin", {(int64_t)3, (int64_t)H, (int64_t)W}, (const float*)out.data);
    fprintf(stderr, "[verify] decode: out shape (%d,%d,%d)\n", out.w, out.h, out.c);
    return 0;
}

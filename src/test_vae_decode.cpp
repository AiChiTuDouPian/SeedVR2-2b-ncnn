// test_vae_decode.cpp — VAE decode 隔离验证：用 ncnn VaeVk 解码"官方 DiT 输出的 x0 latent"，
// 与官方自身 decode 图逐像素对比 → 判定图像差异来自 DiT 累加 还是 VAE decode。
// 用法: seedvr2_test_vae_decode <modeldir> <latent_pt.bin> <out.png> [trueH] [trueW] [scale]
//   latent_pt.bin: 官方 dump 头(4B rank + rank*4B dims) + f32 [1,H8,W8,16] channel-last（已 scale）
//   输出: 反归一化并裁剪 trueH x trueW 的 PNG [0,1]
#include "vae_vk.h"
#include "image_io.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> read_latent(const std::string& path, int& H8, int& W8, int& C, float& scale) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "[FAIL] open %s\n", path.c_str()); exit(1); }
    unsigned int rank = 0;
    fread(&rank, 4, 1, f);
    std::vector<int> dims(rank);
    fread(dims.data(), 4, rank, f);
    long long total = 1;
    for (auto d : dims) total *= d;
    std::vector<float> data(total);
    fread(data.data(), 4, total, f);
    fclose(f);
    fprintf(stderr, "[info] latent rank=%u dims=", rank);
    for (auto d : dims) fprintf(stderr, "%d ", d);
    fprintf(stderr, "\n");
    // 期望 [1,H8,W8,16] channel-last（灵活取最后三维）
    int nd = (int)dims.size();
    if (nd == 4) { H8 = dims[nd-3]; W8 = dims[nd-2]; C = dims[nd-1]; }
    else if (nd == 3) { H8 = dims[0]; W8 = dims[1]; C = dims[2]; }
    else { fprintf(stderr, "[FAIL] 不支持的 dims\n"); exit(1); }
    // 转 channel-first [C,H8,W8] 并除以 scale（对齐官方 decode latent = x0/scale）
    std::vector<float> z((size_t)C * H8 * W8);
    for (int h = 0; h < H8; h++)
        for (int w = 0; w < W8; w++)
            for (int c = 0; c < C; c++)
                z[((size_t)c * H8 + h) * W8 + w] = data[((size_t)h * W8 + w) * C + c] / scale;
    return z;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "用法: %s <modeldir> <latent.bin> <out.png> [trueH] [trueW] [scale]\n", argv[0]); return 1; }
    std::string modeldir = argv[1];
    std::string latf = argv[2];
    std::string outf = argv[3];
    int trueH = 0, trueW = 0;
    float scale = 0.9152f;
    if (argc >= 5) trueH = atoi(argv[4]);
    if (argc >= 6) trueW = atoi(argv[5]);
    if (argc >= 7) scale = (float)atof(argv[6]);

    int H8, W8, C;
    auto z = read_latent(latf, H8, W8, C, scale);
    int padH = H8 * 8, padW = W8 * 8;
    if (trueH <= 0) trueH = padH;
    if (trueW <= 0) trueW = padW;

    VaeVk vae;
    if (!vae.init(modeldir, H8, W8, 0, false)) { fprintf(stderr, "[FAIL] VaeVk init\n"); return 1; }
    ncnn::Mat zmat(W8, H8, C);
    memcpy(zmat.data, z.data(), z.size() * 4);
    ncnn::Mat y;
    if (!vae.decode(zmat, y)) { fprintf(stderr, "[FAIL] decode\n"); return 1; }
    fprintf(stderr, "[info] decode out %dx%dx%d\n", y.w, y.h, y.c);

    std::vector<float> rgb((size_t)trueH * trueW * 3);
    for (int h = 0; h < trueH; h++)
        for (int w = 0; w < trueW; w++)
            for (int c = 0; c < 3; c++) {
                float v = ((const float*)y.data)[((size_t)c * padH + h) * padW + w];
                v = v * 0.5f + 0.5f;
                if (v < 0) v = 0; if (v > 1) v = 1;
                rgb[((size_t)h * trueW + w) * 3 + c] = v;
            }
    if (!img::save(outf, rgb.data(), trueW, trueH)) { fprintf(stderr, "[FAIL] save png\n"); return 1; }
    fprintf(stderr, "[ok] %s (%dx%d)\n", outf.c_str(), trueW, trueH);
    return 0;
}

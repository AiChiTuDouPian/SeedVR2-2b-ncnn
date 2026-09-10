// validate_png.cpp — 校验输出 PNG 是否为有效图像（无 NaN、有合理动态范围）
#include "image_io.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <img.png>\n", argv[0]); return 1; }
    std::vector<float> rgb; int W, H;
    if (!img::load(argv[1], rgb, W, H)) return 1;
    const size_t n = rgb.size();
    if (n != (size_t)W * H * 3) { fprintf(stderr, "[FAIL] size mismatch\n"); return 1; }

    float mn[3] = {1e9f,1e9f,1e9f}, mx[3] = {-1e9f,-1e9f,-1e9f};
    double sum[3] = {0,0,0};
    int nan_count = 0;
    for (size_t i = 0; i < n; i++) {
        float v = rgb[i];
        if (std::isnan(v) || std::isinf(v)) { nan_count++; continue; }
        int c = (int)(i % 3);
        if (v < mn[c]) mn[c] = v;
        if (v > mx[c]) mx[c] = v;
        sum[c] += v;
    }
    printf("[validate] %s  %dx%d  pixels=%zu\n", argv[1], W, H, n/3);
    const char* ch[] = {"R","G","B"};
    for (int c = 0; c < 3; c++) {
        double mean = sum[c] / (n/3);
        printf("  %s: min=%.4f max=%.4f mean=%.4f\n", ch[c], mn[c], mx[c], mean);
    }
    printf("[validate] NaN/Inf count = %d\n", nan_count);
    if (nan_count > 0) { printf("[RESULT] FAIL (含 NaN/Inf)\n"); return 1; }
    bool ok = true;
    for (int c = 0; c < 3; c++) {
        if (mx[c] - mn[c] < 0.02f) ok = false;      // 近乎全黑/全白
        if (mn[c] < -0.01f || mx[c] > 1.01f) ok = false;
    }
    printf("[RESULT] %s\n", ok ? "PASS (有效图像，动态范围正常)" : "WARN (动态范围异常，可能输出异常)");
    return ok ? 0 : 2;
}

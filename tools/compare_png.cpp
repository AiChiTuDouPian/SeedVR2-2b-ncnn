// compare_png.cpp — 两张同尺寸 RGB[0,1] 图像的数值对拍：PSNR / 余弦相似度 / maxdiff / meandiff / NaN
#include "image_io.h"
#include <cstdio>
#include <cmath>
#include <vector>

static int load(const char* path, std::vector<float>& rgb, int& W, int& H) {
    if (!img::load(path, rgb, W, H)) return 1;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <a.png> <b.png>\n", argv[0]); return 1; }
    std::vector<float> a, b; int Wa, Ha, Wb, Hb;
    if (load(argv[1], a, Wa, Ha)) return 1;
    if (load(argv[2], b, Wb, Hb)) return 1;
    if (Wa != Wb || Ha != Hb || a.size() != b.size()) {
        fprintf(stderr, "[FAIL] size mismatch: %dx%d vs %dx%d\n", Wa, Ha, Wb, Hb);
        return 1;
    }
    const size_t n = a.size();
    int nanA = 0, nanB = 0;
    double sse = 0, sa = 0, sb = 0, dot = 0, na = 0, nb = 0;
    double maxd = 0;
    double sumd = 0;
    for (size_t i = 0; i < n; i++) {
        float va = a[i], vb = b[i];
        if (std::isnan(va) || std::isinf(va)) nanA++;
        if (std::isnan(vb) || std::isinf(vb)) nanB++;
        double d = (double)va - (double)vb;
        sse += d * d;
        sumd += std::fabs(d);
        if (std::fabs(d) > maxd) maxd = std::fabs(d);
        sa += va; sb += vb;
        dot += (double)va * (double)vb;
        na += (double)va * va; nb += (double)vb * vb;
    }
    double mse = sse / n;
    double psnr = (mse > 0) ? (10.0 * log10(1.0 / mse)) : 99.0;  // 值域 [0,1]
    double denom = sqrt(na) * sqrt(nb);
    double cos = (denom > 0) ? (dot / denom) : 0.0;
    double meand = sumd / n;
    printf("[compare] %s  vs  %s\n", argv[1], argv[2]);
    printf("  尺寸=%dx%d  像素=%zu\n", Wa, Ha, n/3);
    printf("  MSE=%.3e  PSNR=%.2f dB  cosine=%.8f\n", mse, psnr, cos);
    printf("  max|diff|=%.4f  mean|diff|=%.6f\n", maxd, meand);
    printf("  NaN/Inf: A=%d B=%d\n", nanA, nanB);
    bool ok = (nanA == 0 && nanB == 0 && cos > 0.999 && psnr > 30);
    printf("[RESULT] %s\n", ok ? "PASS (两路径数值一致)" : "WARN/FAIL (差异偏大，需排查)");
    return ok ? 0 : 2;
}

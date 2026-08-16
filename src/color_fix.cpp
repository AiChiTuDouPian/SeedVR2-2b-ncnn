// color_fix.cpp — LAB 色彩校正（忠实移植 ComfyUI color_fix.py::lab_color_transfer）
#include "color_fix.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace colorfix {

// ---- 小波分解（dilation Gaussian blur）----
static void wavelet_blur(const float* src, float* dst, int W, int H, int radius) {
    const float K[3][3] = {
        {0.0625f, 0.125f, 0.0625f},
        {0.125f,  0.25f,  0.125f},
        {0.0625f, 0.125f, 0.0625f},
    };
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float acc = 0.0f;
            for (int ky = 0; ky < 3; ky++) {
                for (int kx = 0; kx < 3; kx++) {
                    int yy = y + (ky - 1) * radius;
                    int xx = x + (kx - 1) * radius;
                    yy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);  // replicate
                    xx = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
                    acc += K[ky][kx] * src[yy * W + xx];
                }
            }
            dst[y * W + x] = acc;
        }
    }
}

// 单通道小波分解：levels=5, radius=2^i，返回 high_freq（细节）和 low_freq（最低频）
static void wavelet_decompose(const float* src, int W, int H,
                              std::vector<float>& high_freq, std::vector<float>& low_freq) {
    const int n = W * H;
    std::vector<float> cur(src, src + n), blur(n);
    high_freq.assign(n, 0.0f);
    for (int i = 0; i < 5; i++) {
        int radius = 1 << i;  // 1,2,4,8,16
        wavelet_blur(cur.data(), blur.data(), W, H, radius);
        for (int j = 0; j < n; j++) high_freq[j] += cur[j] - blur[j];
        cur = blur;
    }
    low_freq = cur;
}

// 小波重构：content 的高频细节 + style 的低频颜色，clamp [-1,1]
static void wavelet_reconstruct(const float* content, const float* style, int W, int H,
                                float* result) {
    const int n = W * H;
    std::vector<float> ch, cl, sh, sl;
    wavelet_decompose(content, W, H, ch, cl);
    wavelet_decompose(style, W, H, sh, sl);
    for (int j = 0; j < n; j++) {
        float v = ch[j] + sl[j];
        result[j] = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
    }
}

// ---- RGB <-> LAB ----
static const float RGB2XYZ[3][3] = {
    {0.4124564f, 0.3575761f, 0.1804375f},
    {0.2126729f, 0.7151522f, 0.0721750f},
    {0.0193339f, 0.1191920f, 0.9503041f},
};
static const float XYZ2RGB[3][3] = {
    { 3.2404542f, -1.5371385f, -0.4985314f},
    {-0.9692660f,  1.8760108f,  0.0415560f},
    { 0.0556434f, -0.2040259f,  1.0572252f},
};

// 单像素 RGB[0,1] -> LAB
static void rgb2lab(float r, float g, float b, float& L, float& a, float& bb) {
    // sRGB 线性化
    auto lin = [](float c) { return c > 0.04045f ? std::pow((c + 0.055f) / 1.055f, 2.4f) : c / 12.92f; };
    float rl = lin(r), gl = lin(g), bl = lin(b);
    // RGB -> XYZ
    float X = RGB2XYZ[0][0] * rl + RGB2XYZ[0][1] * gl + RGB2XYZ[0][2] * bl;
    float Y = RGB2XYZ[1][0] * rl + RGB2XYZ[1][1] * gl + RGB2XYZ[1][2] * bl;
    float Z = RGB2XYZ[2][0] * rl + RGB2XYZ[2][1] * gl + RGB2XYZ[2][2] * bl;
    // D65 白点归一化
    X /= 0.95047f; Z /= 1.08883f;
    // XYZ -> LAB
    const float eps = 6.0f / 29.0f;
    const float eps3 = eps * eps * eps;
    const float kappa = (29.0f / 3.0f) * (29.0f / 3.0f) * (29.0f / 3.0f);
    auto f = [&](float t) { return t > eps3 ? std::pow(t, 1.0f / 3.0f) : (t * kappa + 16.0f) / 116.0f; };
    float fx = f(X), fy = f(Y), fz = f(Z);
    L = fy * 116.0f - 16.0f;
    a = (fx - fy) * 500.0f;
    bb = (fy - fz) * 200.0f;
}

// 单像素 LAB -> RGB[0,1]
static void lab2rgb(float L, float a, float b, float& r, float& g, float& bl) {
    const float eps = 6.0f / 29.0f;
    const float kappa = (29.0f / 3.0f) * (29.0f / 3.0f) * (29.0f / 3.0f);
    float fy = (L + 16.0f) / 116.0f;
    float fx = a / 500.0f + fy;
    float fz = fy - b / 200.0f;
    auto inv = [&](float t) { return t > eps ? t * t * t : (t * 116.0f - 16.0f) / kappa; };
    float X = inv(fx) * 0.95047f;
    float Y = inv(fy);
    float Z = inv(fz) * 1.08883f;
    // XYZ -> RGB
    float rl = XYZ2RGB[0][0] * X + XYZ2RGB[0][1] * Y + XYZ2RGB[0][2] * Z;
    float gl = XYZ2RGB[1][0] * X + XYZ2RGB[1][1] * Y + XYZ2RGB[1][2] * Z;
    float bl_ = XYZ2RGB[2][0] * X + XYZ2RGB[2][1] * Y + XYZ2RGB[2][2] * Z;
    // 逆 gamma
    auto delin = [](float c) {
        return c > 0.0031308f ? std::pow(c < 0 ? 0 : c, 1.0f / 2.4f) * 1.055f - 0.055f : c * 12.92f;
    };
    r = delin(rl); g = delin(gl); bl = delin(bl_);
    r = r < 0 ? 0 : (r > 1 ? 1 : r);
    g = g < 0 ? 0 : (g > 1 ? 1 : g);
    bl = bl < 0 ? 0 : (bl > 1 ? 1 : bl);
}

// 直方图匹配（source 通道对齐到 reference 通道，n_source==n_reference）
static void histogram_match(const float* src, const float* ref, int n, float* matched) {
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int x, int y) { return src[x] < src[y]; });
    std::vector<int> rank(n);
    for (int i = 0; i < n; i++) rank[order[i]] = i;
    std::vector<float> ref_sorted(ref, ref + n);
    std::sort(ref_sorted.begin(), ref_sorted.end());
    for (int j = 0; j < n; j++) matched[j] = ref_sorted[rank[j]];
}

void lab_color_transfer(const float* content, const float* style, int H, int W,
                        float* result, float luminance_weight) {
    const int n = H * W;

    // ---- step 1: 小波重构（content 细节 + style 颜色）----
    std::vector<float> base(3 * n);
    for (int c = 0; c < 3; c++)
        wavelet_reconstruct(content + c * n, style + c * n, W, H, base.data() + c * n);

    // ---- step 2: 转 [0,1] ----
    auto to01 = [&](const float* src) {
        std::vector<float> v(3 * n);
        for (int i = 0; i < 3 * n; i++) {
            float t = (src[i] + 1.0f) * 0.5f;
            v[i] = t < 0 ? 0 : (t > 1 ? 1 : t);
        }
        return v;
    };
    std::vector<float> content01 = to01(base.data());
    std::vector<float> style01 = to01(style);

    // ---- step 3: RGB -> LAB ----
    std::vector<float> content_lab(3 * n), style_lab(3 * n);
    for (int i = 0; i < n; i++) {
        float L, a, b;
        rgb2lab(content01[i], content01[n + i], content01[2 * n + i], L, a, b);
        content_lab[i] = L; content_lab[n + i] = a; content_lab[2 * n + i] = b;
        rgb2lab(style01[i], style01[n + i], style01[2 * n + i], L, a, b);
        style_lab[i] = L; style_lab[n + i] = a; style_lab[2 * n + i] = b;
    }

    // ---- step 4: a*/b* 直方图匹配；L* 加权混合 ----
    std::vector<float> matched_a(n), matched_b(n), matched_L(n);
    histogram_match(content_lab.data() + n, style_lab.data() + n, n, matched_a.data());       // a*
    histogram_match(content_lab.data() + 2 * n, style_lab.data() + 2 * n, n, matched_b.data()); // b*
    histogram_match(content_lab.data(), style_lab.data(), n, matched_L.data());               // L*

    std::vector<float> result_lab(3 * n);
    for (int i = 0; i < n; i++) {
        result_lab[i] = content_lab[i] * luminance_weight + matched_L[i] * (1.0f - luminance_weight);
        result_lab[n + i] = matched_a[i];
        result_lab[2 * n + i] = matched_b[i];
    }

    // ---- step 5: LAB -> RGB ----
    std::vector<float> result_rgb(3 * n);
    for (int i = 0; i < n; i++) {
        lab2rgb(result_lab[i], result_lab[n + i], result_lab[2 * n + i],
                result_rgb[i], result_rgb[n + i], result_rgb[2 * n + i]);
    }

    // ---- step 6: 转 [-1,1] ----
    for (int i = 0; i < 3 * n; i++) result[i] = result_rgb[i] * 2.0f - 1.0f;
}

} // namespace colorfix

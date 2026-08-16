// image_io.cpp — 图像读写 + 预处理
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "image_io.h"

namespace img {

bool load(const std::string& path, std::vector<float>& rgb, int& W, int& H) {
    int w = 0, h = 0, n = 0;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &n, 3);  // 强制 RGB
    if (!data) {
        fprintf(stderr, "[img] 读取 %s 失败: %s\n", path.c_str(), stbi_failure_reason());
        return false;
    }
    W = w; H = h;
    rgb.resize((size_t)w * h * 3);
    for (size_t i = 0; i < rgb.size(); i++) rgb[i] = data[i] / 255.0f;
    stbi_image_free(data);
    return true;
}

bool save(const std::string& path, const float* rgb, int W, int H) {
    std::vector<unsigned char> buf((size_t)W * H * 3);
    for (size_t i = 0; i < buf.size(); i++) {
        float v = rgb[i];
        v = v < 0 ? 0 : (v > 1 ? 1 : v);
        buf[i] = (unsigned char)(v * 255.0f + 0.5f);
    }
    return stbi_write_png(path.c_str(), W, H, 3, buf.data(), W * 3) != 0;
}

// PIL BICUBIC 核（a = -0.5）
static inline float cubic(float x) {
    const float a = -0.5f;
    x = std::fabs(x);
    if (x <= 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
    if (x < 2.0f)  return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
    return 0.0f;
}

void resize_bicubic(const float* src, int srcW, int srcH, int c,
                    float* dst, int dstW, int dstH) {
    const float sx = (float)srcW / dstW;
    const float sy = (float)srcH / dstH;
    for (int dy = 0; dy < dstH; dy++) {
        float fy = (dy + 0.5f) * sy - 0.5f;
        int y0 = (int)std::floor(fy);
        float wy[4];
        for (int k = 0; k < 4; k++) wy[k] = cubic(fy - (y0 - 1 + k));
        for (int dx = 0; dx < dstW; dx++) {
            float fx = (dx + 0.5f) * sx - 0.5f;
            int x0 = (int)std::floor(fx);
            float wx[4];
            for (int k = 0; k < 4; k++) wx[k] = cubic(fx - (x0 - 1 + k));
            for (int ch = 0; ch < c; ch++) {
                float acc = 0.0f;
                for (int ky = 0; ky < 4; ky++) {
                    int yy = y0 - 1 + ky;
                    if (yy < 0) yy = 0;
                    if (yy >= srcH) yy = srcH - 1;
                    for (int kx = 0; kx < 4; kx++) {
                        int xx = x0 - 1 + kx;
                        if (xx < 0) xx = 0;
                        if (xx >= srcW) xx = srcW - 1;
                        acc += src[((size_t)yy * srcW + xx) * c + ch] * wx[kx] * wy[ky];
                    }
                }
                dst[((size_t)dy * dstW + dx) * c + ch] = acc;
            }
        }
    }
}

void resize_shortest_edge(const float* src, int W, int H, int c, int size,
                          std::vector<float>& dst, int& outW, int& outH) {
    if (std::min(H, W) == size) {  // 已等于目标短边
        dst.assign(src, src + (size_t)W * H * c);
        outW = W; outH = H;
        return;
    }
    double scale = (double)size / std::min(H, W);
    int newH = (int)std::round(H * scale);
    int newW = (int)std::round(W * scale);
    dst.resize((size_t)newW * newH * c);
    resize_bicubic(src, W, H, c, dst.data(), newW, newH);
    outW = newW; outH = newH;
}

} // namespace img

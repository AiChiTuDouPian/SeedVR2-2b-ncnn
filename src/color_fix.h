// color_fix.h — LAB 色彩校正（移植 ComfyUI src/utils/color_fix.py 的 lab_color_transfer）
#pragma once
#include <vector>

namespace colorfix {

// LAB 色彩校正：把 content（超分结果）的颜色分布对齐到 style（原始 LR）的颜色。
// content/style 都是 channel-first (3, H, W)，值 [-1,1]。返回同样布局 [-1,1]。
// luminance_weight: 0=完全匹配颜色，1=完全保留细节（默认 0.8）。
void lab_color_transfer(const float* content, const float* style, int H, int W,
                        float* result, float luminance_weight = 0.8f);

} // namespace colorfix

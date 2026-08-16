// image_io.h — 图像读写 + 预处理（bicubic resize / pad / normalize）
#pragma once
#include <vector>
#include <string>

namespace img {

// 读图 -> RGB float [0,1]（连续内存 RGBRGB...，H*W*3）
bool load(const std::string& path, std::vector<float>& rgb, int& W, int& H);

// 写 RGB float [0,1] -> PNG
bool save(const std::string& path, const float* rgb, int W, int H);

// bicubic resize（half-pixel 对齐，PIL BICUBIC a=-0.5）
void resize_bicubic(const float* src, int srcW, int srcH, int c,
                    float* dst, int dstW, int dstH);

// 短边缩放到 size（保持比例，round）
void resize_shortest_edge(const float* src, int W, int H, int c, int size,
                          std::vector<float>& dst, int& outW, int& outH);

} // namespace img

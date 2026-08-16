// verify_colorfix.cpp — 验证 C++ LAB 校正对齐 PyTorch lab_color_transfer
// 用法: seedvr2_verify_colorfix <in.bin> <out.bin>  (in: content+style 拼接 (2,3,H,W))
#include "color_fix.h"
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

static bool read_raw(const std::string& path, std::vector<int64_t>& shape, std::vector<float>& data) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[FAIL] 读 %s\n", path.c_str()); return false; }
    int64_t ndim; f.read((char*)&ndim, 8); shape.resize(ndim);
    for (auto& d : shape) f.read((char*)&d, 8);
    int64_t total = 1; for (auto d : shape) total *= d;
    data.resize(total); f.read((char*)data.data(), total * 4);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "用法: %s <in.bin> <out.bin>\n", argv[0]); return 1; }
    std::vector<int64_t> shape;
    std::vector<float> data;
    if (!read_raw(argv[1], shape, data)) return 1;
    // shape: (2, 3, H, W) = [content(3,H,W), style(3,H,W)]
    int H = (int)shape[2], W = (int)shape[3];
    int n = H * W;
    const float* content = data.data();
    const float* style = data.data() + 3 * n;
    std::vector<float> result(3 * n);
    colorfix::lab_color_transfer(content, style, H, W, result.data(), 0.8f);

    std::ofstream f(argv[2], std::ios::binary);
    int64_t ndim = 3; f.write((char*)&ndim, 8);
    int64_t s3 = 3, sH = H, sW = W;
    f.write((char*)&s3, 8); f.write((char*)&sH, 8); f.write((char*)&sW, 8);
    f.write((char*)result.data(), result.size() * 4);
    fprintf(stderr, "[verify_colorfix] 完成 (%d,%d,%d)\n", 3, H, W);
    return 0;
}

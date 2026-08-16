// verify_rng.cpp — 验证 rng::Randn 与 PyTorch CPU torch.randn 逐位一致
// 用法: seedvr2_verify_rng <seed> <N> <out_raw.bin>
// 输出: N 个 float32（无 header），供 Python 与 torch 逐位对比。
#include "rng.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "用法: %s <seed> <N> <out_raw.bin>\n", argv[0]);
        return 1;
    }
    uint32_t seed = (uint32_t)strtoul(argv[1], nullptr, 10);
    size_t n = (size_t)strtoull(argv[2], nullptr, 10);
    std::vector<float> buf(n);
    rng::Randn rng(seed);
    rng.fill(buf.data(), n);

    std::ofstream f(argv[3], std::ios::binary);
    f.write((const char*)buf.data(), (std::streamsize)(n * sizeof(float)));
    f.close();

    // 打印前 8 个的 hex 便于快速目检
    for (size_t i = 0; i < 8 && i < n; i++) {
        fprintf(stderr, "  [%zu] %08x  %.9g\n", i, rng::f2b(buf[i]), buf[i]);
    }
    fprintf(stderr, "[verify_rng] seed=%u N=%zu -> %s\n", seed, n, argv[3]);
    return 0;
}

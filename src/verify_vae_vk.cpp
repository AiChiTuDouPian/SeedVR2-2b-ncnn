// verify_vae_vk.cpp — VAE 子图 ncnn Vulkan 前向验证
// 用法: seedvr2_verify_vae_vk <param> <bin> <input.bin> <output.bin> [--cpu]
//   input.bin/output.bin: raw 格式 (int64 ndim + int64[] shape + float32[] data)，shape 为 (C,H,W)
//   输入转 ncnn Mat(w=W,h=H,c=C)（channel-major，与 PyTorch (C,H,W) 物理内存一致）。
#include <ncnn/net.h>
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

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "用法: %s <param> <bin> <input.bin> <output.bin> [--cpu]\n", argv[0]); return 1; }
    std::string param = argv[1], bin = argv[2], inpath = argv[3], outpath = argv[4];
    bool cpu = false;
    for (int i = 5; i < argc; i++) if (std::string(argv[i]) == "--cpu") cpu = true;

    std::vector<int64_t> shape;
    auto data = read_raw(inpath, shape);
    fprintf(stderr, "[dbg] 读入 %s ndim=%d\n", inpath.c_str(), (int)shape.size());
    if (shape.size() != 3) { fprintf(stderr, "[FAIL] 输入应为 3 维 (C,H,W)\n"); return 1; }
    int C = (int)shape[0], H = (int)shape[1], W = (int)shape[2];

    ncnn::Net net;
    if (!cpu) net.opt.use_vulkan_compute = true;
    net.opt.use_winograd_convolution = false;  // 关闭 winograd，排除精度差异
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_bf16_storage = false;
    net.opt.use_bf16_packed = false;
    fprintf(stderr, "[dbg] 加载 param...\n");
    if (net.load_param(param.c_str()) != 0) { fprintf(stderr, "[FAIL] load_param\n"); return 1; }
    fprintf(stderr, "[dbg] 加载 model...\n");
    if (net.load_model(bin.c_str()) != 0) { fprintf(stderr, "[FAIL] load_model\n"); return 1; }
    fprintf(stderr, "[dbg] 模型加载完成\n");

    // 输入 (C,H,W) -> ncnn Mat(w=W,h=H,c=C)，channel-major 直接 memcpy
    ncnn::Mat in(W, H, C);
    memcpy(in.data, data.data(), data.size() * 4);
    if (getenv("DUMP_INPUT")) {
        FILE* f = fopen("input_dump.raw", "wb");
        fwrite(in.data, 4, data.size(), f);
        fclose(f);
        fprintf(stderr, "[dbg] dump 输入到 input_dump.raw\n");
    }

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);

    fprintf(stderr, "[verify] %s -> out w=%d h=%d c=%d dims=%d\n", param.c_str(), out.w, out.h, out.c, out.dims);

    // 输出 ncnn Mat(w,h,c) -> raw (c,h,w)
    std::vector<int64_t> oshape;
    std::vector<float> odata;
    if (out.dims == 3) {
        oshape = {(int64_t)out.c, (int64_t)out.h, (int64_t)out.w};
        odata.resize(out.c * out.h * out.w);
        // ncnn channel-major 与 PyTorch (C,H,W) 一致，直接拷贝
        memcpy(odata.data(), out.data, odata.size() * 4);
    } else if (out.dims == 2) {
        oshape = {(int64_t)1, (int64_t)out.h, (int64_t)out.w};
        odata.resize(out.h * out.w);
        memcpy(odata.data(), out.data, odata.size() * 4);
    } else {
        fprintf(stderr, "[FAIL] 输出 dims=%d 未处理\n", out.dims); return 1;
    }
    write_raw(outpath, oshape, odata.data());
    fprintf(stderr, "[verify] 写出 %s\n", outpath.c_str());
    return 0;
}

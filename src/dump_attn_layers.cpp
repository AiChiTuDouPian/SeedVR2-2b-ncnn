// dump_attn_layers.cpp — dump attention 子图各中间 blob（定位哪层出错）
#include <ncnn/net.h>
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

static std::vector<float> read_raw(const std::string& path, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
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
    if (argc < 3) { fprintf(stderr, "用法: %s <param> <bin>\n", argv[0]); return 1; }
    std::string param = argv[1], bin = argv[2];

    ncnn::Net net;
    net.opt.use_winograd_convolution = false;
    net.opt.use_fp16_storage = false; net.opt.use_fp16_packed = false;
    net.opt.use_bf16_storage = false; net.opt.use_bf16_packed = false;
    if (net.load_param(param.c_str()) != 0) { fprintf(stderr, "FAIL load_param\n"); return 1; }
    if (net.load_model(bin.c_str()) != 0) { fprintf(stderr, "FAIL load_model\n"); return 1; }

    std::vector<int64_t> shape;
    auto data = read_raw("e2e_work_vae/attn_in.bin", shape);
    int C = (int)shape[0], H = (int)shape[1], W = (int)shape[2];
    ncnn::Mat in(W, H, C);
    memcpy(in.data, data.data(), data.size() * 4);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    // dump 各中间 blob
    const char* blobs[] = {"b1", "b3", "b4", "b5", "b6", "b7", "b8", "out0"};
    for (auto bname : blobs) {
        ncnn::Mat m;
        int ret = ex.extract(bname, m);
        if (ret != 0) { fprintf(stderr, "extract %s ret=%d\n", bname, ret); continue; }
        std::vector<int64_t> sh;
        std::vector<float> od;
        if (m.dims == 3) { sh = {(int64_t)m.c, (int64_t)m.h, (int64_t)m.w}; od.resize(m.c*m.h*m.w); memcpy(od.data(), m.data, od.size()*4); }
        else if (m.dims == 2) { sh = {(int64_t)1, (int64_t)m.h, (int64_t)m.w}; od.resize(m.h*m.w); memcpy(od.data(), m.data, od.size()*4); }
        else { fprintf(stderr, "%s dims=%d\n", bname, m.dims); continue; }
        std::string path = std::string("e2e_work_vae/layer_") + bname + ".bin";
        write_raw(path, sh, od.data());
        fprintf(stderr, "dump %s: dims=%d w=%d h=%d c=%d\n", bname, m.dims, m.w, m.h, m.c);
    }
    return 0;
}

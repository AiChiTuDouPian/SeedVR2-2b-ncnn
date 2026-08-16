// verify_mha_min.cpp — 最小 MHA 测试（2D 输入 seq×embed）
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

int main(int argc, char** argv) {
    std::string param = argv[1], bin = argv[2], inpath = argv[3], outpath = argv[4];
    ncnn::Net net;
    net.opt.use_winograd_convolution = false;
    net.opt.use_fp16_storage = false; net.opt.use_fp16_packed = false;
    net.opt.use_bf16_storage = false; net.opt.use_bf16_packed = false;
    if (net.load_param(param.c_str()) != 0) { fprintf(stderr, "FAIL load_param\n"); return 1; }
    if (net.load_model(bin.c_str()) != 0) { fprintf(stderr, "FAIL load_model\n"); return 1; }

    std::vector<int64_t> shape;
    auto data = read_raw(inpath, shape);
    int seq = (int)shape[0], embed = (int)shape[1];
    ncnn::Mat in(embed, seq);  // (w=embed, h=seq)
    memcpy(in.data, data.data(), data.size() * 4);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);
    fprintf(stderr, "out dims=%d w=%d h=%d c=%d\n", out.dims, out.w, out.h, out.c);

    std::ofstream f(outpath, std::ios::binary);
    int64_t ndim = 2; f.write((char*)&ndim, 8);
    int64_t h = out.h, w = out.w; f.write((char*)&h, 8); f.write((char*)&w, 8);
    f.write((char*)out.data, (size_t)out.h * out.w * 4);
    return 0;
}

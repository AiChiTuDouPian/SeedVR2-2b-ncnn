// probe_vk.cpp — 最小 Vulkan GEMM 探针：单独加载一个导出的 InnerProduct 层，
// 走 ncnn Vulkan 推理，验证 GPU 通路是否可用（隔离 SIGILL 崩溃）。
#include <ncnn/platform.h>
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

static std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[FAIL] 读 %s\n", path.c_str()); exit(1); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
static std::vector<unsigned char> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[FAIL] 读 %s\n", path.c_str()); exit(1); }
    f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg(); f.seekg(0);
    std::vector<unsigned char> b(n); f.read((char*)b.data(), n); return b;
}

int main(int argc, char** argv) {
    std::string dir = (argc > 1) ? argv[1] : "models/m5/";
    std::string base = (argc > 2) ? argv[2] : "vid_in_proj";
    bool fp16_arith = (argc > 3 && std::string(argv[3]) == "fp16");

    if (ncnn::create_gpu_instance() != 0) { fprintf(stderr, "[FAIL] create_gpu_instance\n"); return 1; }
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(0);
    if (!vkdev) { fprintf(stderr, "[FAIL] get_gpu_device\n"); return 1; }
    fprintf(stderr, "[probe] base=%s fp16_arith=%d\n", base.c_str(), (int)fp16_arith);

    std::string p = read_text(dir + base + ".param");
    std::vector<unsigned char> m = read_bin(dir + base + ".bin");

    ncnn::Net net;
    net.set_vulkan_device(vkdev);
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_arithmetic = fp16_arith;
    net.load_param_mem(p.c_str());
    net.load_model(m.data());
    fprintf(stderr, "[probe] 加载完成\n");

    // 构造一个全 0.01 的输入 token
    int in_dim = 132;
    int out_dim = 2560;
    std::vector<float> x(in_dim, 0.01f);
    ncnn::Mat in(in_dim, 1, (void*)x.data());
    ncnn::Mat out;
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in);
        ex.extract("out0", out);
    }
    fprintf(stderr, "[probe] out w=%d h=%d c=%d\n", out.w, out.h, out.c);
    const float* od = out;
    double s = 0, mx = 0;
    for (int i = 0; i < out_dim; i++) { s += (double)od[i]; if (fabsf(od[i]) > mx) mx = fabsf(od[i]); }
    fprintf(stderr, "[probe] sum=%.6f maxabs=%.6f first5=%.5f %.5f %.5f %.5f %.5f\n",
            s, mx, od[0], od[1], od[2], od[3], od[4]);

    ncnn::destroy_gpu_instance();
    fprintf(stderr, "[probe] OK\n");
    return 0;
}

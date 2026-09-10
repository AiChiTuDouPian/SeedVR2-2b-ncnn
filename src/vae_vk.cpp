// vae_vk.cpp — SeedVR2 video VAE（单帧 2D 等价）ncnn Vulkan 实现
#include "vae_vk.h"
#include <cstdio>
#include <cstring>
#include <vector>

VaeVk::~VaeVk() {
    // ncnn::Net 析构自动释放 Vulkan 资源
}

static void cfg_net(ncnn::Net& net, int precision, bool use_cpu) {
    net.opt.use_vulkan_compute = !use_cpu;   // CPU 模式关闭 Vulkan compute
    net.opt.use_winograd_convolution = false;  // 关 winograd 排除精度差异
    // 存储精度：0=fp32(默认逐位对齐) 1=fp16 2=bf16。bf16 范围同 fp32 不溢出、tensor core 加速。
    net.opt.use_fp16_storage = (precision == 1);
    net.opt.use_fp16_packed  = (precision == 1);
    net.opt.use_bf16_storage = (precision == 2);
    net.opt.use_bf16_packed  = (precision == 2);
}

// 加载固定子图（enc1/enc2/dec1/dec2）
static bool load_sub(ncnn::Net& net, const std::string& param_path, const std::string& bin_path, int precision, bool use_cpu) {
    cfg_net(net, precision, use_cpu);
    if (net.load_param(param_path.c_str()) != 0) {
        fprintf(stderr, "[VaeVk] load_param %s 失败\n", param_path.c_str());
        return false;
    }
    if (net.load_model(bin_path.c_str()) != 0) {
        fprintf(stderr, "[VaeVk] load_model %s 失败\n", bin_path.c_str());
        return false;
    }
    return true;
}

// 动态生成 attention 子图 param（Reshape 的 W/H 依赖 mid 分辨率），加载固定 bin
static bool load_attn(ncnn::Net& net, const std::string& bin_path, const std::string& prefix, int H, int W, int precision, bool use_cpu) {
    cfg_net(net, precision, use_cpu);
    char param[2048];
    snprintf(param, sizeof(param),
             "7767517\n"
             "9 10\n"
             "Input  in0  0 1 in0\n"
             "Split  %s_split  1 2 in0 b1 b2\n"
             "GroupNorm  %s_gn  1 1 b2 b3 0=32 1=512 2=1e-06 3=1\n"
             "Permute  %s_p1  1 1 b3 b4 0=3\n"
             "Reshape  %s_r1  1 1 b4 b5 0=512 1=-1\n"
             "MultiHeadAttention  %s_mha  1 1 b5 b6 0=512 1=1 2=262144 3=512 4=512 5=0\n"
             "Reshape  %s_r2  1 1 b6 b7 0=512 1=%d 2=%d\n"
             "Permute  %s_p2  1 1 b7 b8 0=4\n"
             "BinaryOp  %s_add  2 1 b8 b1 out0 0=0\n",
             prefix.c_str(), prefix.c_str(), prefix.c_str(), prefix.c_str(), prefix.c_str(),
             prefix.c_str(), W, H, prefix.c_str(), prefix.c_str());
    if (net.load_param_mem(param) != 0) {
        fprintf(stderr, "[VaeVk] load_param_mem attention 失败\n");
        return false;
    }
    if (net.load_model(bin_path.c_str()) != 0) {
        fprintf(stderr, "[VaeVk] load_model %s 失败\n", bin_path.c_str());
        return false;
    }
    return true;
}

bool VaeVk::init(const std::string& modeldir, int H_mid, int W_mid, int precision, bool use_cpu) {
    std::string md = modeldir;
    if (md.back() != '/' && md.back() != '\\') md += '/';
    precision_ = precision;
    use_cpu_ = use_cpu;

    fprintf(stderr, "[VaeVk] 加载 enc1... (%s)\n", use_cpu ? "CPU" : "Vulkan"); fflush(stderr);
    if (!load_sub(enc1_, md + "vae_enc1.param", md + "vae_enc1.bin", precision, use_cpu_)) return false;
    fprintf(stderr, "[VaeVk] 加载 enc2...\n"); fflush(stderr);
    if (!load_sub(enc2_, md + "vae_enc2.param", md + "vae_enc2.bin", precision, use_cpu_)) return false;
    fprintf(stderr, "[VaeVk] 加载 dec1...\n"); fflush(stderr);
    if (!load_sub(dec1_, md + "vae_dec1.param", md + "vae_dec1.bin", precision, use_cpu_)) return false;
    fprintf(stderr, "[VaeVk] 加载 dec2...\n"); fflush(stderr);
    if (!load_sub(dec2_, md + "vae_dec2.param", md + "vae_dec2.bin", precision, use_cpu_)) return false;
    fprintf(stderr, "[VaeVk] 加载 attn_enc...\n"); fflush(stderr);
    if (!load_attn(attn_enc_, md + "vae_attn_enc.bin", "vae_attn_enc", H_mid, W_mid, precision, use_cpu_)) return false;
    fprintf(stderr, "[VaeVk] 加载 attn_dec...\n"); fflush(stderr);
    if (!load_attn(attn_dec_, md + "vae_attn_dec.bin", "vae_attn_dec", H_mid, W_mid, precision, use_cpu_)) return false;

    ready_ = true;
    fprintf(stderr, "[VaeVk] init ok (H_mid=%d W_mid=%d)\n", H_mid, W_mid);
    return true;
}

bool VaeVk::encode(const ncnn::Mat& img, ncnn::Mat& mean, ncnn::Mat& logvar) {
    if (!ready_) return false;
    // enc1: img(3ch) -> mid 特征(512ch)
    ncnn::Mat m1;
    {
        ncnn::Extractor ex = enc1_.create_extractor();
        ex.input("in0", img);
        if (ex.extract("out0", m1) != 0) { fprintf(stderr, "[VaeVk] enc1 extract 失败\n"); return false; }
    }
    // attn_enc: mid -> mid（单头注意力 + 残差）
    ncnn::Mat m1a;
    {
        ncnn::Extractor ex = attn_enc_.create_extractor();
        ex.input("in0", m1);
        if (ex.extract("out0", m1a) != 0) { fprintf(stderr, "[VaeVk] attn_enc extract 失败\n"); return false; }
    }
    // enc2: mid -> 32ch（前 16 mean，后 16 logvar）
    ncnn::Mat m2;
    {
        ncnn::Extractor ex = enc2_.create_extractor();
        ex.input("in0", m1a);
        if (ex.extract("out0", m2) != 0) { fprintf(stderr, "[VaeVk] enc2 extract 失败\n"); return false; }
    }
    int W8 = m2.w, H8 = m2.h;  // m2: (w=W8, h=H8, c=32)
    size_t ch_size = (size_t)H8 * W8;
    mean.create(W8, H8, 16);
    logvar.create(W8, H8, 16);
    memcpy(mean.data, m2.channel(0), ch_size * 16 * 4);
    memcpy(logvar.data, m2.channel(16), ch_size * 16 * 4);
    return true;
}

bool VaeVk::decode(const ncnn::Mat& latent, ncnn::Mat& img) {
    if (!ready_) return false;
    // dec1: latent(16ch) -> mid 特征(512ch)
    ncnn::Mat d1;
    {
        ncnn::Extractor ex = dec1_.create_extractor();
        ex.input("in0", latent);
        if (ex.extract("out0", d1) != 0) { fprintf(stderr, "[VaeVk] dec1 extract 失败\n"); return false; }
    }
    // attn_dec
    ncnn::Mat d1a;
    {
        ncnn::Extractor ex = attn_dec_.create_extractor();
        ex.input("in0", d1);
        if (ex.extract("out0", d1a) != 0) { fprintf(stderr, "[VaeVk] attn_dec extract 失败\n"); return false; }
    }
    // dec2: mid -> img(3ch)
    {
        ncnn::Extractor ex = dec2_.create_extractor();
        ex.input("in0", d1a);
        if (ex.extract("out0", img) != 0) { fprintf(stderr, "[VaeVk] dec2 extract 失败\n"); return false; }
    }
    return true;
}

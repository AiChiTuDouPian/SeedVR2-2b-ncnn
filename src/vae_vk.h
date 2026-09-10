// vae_vk.h — SeedVR2 video VAE（单帧 T=1 等价 2D）的 ncnn Vulkan 实现
// 6 个子图（enc1/enc2/dec1/dec2 + attn_enc/attn_dec）已在 ncnn CPU/Vulkan 验证 cos=1.0。
// attention 子图的 Reshape W/H 依赖 mid 分辨率（=输入/8），init 时动态生成 param。
#pragma once
#include <string>
#include <ncnn/net.h>

class VaeVk {
public:
    VaeVk() = default;
    ~VaeVk();

    // modeldir: 含 vae_enc1/enc2/dec1/dec2/attn_enc/attn_dec 的 *.param/*.bin 目录
    // H_mid/W_mid: mid 分辨率 = 输入 H/8、W/8（attention 子图 Reshape 用）
    // precision: 存储精度 0=fp32(默认) 1=fp16 2=bf16
    // use_cpu: true=纯 CPU 推理（关闭 Vulkan compute）
    bool init(const std::string& modeldir, int H_mid, int W_mid, int precision = 0, bool use_cpu = false);

    // encode: img (w=W,h=H,c=3, [0,1] 已归一化到 [-1,1]) -> mean/logvar (w=W/8,h=H/8,c=16)
    // （含 group_norm+silu+conv_out；采样（mean+std*randn）由调用方做）
    bool encode(const ncnn::Mat& img, ncnn::Mat& mean, ncnn::Mat& logvar);

    // decode: latent (w=W/8,h=H/8,c=16) -> img (w=W,h=H,c=3, [-1,1])
    bool decode(const ncnn::Mat& latent, ncnn::Mat& img);

    bool ready() const { return ready_; }

private:
    ncnn::Net enc1_, enc2_, dec1_, dec2_, attn_enc_, attn_dec_;
    int precision_ = 0;   // 0=fp32 1=fp16 2=bf16
    bool use_cpu_ = false;   // true=纯 CPU 推理（关闭 Vulkan compute）
    bool ready_ = false;
};

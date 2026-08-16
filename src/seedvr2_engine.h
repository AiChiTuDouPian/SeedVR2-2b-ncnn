// seedvr2_engine.h — SeedVR2 单帧超分引擎（图片 / 视频帧序列逐帧共用）
//
// 设计要点：
// - DiT（DitVk）常驻：init 一次，跨帧复用。其 CPU 侧权重与 fcache 字节在
//   reset_vulkan_device() 中保留，仅重建 VkDevice，因此多帧间无需重新加载 7.8GB 模型。
// - VAE（VaeVk）不常驻：因为 DiT 分块推理的 reset_vulkan_device() 会销毁并重建全局
//   GPU 实例，VAE 的 ncnn::Net 引用旧 device 会悬空。故 encode / decode 各自保持独立
//   作用域（与历史单帧 main.cpp 完全一致），每帧 encode 前 init、decode 前重新 init。
// - seed = base_seed + frame_idx：图片模式 frame_idx=0，与历史单帧行为逐位一致；
//   视频模式每帧独立 seed，避免跨帧噪声强相关导致闪烁。
#pragma once
#include <string>
#include <vector>
#include <memory>
#include "dit_vk.h"

class SeedVR2Engine {
public:
    struct Config {
        int resolution = 1080;
        std::string modeldir = "models/m5";
        std::string vaedir = "models/m6_vae";
        std::string graphdir = "models/m5_graph";   // 阶段3 GPU 常驻整图（空=禁用，走旧分块路径）
        int seed = 42;              // 基准 seed；逐帧用 seed + frame_idx
        bool color_fix = true;
        int precision = 0;          // 存储精度 0=fp32(默认,逐位对齐) 1=fp16 2=bf16
        bool graph_resident = true; // 低精度图块常驻（多帧加速）；单图/大分辨率显存紧张时 false（逐块释放）
    };

    SeedVR2Engine() = default;

    // 加载文本条件 + DiT 引擎（模型只加载一次）
    bool init(const Config& cfg);

    // 单帧超分：rgb [0,1] HxWx3 (RGBRGB... 连续) -> out_rgb [0,1] outH*outW*3
    // frame_idx：视频帧序号（用于 seed 差异化）；图片传 0。
    bool process(const std::vector<float>& rgb, int W, int H, int frame_idx,
                 std::vector<float>& out_rgb, int& outW, int& outH);

    bool ready() const { return ready_; }

private:
    Config cfg_;
    std::vector<float> txt_;          // 文本条件 (TXT, 5120) 展开
    std::unique_ptr<DitVk> dit_;      // DiT 常驻（唯一持有 GPU 实例，析构时 destroy）
    bool ready_ = false;
};

// seedvr2_pipeline.h — SeedVR2-2b-ncnn 的端到端推理管线封装（类 zimage 的 pipeline）
// 负责：载入 workdir 的网格/文本/窗口 -> 调用 DitVk(Vulkan DiT + 自定义 AWA) -> 写出 SR latent。
// 预处理(图像 resize/normalize) 与 VAE 编解码 暂由 Python 桥接（run_e2e.py），
// 这里只做“纯 C++ 可跑”的 DiT 加速核心，方便后续接任意前端。
#pragma once
#include <string>

namespace seedvr2 {

// 在 workdir 中读取 vid_grid.bin / txt.bin / params.txt / win_ns.bin / win_sh.bin，
// 用 modeldir 的权重跑一次 Vulkan DiT 前向，写出 sr_latent.bin。
// fp16_arith: true=GPU fp16 计算（更快，略损精度）；false=GPU fp32（贴近参考）。
// 返回 0 成功，非 0 失败。
int run_dit(const std::string& workdir, const std::string& modeldir, bool fp16_arith = false);

} // namespace seedvr2

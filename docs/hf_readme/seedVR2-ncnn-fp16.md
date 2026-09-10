---
license: mit
---

# SeedVR2-2b-NCNN · fp16 权重

ByteDance **SeedVR2** 的 NaDiT（2B）扩散 Transformer，在 **ncnn + 自研 Vulkan compute shader** 下的
C++ 推理权重（**float16 计算精度**版）。

配套代码 → [AiChiTuDouPian/SeedVR2-2b-ncnn](https://github.com/AiChiTuDouPian/SeedVR2-2b-ncnn)

## ⚠️ 使用前请注意

本仓库是三种精度里**最不推荐**的一个：

- **速度与 bf16 基本相同**（1080p 下 DiT 105.4 s vs 105.6 s），没有速度收益；
- 但**动态范围不足**（fp16 最大 65504 / 最小正规数 6e-5），大激活处会溢出：
  逐层与 fp32 对比，前 4 层 cos 高达 0.9999998（比 bf16 还准），**第 12 层起跌破 0.996，
  第 30 层只剩 0.90~0.93** —— 是**范围问题而非精度问题**。

⇒ 除非你在做精度对比实验，否则请用
[`seedVR2-ncnn-bf16`](https://huggingface.co/xxzigou/seedVR2-ncnn-bf16)（推荐，cos ≥ 0.9993）
或 [`SeedVR2-2b-NCNN`](https://huggingface.co/xxzigou/SeedVR2-2b-NCNN)（fp32 基准）。

## 文件结构

```
models/
├── m5/                    # DiT 逐层权重（849 文件）+ 自定义 Vulkan shader（*.spv，必需）
├── m6_vae/                # VAE encoder / decoder 权重
└── m5_graph/              # 合并计算图权重块（fp16）：dit_block_f16_0..31.{bin,param}
                           # chunk=1，每块 1 层，共 32 块
```

> `models/m5` 是三种精度**共用**的基础权重；文件名里的精度指**计算精度**而非存储格式
> （`dit_block_f16_*` 的权重确实是 fp16 存储，而 `dit_block_bf16_*` 反而是 fp32 存储）。

## 使用

```bash
git clone https://github.com/AiChiTuDouPian/SeedVR2-2b-ncnn
cd SeedVR2-2b-ncnn

# 下载权重（约 17.5 GB）
hf download xxzigou/seedVR2-ncnn-fp16 --local-dir .

# 推理（--fp16 开启低精度路径）
./build-cmake/seedvr2_run.exe input.png output.png --resolution 1080 --fp16
```

## 性能参考（RTX 5060 Ti 16GB，1080p）

| 配置 | DiT 耗时 | 出图质量（vs fp32） |
|---|---|---|
| fp16 + 合并图 chunk=1（本仓库） | ~105 s | PSNR 31 dB（少数像素高误差） |
| bf16 + 合并图 chunk=1 | ~106 s | PSNR 48 dB |

## 许可

本仓库以 MIT 发布；模型权重源自 ByteDance SeedVR2，请同时遵循其原始许可。

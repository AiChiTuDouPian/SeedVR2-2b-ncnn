---
license: mit
---

# SeedVR2-2b-NCNN · fp32 权重

ByteDance **SeedVR2** 的 NaDiT（2B）扩散 Transformer，在 **ncnn + 自研 Vulkan compute shader** 下的
C++ 推理权重（**fp32 计算精度**版）。

配套代码 → [AiChiTuDouPian/SeedVR2-2b-ncnn](https://github.com/AiChiTuDouPian/SeedVR2-2b-ncnn)
（纯 C++ 端到端推理，自适应窗口注意力 AWA 等算子以自定义 Vulkan compute shader 实现）

## 选哪个仓库

三个仓库**各自自包含**（都含 `models/m5` + `models/m6_vae` + 该精度的块图），下载任一个即可运行：

| 仓库 | 计算精度 | 说明 |
|---|---|---|
| **本仓库** | fp32 | 数值基准，逐位对齐；最慢、显存占用最高 |
| [`seedVR2-ncnn-bf16`](https://huggingface.co/xxzigou/seedVR2-ncnn-bf16) | bf16 | **推荐**：范围安全，逐层 cos ≥ 0.9993，比 fp32 快约 1.7× |
| [`seedVR2-ncnn-fp16`](https://huggingface.co/xxzigou/seedVR2-ncnn-fp16) | fp16 | 最快，但大激活会溢出（第 30 层 cos 仅 ~0.90），不建议出图 |

## 文件结构

```
models/
├── m5/         # DiT 逐层权重（849 个文件）+ 自定义 Vulkan shader（*.spv，必需）
├── m6_vae/     # VAE encoder / decoder 权重
└── m5_graph/   # 合并计算图权重块：dit_block_0..15.{bin,param}（chunk=2，每块 2 层）
```

> `models/m5` 是三种精度**共用**的基础权重（其中 `awa.spv` 等 shader 是自定义算子，缺了跑不起来）；
> 只有 `m5_graph/` 下的块图区分精度。
> ⚠️ 文件名里的 fp32/fp16/bf16 指**计算精度**，不是权重存储格式——`dit_block_*` 的权重实际以 fp16 存储。

## 使用

```bash
git clone https://github.com/AiChiTuDouPian/SeedVR2-2b-ncnn
cd SeedVR2-2b-ncnn

# 下载权重（约 17.5 GB）
hf download xxzigou/SeedVR2-2b-NCNN --local-dir .

# 编译后推理（构建步骤见 GitHub README）
./build-cmake/seedvr2_run.exe input.png output.png --resolution 1080
```

`--resolution` 指定**短边**（按短边等比缩放）。默认路径即
`--modeldir models/m5 --graphdir models/m5_graph --vaedir models/m6_vae`。

## 说明

- fp32 版是**数值基准**：1080p 下与官方 PyTorch 实现对比 PSNR 46.6 dB / cos 0.9999。
- 1080p 时 fp32 合并图路径的显存需求较高（16 GB 卡可能不足），引擎会自动回退到逐层路径；
  低分辨率（360p / 720p）下 fp32 合并图可正常使用。
- 逐层精度表、性能分解、复现命令见 GitHub 仓库的 `README.md` 与 `bench/`。

## 许可

本仓库以 MIT 发布；模型权重源自 ByteDance SeedVR2，请同时遵循其原始许可。

---
license: mit
---

# SeedVR2-2b-NCNN · bf16 权重（推荐）

ByteDance **SeedVR2** 的 NaDiT（2B）扩散 Transformer，在 **ncnn + 自研 Vulkan compute shader** 下的
C++ 推理权重（**bfloat16 计算精度**版）。

配套代码 → [AiChiTuDouPian/SeedVR2-ncnn](https://github.com/AiChiTuDouPian/SeedVR2-ncnn)
（自适应窗口注意力 AWA 等算子以自定义 Vulkan compute shader 实现）

## 为什么推荐 bf16

- **动态范围与 fp32 相同**（8 位指数），不像 fp16 会在大激活处溢出；
- 逐层与 fp32 对比 cos **≥ 0.9993**（几乎无损），1080p 出图 PSNR **45.5 dB**；
- 比 fp32 快约 **1.7×**，且与 fp16 速度相当。

## 文件结构

```
models/
├── m5/                          # DiT 逐层权重（849 文件）+ 自定义 Vulkan shader（*.spv，必需）
├── m6_vae/                      # VAE encoder / decoder 权重
└── m5_graph/                    # 合并计算图权重块（bf16），三种切分粒度：
    ├── dit_block_bf16_c2_0..15     # chunk=2，16 块 —— ★ 引擎默认，只需这一套
    ├── dit_block_bf16_0..31        # chunk=1，32 块 —— 缺 c2 时的自动回退路径
    └── dit_block_bf16_c4_0..7      # chunk=4，8 块 —— 实测无进一步收益，仅作对照
```

> 只下载 `dit_block_bf16_c2_*`（约 9.5 GB）即可运行，其余变体可省。
> `models/m5` 是三种精度**共用**的基础权重；文件名里的精度指**计算精度**而非存储格式
> （`dit_block_bf16_*` 的权重实际是 fp32 存储，`dit_block_f16_*` 才是 fp16 存储）。

## 使用

```bash
git clone https://github.com/AiChiTuDouPian/SeedVR2-ncnn
cd SeedVR2-ncnn

# 只拉必需部分（约 17.5 GB；跳过 c1/c4 变体可再省约 19 GB）
hf download xxzigou/seedVR2-ncnn-bf16 --local-dir . \
  --include "models/m5/*" "models/m6_vae/*" "models/m5_graph/dit_block_bf16_c2_*"

# 推理（--bf16 开启低精度路径）
./build-cmake/seedvr2_run.exe input.png output.png --resolution 1080 --bf16
```

帧序列（视频逐帧超分）：把两个路径参数换成输入/输出**目录**即可，DiT 权重跨帧常驻。

## 性能参考（RTX 5060 Ti 16GB，1080p）

| 配置 | DiT 耗时 |
|---|---|
| bf16 + 合并图 chunk=2（默认） | ~97 s |
| bf16 + 合并图 chunk=1 | ~106 s |

帧序列模式下 720p bf16 可常驻显存，稳态约 **34 s/帧**。

## 许可

本仓库以 MIT 发布；模型权重源自 ByteDance SeedVR2，请同时遵循其原始许可。

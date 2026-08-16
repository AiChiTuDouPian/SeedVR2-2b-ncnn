# SeedVR2 视频超分 ncnn Vulkan 移植 — 技术报告

> 项目路径：`F:/Seedvr2/seedvr2-ncnn`
> 日期：2026-08-15

## 1. 项目概述

本项目将腾讯 **SeedVR2**（NaDiT 架构的视频超分模型，3B 参数）从 PyTorch 移植为**自包含的 C++ 端到端推理程序**，推理后端采用 **ncnn Vulkan**。输入一张低分辨率图像，一条命令输出超分结果，全程无 Python 依赖。

核心难点与成果：

- 把 NaDiT 的 **Adaptive Window Attention（AWA）** 实现为**自定义 Vulkan compute shader**（非 ncnn 标准算子），并与 CPU 参考逐位对齐（cos=1.0）。
- 把 **3D 视频 VAE** 在单帧（T=1）下精确退化为 **2D 卷积网络**，导出为 ncnn 子图，encode/decode 均 cos=1.0。
- 把 **随机数生成（RNG）** 逐位对齐 PyTorch CPU 的 `torch.randn`（mt19937 + cephes Box-Muller）。
- 移植 **LAB 色彩校正** 后处理，cos=0.9996。

## 2. 系统架构

```
输入 LR 图
   │  stb_image 读图
   ▼
预处理（image_io）         bicubic 短边=1080 上采样 → clamp → pad16 → normalize[-1,1]
   │  (3, padH, padW) channel-first
   ▼
VAE encode（vae_vk）        4 个 ncnn 子图 + 2 个 attention → mean/logvar (16, H/8, W/8)
   │
   ▼
采样（rng）                 latent = mean + exp(0.5·logvar) · randn(seed+1e6)
   │
   ▼
构造 vid_grid                [noise(16) + cond·0.9152(16) + mask(1)] = 33 通道 channel-last
   │  noise = randn(seed)
   ▼
DiT 32 层（dit_vk + awa_vk） 每层：qkv 投影 → AWA(Vulkan shader) → mlp_g → mlp_in → mlp_out
   │  线性层用 ncnn Net(GEMM Vulkan)，AWA 用自定义 shader
   ▼
sr_latent (16, H/8, W/8)     unpatchify 到空间
   │
   ▼
upscaled = noise - sr        v_lerp 单步（t/T=1）
   │
   ▼
VAE decode（vae_vk）         /0.9152 → 4 子图 + 2 attention → SR 图 (3, padH, padW)
   │
   ▼
LAB 色彩校正（color_fix）     小波重构 + RGB↔LAB + 直方图匹配（luminance_weight=0.8）
   │
   ▼
反 normalize + 裁 pad + 写图  stb_image_write
```

## 3. 核心模块

| 模块 | 文件 | 职责 |
|---|---|---|
| 主程序 | `src/main.cpp` | 端到端 CLI，编排全链路 |
| DiT 前向 | `src/dit_vk.cpp/.h` | 32 层前向、AWA 调用、分块推理、Net LRU 缓存 |
| AWA shader 封装 | `src/awa_vk.cpp/.h` | 自定义 Vulkan pipeline，上传/绑定/回读 |
| AWA GLSL | `src/shaders/awa.comp` | partition + qk_norm + RoPE + varlen SDPA + unpartition |
| VAE | `src/vae_vk.cpp/.h` | encode/decode，加载 6 个子图 |
| 窗口生成 | `src/awa_window.cpp/.h` | 720P 窗口划分 + 3D mmrope 频率 |
| 图像 IO | `src/image_io.cpp/.h` | stb 读图 + bicubic resize + 写图 |
| 色彩校正 | `src/color_fix.cpp/.h` | LAB 直方图匹配 |
| RNG | `src/rng.h` | 逐位对齐 PyTorch 的 mt19937 + cephes Box-Muller |
| 导出脚本 | `export/export_vae.py` 等 | PyTorch 权重 → ncnn param/bin |
| 参考/验证 | `tools/*.py` | Python 端到端驱动 + 数值对拍 |

## 4. 关键技术点

### 4.1 AWA 自定义 Vulkan shader

SeedVR2 的 NaDiT 使用 Adaptive Window Attention（自适应窗口注意力），ncnn 无对应标准算子，因此手写 GLSL compute shader 实现。单个 shader 内完成：

`partition → qk_norm(RMSNorm) → 3D mmrope RoPE → varlen SDPA(含 txt 重复) → unpartition → coalesce`

关键常量：`HEADS=20, HEAD_D=128, DIM=2560, QKV=7680, ROPE_ROT=126, EPS=1e-5`。

**自定义 flat-float shader 的三个强制约定**（踩坑总结）：

1. shader 必须声明 `layout(local_size_x/y/z = N) in;` 且与 `pipe->set_local_size_xyz(N,N,N)` 一致，否则间歇性 `vkWaitForFences failed -4`。
2. 离散 GPU 默认 fp16 存储，flat-float shader 用 `float[]` 按 4 字节索引会把 half 位模式读成 float，必须关 `use_fp16_storage`/`use_fp16_packed`/`use_bf16_storage`/`use_bf16_packed`。**GEMM（Net 的 linear 层）也必须强制 fp32**，否则真实大激活（Lv=6936）在 fp16 累加溢出 `Inf` 污染整图。
3. **大 buffer 上传必须用 2 维 Mat**（`m.create(size, 1, sizeof(float), 1)`）。ncnn `record_upload` 对 1 维 Mat 无条件 pack4，`dispatcher.h = size/4` 超过 Vulkan `maxComputeWorkGroupCount[y]=65535` 时 pack4 静默失败，大 buffer（如 53M float 的 vqkv）损坏，小 buffer（vidx 1734）正常——这是「单测 PASS、真实图崩」的根因。

### 4.2 VAE 的 2D 等价降维

3D 视频 VAE（`DownEncoderBlock3D×4 + UpDecoderBlock3D×4`，带 causal temporal padding）在**单帧 T=1 时精确退化为纯 2D 卷积网络**：

| 原 3D 结构 | 2D 等价 |
|---|---|
| InflatedCausalConv3d (k_t=3) | Conv2D，核 = `W.sum(dim=2)` |
| InflatedCausalConv3d (k_t=1) | Conv2D，核 = `W[:,:,0]` |
| Downsample3D | Padding(右下1) + Conv2D(stride=2) |
| Upsample3D (temporal_up) | 1×1 Conv(z=0 子集重排) + PixelShuffle(2) + Conv2D |
| GroupNorm / SiLU / 单头 Attention | 原样 2D |

导出为 6 个子图：`vae_enc1/enc2/dec1/dec2`（纯 conv）+ `vae_attn_enc/dec`（GroupNorm+Permute+Reshape+MHA+残差）。

**坑**：mid_block 前向顺序是 `resnet0 → attention → resnet1`（非 attention 在前），写反会让 logvar 输出差 0.43。

### 4.3 RNG 逐位对齐 PyTorch

逆向 torch 2.13 源码，`torch.randn`（CPU）链路为：

1. 引擎 `at::mt19937` ≡ 标准 `std::mt19937`（**非 Philox**，Philox 只用于 CUDA/MPS）。
2. uniform：`u = (uint32 & 0xFFFFFF) * 2⁻²⁴`。
3. Box-Muller（AVX2 cephes 近似，16 元素一批）：`radius = sqrt(-2·cephes_log(u1))`，`theta = float32(6.2831855·u2)`，`out = radius·(cos/sin)(theta)`。
4. 末尾（size%16≠0）：重新生成最后 16 个 uniform **覆盖**。

**坑**：float 的 normal 走 AVX2 SIMD 路径，log/sin/cos 是 cephes 多项式近似，与标量函数差 1~2 ULP；多项式需 `-ffp-contract=off` 防 FMA 融合。

### 4.4 LAB 色彩校正

移植官方 `lab_color_transfer`：5 级 dilation 高斯模糊的小波重构 + RGB↔LAB(D65) + a\*/b\* 通道 CDF 直方图匹配 + L\* 按 0.8 权重混合，消除单步扩散的偏色/颗粒感。

## 5. 数值验证

| 验证项 | 结果 |
|---|---|
| AWA 单层 GPU vs CPU | cos = 1.000000 |
| sr_latent GPU vs CPU | cos = 1.000000 |
| DiT 端到端 vs Python numpy 参考 | cos = 1.000000 |
| DiT 逐层 vs PyTorch fp32 | cos ≥ 0.9977 |
| VAE 6 子图（CPU/Vulkan） | cos = 1.000000 |
| VAE 完整 encode/decode | cos = 1.000000 |
| RNG vs torch.randn（N=439552） | 逐位一致（439552/439552） |
| LAB 校正 vs PyTorch | cos = 0.9996 |
| 端到端图 vs 官方 bf16 | cos = 0.969（官方 bf16 量化噪声，非 bug） |

## 6. 工程结构（清理后）

```
seedvr2-ncnn/
├── CMakeLists.txt             # 构建配置
├── PLAN_AND_PITFALLS.md       # 早期踩坑记录
├── seedvr2_run.exe            # 自包含推理主程序
├── sadhu_vk_*.png             # Sadhu 示例输出（3 张：无校正/校正/最新）
├── src/                       # C++ 源码（约 338K）
│   ├── main.cpp               #   端到端 CLI 入口
│   ├── dit_vk.cpp/.h          #   DiT 前向（32 层）
│   ├── awa_vk.cpp/.h          #   AWA 自定义 shader 封装
│   ├── vae_vk.cpp/.h          #   VAE encode/decode
│   ├── awa_window.cpp/.h      #   窗口生成
│   ├── image_io.cpp/.h        #   图像 IO + 预处理
│   ├── color_fix.cpp/.h       #   LAB 色彩校正
│   ├── rng.h                  #   RNG（对齐 PyTorch）
│   ├── shaders/awa.comp       #   AWA GLSL
│   └── verify_*/test_*/*.cpp  #   验证/测试源码（可重编译）
├── models/                    # 推理权重（约 16G）
│   ├── m5/                    #   DiT ncnn 权重 + awa.spv（带 GPU AWA）
│   ├── m5_nospv/              #   DiT ncnn 权重（无 spv，CPU 回退对拍）
│   └── m6_vae/                #   VAE 6 子图权重
├── export/                    # PyTorch → ncnn 导出脚本（约 148K）
├── tools/                     # Python 端到端驱动 + 对拍脚本（约 12M）
└── third_party/               # stb_image / stb_image_write
```

清理前后对比：

| 项 | 清理前 | 清理后 |
|---|---|---|
| 总占用 | ~90 GB | ~16 GB |
| 删除内容 | 模型导出中间产物（.pt/.onnx/.pnnx ≈65G）、历史模型 m1-m4（≈4G）、测试 exe、调试 dump、构建缓存、e2e 中间数据 | — |

## 7. 构建与使用

```bash
# 构建（w64devkit + cmake + ninja）
cmake -G Ninja -B . && ninja seedvr2_run

# 推理
./seedvr2_run.exe <input.png> <output.png> [--resolution 1080] [--seed 42] [--no-colorfix]
```

## 8. 已知边界

1. **精度**：全链路 fp32（对齐 PyTorch fp32），官方 pipeline 用 bf16。端到端图 vs 官方 cos=0.969 是官方 bf16 量化噪声，非本实现 bug。
2. **RNG**：对齐的是 PyTorch **CPU** 路径；CUDA 用 curand Philox，算法不同。VAE 采样因 C++ VAE 是 fp32、官方是 fp16，latent 不会端到端逐位一致（属 VAE 精度差异）。
3. **attention 子图分辨率**：VAE attention 的 Reshape 写死 1080p 的 136×202，换输入尺寸需改 `export_vae.py` 里的 W/H 或改成动态 shape_expr。
4. **性能**：VAE/DiT 全 fp32 偏慢，可评估 fp16 提速（需 shader 改 `float16_t` 并处理精度）。

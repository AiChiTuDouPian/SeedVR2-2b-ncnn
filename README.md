# SeedVR2-ncnn

SeedVR2 (NaDiT 3B) 视频/图像超分的 **自包含 C++ ncnn + Vulkan** 推理实现。

- DiT 与自适应窗口注意力（AWA, Adaptive Window Attention）前向在自定义 Vulkan compute shader 中完成（含 qk_norm / 3D mmrope RoPE / varlen SDPA / txt 跨窗口平均）。
- **合并计算图（graph）**：32 层 DiT 切分为若干连续前向块（fp32 每块 2 层 / bf16 每块 1 层），每块是一个独立 ncnn::Net，块内全程 GPU 连续执行、只在块末 download 一次，大幅减少 CPU↔GPU 往返（旧逐算子架构 320 次同步 → graph 后仅 16~32 次）。
- VAE 调用外部 PyTorch 导出流程得到的 ncnn 权重（本项目仅做推理端 C++ 化）。
- 同时支持**图像**与**视频（帧序列目录，逐帧独立超分）**两种输入模式。
- 精度：默认 fp32（与参考实现对拍逐位一致）；可选 `--bf16`（推荐提速路径）；`--fp16` 已实现但失真，暂不推荐。

## 依赖

- 自行编译的 **ncnn（开启 Vulkan）**：`NCNN_DIR` 指向其 install 目录（需含 `libncnn.a` 与 glslang 静态库）。
- **Vulkan SDK**（含 `glslangValidator`，用于把 `src/shaders/*.comp` 编译为 `.spv`）。
- CMake ≥ 3.18，C++17 编译器（MinGW / MSVC / clang 均可）。

## 构建

```bash
cmake -B build-cmake \
  -DNCNN_DIR="<你的 ncnn install>" \
  -DVULKAN_SDK="<你的 VulkanSDK>" \
  -G "MinGW Makefiles"
cmake --build build-cmake -j
```

构建时 CMake 会用 `glslangValidator` 将 `src/shaders/*.comp`（含 `awa.comp`、`awa_coalesce.comp`、`cast_f16_f32.comp`、`cast_f32_f16.comp` 等）编译为对应 `.spv`。
若环境中没有 `glslangValidator`，需手动把 `.spv` 放到 `models/m5/`。

## 模型权重

`models/` 下的权重目录（`m5/`、`m5_nospv/`、`m6_vae/`、`m5_graph/`，共 16G+）**未入库**。

- `models/m5/`      —— DiT 权重（含 `awa.spv`）
- `models/m6_vae/`  —— VAE 权重
- `models/m5_graph/`—— **合并计算图权重块**：由 `export/export_dit_graph.py` 从 `models/m5/` 重新生成（fp32 16 块 + bf16 32 块 + f16 32 块，53G+），不入库。

请用 `export/` 下的导出脚本从官方 SeedVR2 权重重新导出，或自行放置权重。

生成合并计算图块：

```bash
# 导出 fp32 块（CH=2，16 块，覆盖 32 层）
python export/export_dit_graph.py models/m5 32 0 2   dit_block_0
python export/export_dit_graph.py models/m5 32 2 4   dit_block_1
# ... 每 2 层一块，直到 32 层（共 16 块）
# bf16 块（CH=1，32 块）：prefix 用 dit_block_bf16_
python export/export_dit_graph.py models/m5 32 0 1   dit_block_bf16_0
# ... 每层一块，直到 32 层（共 32 块）
```

## 用法

```bash
# 图像超分（单帧）
seedvr2_run.exe <输入图片.png> <输出图片.png> [--resolution N] [--bf16] [--seed N]

# 视频超分（逐帧独立）：输入一个存放帧的目录，输出同名帧目录
seedvr2_run.exe <frames_in/> <frames_out/> [--resolution N] [--bf16] [--seed N]
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--resolution N` | 输入图最短边 bicubic resize 到 N（如 360 / 1080 / 2160），再按 8× 下采样得到 latent 网格 |
| `--fp16` / `--bf16` | 精度选项；默认 fp32。bf16 走 tensor core 更快；fp16 已实现对齐但失真，不推荐 |
| `--seed N` | 随机种子（图片默认 0，视频每帧 seed = base + frame_idx 保证逐帧确定） |
| `--no-graph` | 禁用合并计算图，回退旧路径 `forward_latent`（逐算子 Vulkan 注意力，慢且高分辨率失真，仅用于对拍） |
| `--graphdir DIR` | 指定合并计算图目录（默认 `models/m5_graph`） |

环境变量：

- `SEEDVR_GRAPH_RELEASE=1`：强制逐块释放 DiT 权重（图片模式默认即逐块释放，帧序列模式默认常驻以保留帧间加速）。

说明：

- 视频模式为官方 `batch_size=1` 的特例（逐帧独立，不做时序融合）。
- **图片模式**默认逐块释放 DiT 权重（显存安全）；**帧序列模式** DiT 权重常驻（bf16 10GB < 16GB 显存，帧间零重载，约 5× 加速）。

## 精度与质量现状

| 精度 | 1080p 可用性 | 与 fp32 参考 PSNR | 说明 |
|------|------------|------------------|------|
| **fp32（默认）** | ✅ 正确 | cos=1.0 逐位一致 | graph 路径与 CPU `forward_latent` 参考逐位一致；单图墙钟 ~140s（含首次 pipeline 编译 ~30s） |
| **bf16** | ✅ 可用（已修复 segfault） | ~14.5 dB（bf16 精度水平） | graph 路径 1080p 之前段错误，已修复；默认图片模式自动逐块释放；1080p 墙钟 ~118s |
| **fp16** | ❌ 失真（油画化） | ~15 dB 但内容错乱 | AwaLayer 已加 fp16↔fp32 cast 分支修复位级错乱，但 graph 的 ada/RMSNorm 在 fp16 域溢出 65504 → 输出碎片状失真。**已放弃 fp16，建议用 bf16** |

> 旧路径 `forward_latent`（非 graph）：360p 正常，但 1080p 起 AWA 手动注意力分支失真（32.89 dB），仅用于开发期对拍，不推荐日常使用。

## 推理性能（RTX 5060 Ti 16GB，合并计算图，单图）

| 分辨率 | 精度 | VAE enc | DiT(32层) | VAE dec | 计算段合计 | 单图墙钟* |
|--------|------|--------:|----------:|--------:|-----------:|----------:|
| 360p (540×360) | fp32 | 1.08s | 31.91s | 1.25s | 34.2s | ~64s |
| 360p | bf16 | 1.01s | 35.62s | 2.42s | 39.1s | ~69s |
| 1080p (1620×1080) | fp32 | 3.76s | 100.30s | 5.46s | 109.5s | ~140s |
| 1080p | bf16 | 3.58s | 79.26s | 5.12s | 88.0s | ~118s |

\* 墙钟 = 计算段 + 首次冷启动（~700 个 Vulkan pipeline 编译，约 30s）；同进程跑第 2 张起冷启动归零。

对比：旧逐算子路径 1080p DiT 184s → graph fp32 100s（**1.84× 加速**）。瓶颈为 ncnn fp32 不吃 tensor core（GEMM 慢）+ 块间 CPU 往返；bf16 已通过 tensor core 缓解。

## 已知限制

- **4K（3840×2160）暂不支持**：VAE 在 4K 对 270×480=129600 个 mid-token 做 O(n²) self-attention，显存远超 16GB（fp32 / bf16 均在 VAE 阶段 `vkAllocateMemory failed`）。DiT graph 本身正确，需实现 **tiled VAE**（分块编/解码 + 重叠融合）才能打通 4K。
- **fp16 不推荐**（见上，失真）。
- ncnn 源码（`NCNN_DIR` 指向的库）不可修改，本项目仅在自有代码内修复 bug / 扩展。

## 目录结构

- `src/` —— C++ 推理引擎：
  - `seedvr2_engine.*` —— 端到端引擎（预处理 / VAE / DiT 调度 / 后处理），图片与帧序列双模式。
  - `dit_vk.*` —— DiT Vulkan 封装（init / 旧路径 `forward_latent` / graph 接口）。
  - `dit_graph.cpp` —— **合并计算图前向**：块加载、块间残差、精度 opt 区分、释放策略。
  - `awa_vk.*` / `awa_window.*` —— 旧路径 AWA 注意力与窗口几何（对拍用）。
  - `awa_layer.*` —— **自定义 ncnn::Layer**（AWA 的 flat-float fp32 Vulkan shader + bf16/fp16 cast 分支）。
  - `cast_bf16_layer.*` —— bf16 cast 层（预留，当前 AWA 内部转换优先）。
  - `vae_vk.*` —— VAE encode/decode（独立作用域，逐帧重建，避免 DiT 分块 reset 悬空）。
  - `main.cpp` —— CLI 入口与参数解析。
  - `shaders/*.comp` —— 自定义 GLSL compute shader（awa / awa_coalesce / awa_init / cast_f16_f32 / cast_f32_f16 / cast_bf16_f32 / cast_f32_bf16）。
- `export/` —— PyTorch → ncnn 的权重导出脚本（`export_dit_graph.py` 合并 32 层为计算图块）。
- `tools/` —— 调试 / 对拍辅助脚本（与官方 PyTorch 参考实现比对）。
- `third_party/` —— stb_image 头文件（图像读写）。

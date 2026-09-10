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
| `--resolution N` | 输入图最短边 bicubic resize 到 N（如 360 / 720 / 1080），再按 8× 下采样得到 latent 网格；输出保持原始宽高比 |
| `--fp16` / `--bf16` | 精度选项；默认 fp32。两者都走 tensor core（cooperative matrix），bf16 激活层间自动回落 fp32 存储故精度远好于 fp16 |
| `--seed N` | 随机种子，**默认 42**（与官方 CLI `--seed 42` 一致）；帧序列模式每帧 seed = base + frame_idx |
| `--no-graph` | 禁用合并计算图，回退 `forward_latent`（逐层 ncnn Net + C++ 注意力参考）。fp32 1080p 因显存限制走此路径 |
| `--cpu` | 纯 CPU 推理（关闭 Vulkan compute，自动强制 `--no-graph`）。注意本机 15.8GB 内存下 1080p 会 OOM，见下 |
| `--no-colorfix` | 关闭 LAB 色彩校正。**与官方 PyTorch 对拍时必须加**（官方对应 `--color_correction none`） |
| `--graphdir DIR` | 指定合并计算图目录（默认 `models/m5_graph`） |

环境变量：

- `SEEDVR_GRAPH_RELEASE=1`：强制逐块释放 DiT 权重（图片模式默认即逐块释放，帧序列模式默认常驻以保留帧间加速）。
- `SEEDVR_GRAPH_CHUNK=N` / `SEEDVR_BLOCK_PREFIX=...`：覆盖图块层数与块文件前缀（二分实验用）。
- 诊断 dump（默认关闭，env 门控）：`SEEDVR_DUMP_B0MID[_GPU]`、`SEEDVR_DUMP_BLOCKOUT`、`SEEDVR_DUMP_VIDGRID`、`SEEDVR_DUMP_TAIL`、`SEEDVR_LOAD_X0=<bin>`、`SEEDVR_NO_REUSE`、`SEEDVR_FULLWIN` 等。

> ⚠️ **与官方 PyTorch 对拍的口径要求（否则数值无意义）**
> 1. 色彩校正必须两边同时关闭：ncnn 加 `--no-colorfix`，官方用 `--color_correction none`。
>    两边算法实现不同（ncnn 为 LAB transfer 强度 0.8，官方为小波多尺度），只关一边会引入全局色调差，
>    1080p PSNR 会从 **46.6dB 掉到 27.4dB**。
> 2. seed 必须一致（默认都是 42）。引擎的噪声用 `rng::PhiloxRandn` 复刻官方 CUDA Philox4x32-10，
>    与 `torch.randn(CUDA)` 同源，因此噪声可逐位对齐。
> 3. DiT 权重来源要认：官方 fp8 权重与 ncnn 导出的 fp16 权重不同源，会有约 9% 底噪；
>    图像级 ground truth 请用官方 **fp16** 全流程图。

说明：

- 视频模式为官方 `batch_size=1` 的特例（逐帧独立，不做时序融合）。
- **图片模式**默认逐块释放 DiT 权重（显存安全）；**帧序列模式** DiT 权重常驻（bf16 约 10.2GB < 16GB 显存，帧间零重载）。

## 精度与质量现状

以官方 PyTorch fp16 全流程为 ground truth，**椎名真白 1131×960 / seed 42 / 色彩校正两边同关**
（完整报告与口径说明见 [`bench/PT_COMPARISON.md`](bench/PT_COMPARISON.md)）：

| 精度 | vs PyTorch @1080p | vs ncnn-fp32 @1080p | 1080p 耗时 | 结论 |
|---|---:|---:|---:|---|
| **fp32（默认）** | **cos 0.999977 / 46.55 dB** | — | 196 s | 与官方数值等价（SSIM 0.9978） |
| **bf16** | **cos 0.999962 / 45.45 dB** | cos 0.999981 / **48.16 dB** | **114 s** | **≈无损，生产推荐** |
| fp16 | cos 0.999241 / 31.06 dB | cos 0.999203 / 30.53 dB | 113 s | 速度与 bf16 相同但精度差一个量级，**不推荐** |

其他分辨率（vs PyTorch，fp32 / bf16 / fp16）：

| 分辨率 | 输出 | fp32 | bf16 | fp16 |
|---|---|---:|---:|---:|
| 360p | 424×360 | 23.08 dB | 22.89 dB | 20.36 dB |
| 720p | 848×720 | 34.81 dB | 34.80 dB | 27.46 dB |
| 1080p | 1272×1080 | **46.55 dB** | **45.45 dB** | 31.06 dB |

> **为什么 360p/720p 的 PSNR 偏低？** 因为源图 1131×960 在这两档下是被**缩小**的
> （0.375× / 0.75×），而 SeedVR2 是超分模型，缩小任务属训练分布之外，微小数值差异会被放大。
> 对照实验（源图预缩到 212×180 → 360p，变成 2.0× 真放大）显示 PSNR 回到 **45.82 dB**。
> 即：**只要任务是真放大（≥1×），ncnn 与 PyTorch 的一致性稳定在 45~46.6 dB**。
> 详见 [`bench/PT_COMPARISON.md`](bench/PT_COMPARISON.md) §4。

CPU 与 GPU 在 360p fp32 下几乎逐位一致（cos 1.000000 / PSNR 80.52 dB）。

## 推理性能（RTX 5060 Ti 16GB，单图）

完整报告见 [`bench/PERFORMANCE_ANALYSIS.md`](bench/PERFORMANCE_ANALYSIS.md)。

| 分辨率 | latent Lv | fp32（forward_latent） | bf16（graph 32 块） | fp16（graph 32 块） |
|---|---:|---:|---:|---:|
| 360p (424×360) | 621 | 49 s（DiT 45.1 s） | **40 s**（DiT 37.7 s） | 42 s（DiT 39.4 s） |
| 720p (848×720) | 2385 | 100 s（DiT 94.5 s） | **61 s**（DiT 57.1 s） | 65 s（DiT 60.2 s） |
| 1080p (1272×1080) | 5440 | 196 s（DiT 186.0 s） | **114 s**（DiT 105.6 s） | 113 s（DiT 105.4 s） |

- **DiT 占 wall 的 85%~95%**，是唯一值得优化的部分。
- 低精度 graph 比 fp32 快 **1.7~1.8×**。
- fp32 `forward_latent` 的 DiT 构成（1080p）：GEMM 52~58% / AWA 自定义算子 13~15% / CPU 循环 28~35%。
- **当前最大瓶颈：低精度 graph 的块间搬运**（32 块，每块 download+upload 约 2.3 s，
  合计约 74 s ≈ DiT 的 70%，32 块共约 3.5 GB 过 PCIe）。
  **把 chunk 1→2/4（32 块→16/8 块）预计可让 DiT 再降 35%~45%**，是首选优化。
- 帧序列模式下 DiT 常驻：720p bf16 首帧 65.7 s → **稳态 33.5 s/帧**（省约 47%）。
  16 GB 显存下驻留仅 720p bf16 可行（低精度块权重常驻 10.18 GB）。

## 已知限制

- **4K（3840×2160）暂不支持**：VAE 在 4K 对 270×480=129600 个 mid-token 做 O(n²) self-attention，显存远超 16GB（fp32 / bf16 均在 VAE 阶段 `vkAllocateMemory failed`）。DiT graph 本身正确，需实现 **tiled VAE**（分块编/解码 + 重叠融合）才能打通 4K。
- **CPU 路径 1080p 不可用**：本机系统内存仅 15.8 GB，而引擎 CPU 模式把全部层权重常驻
  （`evict_nets()` 的 `cap=1024`）+ `fcache` 永久驻留 bin 字节。修复方向见性能报告 §4。
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
- `tools/` —— 调试 / 对拍辅助脚本（与官方 PyTorch 参考实现比对）：
  - `pt_reference.py` —— 生成官方 PyTorch 参考图（固定对拍口径）
  - `img_metrics.py` —— 两张 PNG 的 cos / PSNR / mean|d| / max|d| / SSIM
  - `benchmark.cpp` —— 逐阶段计时 benchmark（`seedvr2_bench`）
- `bench/` —— **性能与精度基准**（脚本 + 报告 + PyTorch 基准图）：
  - `run_pt_compare.sh` —— ncnn × {fp32,fp16,bf16} × {360,720,1080} vs 官方 PyTorch 矩阵
  - `run_perf_cpu_gpu.sh` —— fp16/bf16/fp32 × GPU/CPU @1080p 性能批测
  - `PT_COMPARISON.md` / `PERFORMANCE_ANALYSIS.md` —— 对比与性能分析报告
- `docs/` —— 技术文档（`GPU加速与自定义算子技术文档.md`）；`docs/archive/` 为历史快照。
- `third_party/` —— stb_image 头文件（图像读写）。

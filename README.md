# SeedVR2-ncnn

SeedVR2 (NaDiT 3B) 视频/图像超分的 **自包含 C++ ncnn + Vulkan** 推理实现。

- DiT 与自适应窗口注意力（AWA, Adaptive Window Attention）前向在自定义 Vulkan compute shader 中完成（含 qk_norm / 3D mmrope RoPE / varlen SDPA）。
- VAE 仍调用外部 PyTorch 导出流程得到的 ncnn 权重（本项目仅做推理端 C++ 化）。
- 同时支持**图像**与**视频（帧序列目录，逐帧独立超分）**两种输入模式。
- 精度：默认 fp32（与参考实现对拍逐位一致）；可选 `--fp16` / `--bf16`（混合精度，fp16 存储 + fp32 累加）。

## 依赖

- 自行编译的 **ncnn（开启 Vulkan）**：`NCNN_DIR` 指向其 install 目录（需含 `libncnn.a` 与 glslang 静态库）。
- **Vulkan SDK**（含 `glslangValidator`，用于把 `src/shaders/*.comp` 编译为 `awa.spv`）。
- CMake ≥ 3.18，C++17 编译器（MinGW / MSVC / clang 均可）。

## 构建

```bash
cmake -B build-cmake \
  -DNCNN_DIR="<你的 ncnn install>" \
  -DVULKAN_SDK="<你的 VulkanSDK>" \
  -G "MinGW Makefiles"
cmake --build build-cmake -j
```

构建时 CMake 会用 `glslangValidator` 将 `src/shaders/awa.comp` 编译成 `models/m5/awa.spv`。
若环境中没有 `glslangValidator`，需手动把 `awa.spv` 放到 `models/m5/awa.spv`。

## 模型权重

`models/` 下的权重目录（`m5/`、`m5_nospv/`、`m6_vae/`，共 16G+）**未入库**。
请用 `export/` 下的导出脚本从官方 SeedVR2 权重重新导出，或将权重放置到：

- `models/m5/`      —— DiT 权重（含 `awa.spv`）
- `models/m6_vae/`  —— VAE 权重

## 用法

```bash
# 图像超分（单帧）
seedvr2_run.exe <输入图片.png> <输出图片.png> [--fp16|--bf16] [--seed N]

# 视频超分（逐帧独立）：输入一个存放帧的目录，输出同名帧目录
seedvr2_run.exe <frames_in/> <frames_out/> [--fp16|--bf16] [--seed N]
```

- 视频模式为官方 `batch_size=1` 的特例（逐帧独立，不做时序融合）。
- `--fp16` / `--bf16` 仅改变 GEMM 存储精度，累加器仍为 fp32（安全、不溢出）。

## 目录结构

- `src/`         —— C++ 推理引擎（`seedvr2_engine.*`、`dit_vk.*`、`vae_vk.*`、`awa_vk.*`、`main.cpp`）与 `.comp` 自定义 shader。
- `export/`      —— PyTorch → ncnn 的权重导出脚本。
- `tools/`       —— 调试 / 对拍辅助脚本（与官方 PyTorch 参考实现比对）。
- `third_party/` —— stb_image 头文件（图像读写）。

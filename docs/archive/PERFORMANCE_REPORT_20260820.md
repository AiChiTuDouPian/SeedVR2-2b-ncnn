# SeedVR2-ncnn 性能分析报告

> 生成时间：2026-08-20
> 硬件：NVIDIA GeForce RTX 5060 Ti 16GB（bf16-cm=16x16x16，tensor core 支持）
> 输入：test_frames_in/000.png（540×360）
> 测试工具：tools/benchmark.cpp（seedvr2_bench）

---

## 一、执行路径与耗时总览

模型存在三种加载/执行路径，性能差异显著：

| 路径 | 加载方式 | DiT(32层) | 端到端 | 相对提升 |
|---|---|---|---|---|
| 旧分块路径 | 逐层 ncnn Net，每层 CPU↔GPU 往返 | 55.9 s | 58.9 s | 1.0×（基线） |
| **分块图常驻** | dit_block_bf16_*，32 块×1 层，GPU 常驻 | **11.3 s** | **13.8 s** | **4.3×** |

**核心结论：分块图常驻是当前唯一生效的大幅优化，把 DiT 从 55.9s 压到 11.3s（5 倍）。**

---

## 二、分块图常驻路径的逐阶段耗时（480p）

| 阶段 | 耗时(ms) | 占比 | 计算位置 |
|---|---|---|---|
| 预处理 (resize/pad/norm) | 20 | 0.1% | CPU |
| VAE encode (子图GPU/采样CPU) | 1125 | 8.2% | GPU+CPU |
| **DiT (32层 NaDiT/AWA)** | **10700** | **77.5%** | GPU |
| upscaled (noise-sr)/0.9152 | 0.2 | 0.0% | CPU |
| VAE decode (子图GPU) | 1468 | 10.6% | GPU |
| LAB 色彩校正 | 111 | 0.8% | CPU |
| 反归一化+裁剪 | 1.7 | 0.0% | CPU |
| **总计** | **13811** | 100% | — |

---

## 三、DiT 内部耗时分解（分块图）

通过块级计时（32 块，每块 1 层）定位：

- 每块总耗时：~330-370 ms
- extract（GPU 计算 + 块末下载）：~325-365 ms，占 **98%**
- 块间往返/upload/setup：**仅 ~2%**（约 5-8 ms/块）

**结论：DiT 11.3s 是纯 GPU 计算瓶颈（GEMM 为主），块间搬运已被分块图优化到可忽略。**

---

## 四、GPU 利用率实测

推理期间 nvidia-smi 采样：

| 指标 | 数值 |
|---|---|
| GPU 利用率 | **84% - 99%**（大部分时间 90-99%）|
| SM 时钟 | 2790 MHz（满频）|
| 功耗 | ~91 W（未撞功耗墙，5060 Ti 上限 ~160W+）|
| 显存占用 | ~14 GB / 16 GB |

**结论：DiT 阶段 GPU 已基本吃满，是 compute-bound，不是 memory-bound 或 latency-bound。**

---

## 五、已尝试/已完成的优化

### 5.1 分块图跨帧常驻（已生效，收益最大）
- 修改 `seedvr2_engine.cpp`：`release_graph()` 仅在图非常驻模式（`graph_resident=false`）时调用。
- **效果**：DiT 55.9s → 11.3s（5 倍），端到端 58.9s → 13.8s。

### 5.2 LAB 色彩校正 OpenMP 多线程（已生效）
- 给逐像素/逐行循环加 `#pragma omp parallel for` + `omp parallel sections`。
- **效果**：LAB 0.49s → 0.11s（约 4.5 倍）。

### 5.3 fp16 累加（`--fp16-arith`）（无效，已否定）
- 预期 2 倍 GEMM 吞吐，实测 DiT 11.44s ≈ fp32 累加 11.34s。
- **结论**：bf16 存储下 fp16/fp32 累加对 InnerProduct 吞吐无影响，此优化路线失效。

---

## 六、优化可行性结论

### 已被数据否定的方向
1. **合并 DiT 块层数（chunk 2/4）**：块间往返仅占 2%，合并块数收益 <3%，不值得。
2. **fp16 累加**：实测无提升，已到 GEMM 吞吐上限。

### 仍有空间的优化
1. **VAE 走 bf16**：VAE 现强制 fp32（`seedvr2_engine.cpp`），VAE encode+decode 共 2.6s（占 18.8%）。bf16 存储可减半 GEMM 带宽，预计 VAE 2.6s → ~1.5s，端到端省 ~1s。**需先解决"bf16 模式下 VAE 析构 pool allocator 崩溃"问题**。

### 已到硬件极限的方向
- **DiT 11.3s**：GPU 利用率 90-99% + SM 满频 + fp16/fp32 累加无差别，确认为 RTX 5060 Ti 在 480p 的算力上限。软件层面无进一步压缩空间，仅能靠更高分辨率摊薄固定开销或换更强硬件。

---

## 七、瓶颈与建议

### 当前瓶颈（按占比）
1. **DiT GEMM（77.5%）** — 硬件算力上限，无法软件突破
2. **VAE（18.8%）** — 强制 fp32，是最具性价比的剩余优化点
3. CPU 各阶段（LAB 等）— 已优化，占比 <2%

### 建议优先级
1. **VAE bf16**（预期省 ~1s，需解决崩溃）
2. 若追求更高吞吐：提高分辨率（tensor core 利用率更高，单位吞吐反而更好），或升级更大显存 GPU 以支持整图单 Net（消除剩余 2% 往返）

---

## 附：复现命令

```powershell
# 性能基准（分块图常驻，bf16）
cd F:\Seedvr2\seedvr2-ncnn
Remove-Item Env:SEEDVR_SINGLE_NET -ErrorAction SilentlyContinue
.\build\seedvr2_bench.exe --model models\m5 --image F:\Seedvr2\seedvr2-ncnn\test_frames_in\000.png --precision 2 --resolution 480 --warmup 1 --runs 3

# 推理单图并保存（bf16）
.\build\seedvr2_run F:\Seedvr2\seedvr2-ncnn\test_frames_in\000.png output\000_upscaled.png --bf16 --resolution 480
```

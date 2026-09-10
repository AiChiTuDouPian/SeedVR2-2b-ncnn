# SeedVR2-ncnn GPU 加速技术文档：工程实现与自定义算子

> 项目：`F:/Seedvr2/seedvr2-ncnn` ｜ 日期：2026-09-10
> 平台基线：RTX 5060 Ti 16GB ｜ ncnn（自编译，Vulkan 开启）+ Vulkan SDK（glslangValidator）
> 数据基线：椎名真白源图（1131×960）→ `--resolution 1080`（输出 1272×1080，latent token Lv=5440），seed=42

---

## 1. 文档范围

本文档面向**实现与性能优化**，回答三个问题：

1. **整个工程怎么跑在 GPU 上**：从单张图到 32 层 DiT 的整条 Vulkan 链路、显存/生命周期策略、精度体系、块间数据搬运。
2. **自定义算子怎么做**：AWA（自适应窗口注意力）、AdaCompose、Cast 系列如何以「ncnn 自定义 Layer + 手写 GLSL compute shader」方式实现，各自的接口约定、数据布局与踩坑。
3. **现在有多快、瓶颈在哪、下一步怎么优化**：附 2026-09-10 的 1080p 三精度实测数据。

配套源码索引（读代码按此顺序最佳）：

| 文件 | 内容 |
|---|---|
| `src/main.cpp` | CLI：图/帧序列双模式、`--resolution/--seed/--fp16/--bf16/--cpu/--graphdir/--no-graph` |
| `src/seedvr2_engine.cpp/.h` | 单帧主链编排：resize→VAE encode→采样→DiT→VAE decode→color_fix |
| `src/dit_vk.cpp/.h` | DiT 宿主：层网缓存(LRU)、forward_latent 逐层路径、`forward_graph` 分块路径、几何/权重管理 |
| `src/dit_graph.cpp` | **分块合并计算图**：块 Net 加载、Input 上传、GPU extract、块间残差传递（GPU 加速核心） |
| `src/awa_layer.cpp/.h` | AWA 自定义 ncnn Layer（graph 整图内嵌） |
| `src/awa_vk.cpp/.h` | AWA 独立引擎路径（forward_latent 用） |
| `src/awa_window.cpp/.h` | 窗口网格/几何生成（shifted/nonshifted） |
| `src/ada_compose.cpp/.h` | AdaCompose 自定义层（GPU 算全部 ada 向量） |
| `src/cast_bf16_layer.cpp/.h` | Bf16Cast 自定义层 |
| `src/vae_vk.cpp/.h` | VAE 编解码 ncnn 子图（含 attention） |
| `src/shaders/*.comp` | 全部 GLSL compute shader（CMake 自动编 .spv 到 models/m5/） |
| `src/rng.h` | CPU 采样 RNG（PhiloxRandn / mt19937，与 torch.randn 对齐） |

---

## 2. 系统架构与整图数据流

```
输入 LR 图
  │ image_io (stb_image)
  ▼
① 预处理      bicubic 短边=resolution → clamp → pad16 → normalize[-1,1]（CPU）
  ▼
② VAE encode  16 个 latent 通道：mean/logvar（4 子图 + 2 attention，全程 GPU）
  ▼
③ 采样        latent = mean + exp(0.5·logvar)·randn(seed+1e6)（CPU，rng.h）
              noise  = randn(seed)（CPU）
  ▼
④ 构造 vid_grid  [noise(16) | latent·0.9152(16) | mask(1)] = 33 通道 channel-last，
              按 2×2 patch 打包成 132 维 vid token（Lv 个）+ txt 条件 token（TXT 个，5120 维）
  ▼
⑤ DiT 32 层   输入 vid(token,132) + txt(token,5120)
              ── graph 路径：8/16/32 块 ncnn::Net，块内连续 GPU，块末 1 次 download
              ── forward_latent 路径：逐层 Net + 层间残差
  ▼  sr_latent (16, H/8, W/8)
⑥ x0 = noise − sr；/0.9152（unpatchify 已在层内完成）
  ▼
⑦ VAE decode → RGB
  ▼
⑧ LAB color_fix（可选）→ 保存
```

**模型架构常量**（NaDiT，3B）：

| 常量 | 值 | 说明 |
|---|---|---|
| 层数 | 32 | 其中前 10 层为 MM_LAYERS（vid/txt 双分支调制） |
| HEADS / HEAD_D | 20 / 128 | 多头注意力 |
| DIM | 2560 | vid/txt 主干维 |
| QKV | 7680 | q/k/v 各 2560，vqkv 布局 `[Lv, QKV]`，偏移 0/2560/5120 |
| ROPE_ROT | 126 | 3D mmrope 旋转维数（vid 空间 2D + txt） |
| RMSNorm eps | 1e-5 | qk_norm 等 |
| vid token | 132 维 | 2×2 patch × 33 通道 |
| txt token | 5120 维 | 文本条件（TXT≈58 @1080p） |
| 采样 | flow 语义 | DiT 输出 v，x0 = noise − v，SCALING_FACTOR=0.9152 |

---

## 3. GPU 加速总体设计

### 3.1 Vulkan 设备与显存生命周期（dit_vk.cpp）

- **单一 VulkanDevice** 贯穿引擎；`set_vulkan_device` 后所有 Net/自定义层共用。
- **allocator 只 acquire 一次**（`init` 时）：`blob_alloc` / `staging_alloc` 全局共享给所有 ncnn::Net、AWA、graph 块。
  - 历史教训：早期按需 `acquire_blob_allocator()`「拿而不还」，累积数百个 allocator → 32 层深推理 `vkAllocateMemory failed -2`。改为共享后解决。
- **逐层 Net 缓存 LRU**：`net_cache` + `net_lru`。GPU 上限 `NET_CACHE_CAP=16`，超出淘汰最久未用（控制显存）。权重经 `VkWeightAllocator` 驻留 device-local。
- **`reset_vulkan_device()` 兜底**：分块间重建 VkDevice + allocator 回到干净状态（CPU 侧权重字节保留在 `fcache`，不重读磁盘），解决深层不稳定与碎片分配失败。
- **析构顺序敏感**：`net_cache`（含权重分配器/pipeline）→ graph 块 Net → AWA pipeline，都必须在 `destroy_gpu_instance` 之前析构。
- CPU 侧 `fcache` 缓存 param/bin 原始字节：Net 被 LRU 淘汰后可免磁盘重读重建。

### 3.2 合并计算图（graph）——消除 CPU↔GPU 往返（dit_graph.cpp）

旧逐算子/逐层架构每层 2 upload + 2 download；graph 把 **32 层切成连续块**，每块一个独立 ncnn::Net，块内全程 GPU 连续 record，只在块末 download 一次：

| 精度 | 块文件前缀 | 每块层数 | 块数 | 说明 |
|---|---|---|---|---|
| fp32 | `dit_block_` | 2（chunk=2） | 16 | 块权重 fp32；1080p 整块执行 16GB 显存吃紧（已 OOM），主要服务 360p/对拍 |
| fp16 | `dit_block_f16_` | 1（chunk=1） | 32 | 块权重低 16 位 |
| bf16 | `dit_block_bf16_` | 1（chunk=1） | 32 | 块权重低 16 位 |
| bf16 | `dit_block_bf16_c2_` | 2（chunk=2） | 16 | **引擎默认路径**；净图结构开销比 chunk=1 低约 1.8%，输出逐位一致 |
| bf16 | `dit_block_bf16_c4_` | 4（chunk=4） | 8 | 仅作对照：实测无进一步收益（大 Net 开销抵消块数减少） |

块文件位置：`models/m5_graph/`（该目录约 95GB，每套块图约 9.7GB；不入库，由 `export/export_dit_graph.py` 生成）。
chunk 由 `src/dit_graph.cpp` 的 `block_prefix(precision, chunk)` 决定：低精度 chunk>1 追加 `c{N}` 标记，
与 chunk=1 的旧文件并存；`SEEDVR_GRAPH_CHUNK` / `SEEDVR_BLOCK_PREFIX` 可覆盖（诊断用）。
chunk 收益与开销构成见 `bench/PERFORMANCE_ANALYSIS.md` §3。

块执行要点：

1. **Input 上传防覆盖**：ncnn 的 blob allocator 会让多个 Input blob 复用同一 buffer → 后上传覆盖先上传。修复：**每个 Input 用独立 `VkBlobAllocator` 实例**（`up_alloc_[b][seq]`），物理隔离，跨帧 `clear()` 复用。
2. **块间残差传递**：非末块输出 blob 命名 `v_cur_{l1}` / `t_cur_{l1}`（l1=块结束层号），GPU extract 下载 → CPU 向量 → 下一块 upload。末块输出 `out0`（64 维/帧像素通道）。
3. **帧序列常驻**：视频/帧序列模式低精度权重 < 显存 → 块 Net 常驻（`persistent`），帧间零加载；单图模式 fp32 权重 20GB 超显存 → 逐块加载/释放。
4. **单 Net 模式**（`load_single`）：小层数整图合并 1 个 Net（如 4 层对拍），用于验证「合并机制」数值正确。

### 3.3 块内提取语义（本战役最大坑，见 §5.1）

块间 vid/txt 与末块 out0 一律用 **GPU extract**：

```cpp
ncnn::VkMat vk_out, vk_out2;
ex.extract(outn, vk_out, cmd);            // 与主 forward 同一 cmd 连续 record
{ ncnn::Option od = bn->opt; od.use_packing_layout = false;
  cmd.record_download(vk_out, mv, od); }  // 内置 fp16/bf16→fp32 cast
cmd.submit_and_wait(); cmd.reset();
```

**绝不使用 `ex.extract(Mat&)`（CPU 版）**：在 Vulkan Net 上它会自建独立 VkCompute + 重 forward（ncnn `net.cpp` extract 路径），脱离主 forward 的 cmd/allocator 上下文 → 低精度下块间数据损坏（见 §5.1 复盘）。

### 3.4 低精度体系（precision_：0=fp32 / 1=fp16 / 2=bf16）

统一原则：**存储低 16 位 + 累加 fp32**，即 `use_fp16/bf16_storage=true` 但 **`use_fp16_arithmetic=false`**（fp16/bf16 累加器在真实大激活会溢出 Inf→NaN；fp32 累加器 + cooperative matrix = FLOAT16/BFLOAT16 输入 + FLOAT32 累加，见 ncnn `gemm_vulkan.cpp`）。

**fp16 与 bf16 的语义差异（关键认知）**：

| | fp16 (`--fp16`) | bf16 (`--bf16`) |
|---|---|---|
| 层间激活 | 整链真实 fp16 存储（带宽减半，加速面最宽） | ncnn 无 Vulkan 层声明 `support_bf16_storage` → 层间 cast 回 **fp32** |
| GEMM 内核 | fp16 tensor core | bf16 tensor core（5060 Ti 支持 bf16-cm） |
| 精度实测（360p vs fp32） | cos 0.995 / 25dB（32 层激活量化累积墙） | cos 0.99998 / **48.6dB** |
| 大激活溢出风险 | 有（曾 NaN） | 无（指数位同 fp32） |
| 结论 | 已可用但失真 | **生产低精度推荐** |

配套：AWA 等 flat-float 自定义层无法直接读 half/bf16（见 §4.1/§5.3），前后用 Cast 自定义层/shader 转回 fp32；VAE 子图同样跟随精度。

---

## 4. 自定义算子实现

### 4.1 AWA —— 自适应窗口注意力（核心算子）

**为什么自研**：NaDiT 的 AWA 在整图上做「窗口化可变长注意力」，ncnn 标准算子无法表达（partition 按窗口 gather + 跨窗口共享 txt + 两遍 softmax + unpartition）。全链路一次 dispatch 完成：

```
partition → qk_norm(RMSNorm) → 3D mRoPE(RoPE) → varlen SDPA(两遍 softmax) → unpartition
```

**shader 族（src/shaders/）**：

| 文件 | 职责 |
|---|---|
| `awa.comp` | 主 shader：partition（按 vidx gather）+ qk_norm + mmrope + 两遍 softmax SDPA + 写回。每 workgroup 处理一个 (window, head)，组内 128 线程逐 query 遍历窗口内全部 key |
| `awa_coalesce.comp` | txt 注意力跨窗口平均：`nwin×TXT×DIM → TXT×DIM`（原 CPU 宿主归约下沉到 GPU） |
| `awa_copy.comp` | 输入复制（防图内 blob 复用覆盖 AWA 输入） |
| `awa_init.comp` | 输出清零（窗口未覆盖 token 无线程写入，必须先清零） |
| `awa_mid*.comp` / `awa_min.comp` | 二分隔离诊断 shader（定位 device-lost / 错位的调试资产） |

**ncnn Layer 集成（awa_layer.cpp/.h）**：

- 继承 `ncnn::Layer`，注册名 `"AWA"`，graph/单 Net param 中内嵌；输入 `[vqkv, tqkv]`，输出 `[vattn, tattn]`。
- param 约定：`0=spv 目录, 1=win_type(0=nonshifted/1=shifted), 2=Lv(占位), 3=TXT(占位)`；bin 权重 `nq_v/nk_v/nq_t/nk_t`（各 128=HEAD_D）。
- **几何运行时注入**：`set_window_geometry(vidx, cumf, vfreq, tfreq, nwin, Lv, TXT)`，由宿主在块加载后按当前分辨率重算（分辨率可变，param 里 Lv 只是占位）。几何常量经 `upload_model(VkTransfer)` 预上传成常驻 buffer。
- **低精度 I/O**：`support_fp16_storage/support_bf16_storage=true`，但层内用 4 个 Cast pipeline（`cast_f16_f32/f32_f16/bf16_f32/f32_bf16`）把 I/O 转 fp32 再进 flat-float shader —— 见 §5.3 原理。
- **dispatch 约定**：`local_size=128`，与 `pipe->set_local_size_xyz(128,1,1)` 严格一致（否则间歇 `vkWaitForFences failed -4`）；`dispatcher.w/h/c` 传的是**元素数**而非 workgroup 数（ncnn 内部除以 local_size）。

**数值语义**：与 CPU 参考 `awa_forward` 完全一致（先按 vidx gather 窗口 token，再注意力；不可直接用 `(cumf+local)` 当 token）。qk_norm = 逐 head RMSNorm(q)/RMSNorm(k)；SDPA 用**两遍 softmax**（在线 max/denom 数值稳定，先求 max 与 exp 和，再归一）保证与 CPU 对拍 cos=1.0。

### 4.2 AdaCompose —— ada 调制向量 GPU 化

**动机**：旧图把每层每流 6 个 2560 维 ada 向量（共 32×2×6=384 个 + 2 个 final）从 CPU 算好再逐个 upload（384 次 record_upload）。AdaCompose 改为**只 upload 1 个 emb(15360)**，GPU shader 一次算完全部 ada 输出。

- param 语义：输入 `emb`(15360=2560×6 槽)，输出 12 流（每层 vid/txt × a_sc/a_sh/a_g/m_sc/m_sh/m_g）或按块内层数。
- 权重布局（`ada_compose.cpp` 读 bin 重排）：磁盘顺序 `attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate` → 输出顺序 `a_sc=scale, a_sh=shift, a_g=gate, m_sc=mlp_scale, m_sh=mlp_shift, m_g=mlp_gate`。
- **emb 槽位语义（重大 bug 修复）**：emb 6 槽布局下，final 层取 **attn 组**：`fin_sc=emb[d*6+1]+bias`、`fin_sh=emb[d*6+0]+bias`（此前 `d*3` 错位导致尾部 64 维投影 rel≈40% → 图像「糊」，修复后图像级 cos 0.999905/46.6dB）。
- `ada_compose.comp` / `_fp16.comp` / `_bf16.comp` 三变体，低精度图用 fp16/bf16 存储变体。
- 相对路径坑：spv 目录默认相对 cwd（`models/m5/`），换目录运行会 create_pipeline 失败 → SIGSEGV/SIGILL。**必须从仓库根目录运行**（或给绝对 spv 路径）。

### 4.3 Cast 系列 —— 低精度 ↔ fp32 位级转换

- `cast_f32_f16.comp` / `cast_f16_f32.comp`：fp32 线性 buffer ↔ fp16（half），逐元素位转换。
- `cast_f32_bf16.comp` / `cast_bf16_f32.comp`：fp32 ↔ bf16（截断/补齐 16 位指数+尾数）。
- 写回侧用 **atomicCompSwap CAS 循环**（ncnn 官方同款），避免奇偶线程对同一 uint 的非原子 RMW 竞争（防御性加固，本 GPU 上旧写法实测同结果）。
- AWA/AdaCompose 前后、低精度图块接口处使用。

### 4.4 自定义 flat-float shader 的强制约定（坑清单，务必遵守）

1. shader `layout(local_size_x/y/z=N)` 必须与宿主编译管线 `set_local_size_xyz(N,N,N)` 一致 → 否则间歇性 `vkWaitForFences failed -4`。
2. 自定义 shader 的 `ncnn::Option` 必须 `use_packing_layout=false` **且** `use_fp16_storage=use_fp16_packed=use_bf16_storage=use_bf16_packed=false`（按上传场景）。离散 GPU 默认 fp16，不关会把上传数据压成 half 而 shader 按 fp32 读 → 位级乱码。
3. elempack=4 的存储是**线性**的（pack1to4 产出连续 4 float），`float[]` 直接线性索引，无需 vec4 拆解。
4. `dispatcher.w/h/c` 是**元素数**（ncnn 在 `command.cpp` 内部 `/local_size`），不是 workgroup 数。传错只 dispatch 极少 workgroup，前 N 个元素正确、后面全 0/旧值——**小测试只查头部会误报 PASS**。

---

## 5. 关键问题复盘（现象 → 根因 → 修复）

### 5.1 fp16/bf16 graph 整图乱码 —— CPU extract 语义（2026-09-10 修复，commit a19894d）

- 现象：低精度 graph 出图整图乱码（PSNR ~8-14dB）；块 0 首个 GEMM 输出量级缩 ~27x、cos≈0.019；blob 级对拍显示 ada 链全零/错位。
- 排查链：① f16 文件 + fp32 引擎 → 字节级一致（文件没问题）；② fp32 文件 + fp16 opt → 仍坏（问题在引擎低精度执行路径）；③ NO_REUSE / FP32STORAGE 均无效。
- 根因：低精度分支用 **`ex.extract(Mat&)`（CPU 版）** 取块间 vid_cur/txt_cur 与 out0。Vulkan Net 上该 API **内部自建独立 VkCompute + 重 forward**（ncnn net.cpp ~2886-2910），块间传递数据损坏。此前 blob 级「坏」其实都是 CPU extract 重 forward 的产物。
- 修复：低精度 mv/mt/out0 全部改走与 fp32 相同的 **GPU extract 主路径**（§3.3）。`record_download` 内部 `convert_packing(cast_type_to=1)` 自动做 fp16/bf16→fp32。
- 验证：fp32 PNG **字节级 SAME（零影响）**；bf16 PSNR 8→**48.6dB**；fp16 14→25dB（残余为 32 层 fp16 存储固有量化墙，非 bug）。

### 5.2 fp16 溢出 NaN —— 区分「存储」与「累加」

- 现象：真实大激活（1080p）fp16 全图 NaN。
- 根因：`use_fp16_arithmetic=true`（fp16 累加器），与 fp16 存储无关。
- 修复：一律 `use_fp16_arithmetic=false`（混合精度：低 16 位存储 + fp32 累加）。cooperative matrix 走 FLOAT16/BFLOAT16 输入 + FLOAT32 累加。

### 5.3 为什么 AWA 不能直接吃 fp16/bf16 存储

fp16 存储时 ncnn `VkTransfer` 上传把 fp32 逐元素压成 half 写入 buffer（字节减半、布局改变）；flat-float shader 用 `float[]` 按 4 字节/元素索引，会把两个 half 位模式拼成一个 float → **位级错位胡值**（几十/几百倍，非精度损失）。标准算子能 fp16 是因为其 shader 配套 `float16_t`/`uvec4` 并对齐上传约定；自研层绕过张量约定后必须保证 buffer 是 fp32 线性布局 → AWA/AdaCompose 前后用 Cast 层显式转换。

### 5.4 blob 复用覆盖 Input / allocator 泄漏 / 深层崩溃

- Input 被覆盖 → 每 Input 独立 `VkBlobAllocator`（§3.2）。
- allocator「拿而不还」→ 全局共享一次性 acquire + LRU cap16 + 析构顺序修正 + `reset_vulkan_device()` 兜底（§3.1）。
- 窗口几何注入必须在块 Net 加载之后（`graph_load_block` 后）→ 否则 AwaLayer 落回 win bin 固定网格（nwin=8/3200）→ 窗口错乱 → 黑图。

### 5.5 尾段 emb 槽位 bug（d*6 vs d*3）

官方 `vid_out_ada layers=["out"]` 取 attn 组 → `emb[d*6+0/1]`。三处同源实现（ada_compose final 分支、dit_graph CPU 兜底、verify/test 参考）曾用 `d*3` → 尾部 2560→64 投影错位、图像糊。修复后图像级 cos 0.999905 / 46.6dB（官方 fp16 为基准）。

---

## 6. 性能数据与优化方向

> 完整数据（含 360p/720p/1080p × fp32/fp16/bf16 × GPU/CPU 与 PyTorch 精度对比）见
> [`bench/PERFORMANCE_ANALYSIS.md`](../bench/PERFORMANCE_ANALYSIS.md) 与
> [`bench/PT_COMPARISON.md`](../bench/PT_COMPARISON.md)。本节只摘录要点。

输入 1131×960，seed 42，`--no-colorfix`（Lv = latent token 数）：

| 分辨率 | latent Lv | fp32（forward_latent） | bf16（graph 32 块 chunk=1） | fp16（graph） |
|---|---:|---:|---:|---:|
| 360p (424×360) | 621 | wall 49 s / DiT 45.1 s | **40 s / 37.7 s** | 42 s / 39.4 s |
| 720p (848×720) | 2385 | wall 100 s / DiT 94.5 s | **61 s / 57.1 s** | 65 s / 60.2 s |
| 1080p (1272×1080) | 5440 | wall 196 s / DiT 186.0 s | **114 s / 105.6 s** | 113 s / 105.4 s |

- **DiT 占 wall 的 85%~95%**；低精度 graph 比 fp32 快 **1.7~1.8×**；fp16 与 bf16 速度相同
  （都被块间搬运支配）但 fp16 精度差一个量级 → 生产选 bf16。
- fp32 `forward_latent` 的 DiT 内部构成（1080p 每 8 层计时段）：
  **GEMM 52~58% / AWA 13~15% / CPU 循环 28~35%**（分辨率越低 GEMM 占比越高，360p 达 74~81%）。
- CPU 360p fp32 可跑通：DiT 267 s（GEMM 76~79%，AWA=0 因 CPU 走 C++ 注意力参考）；
  **CPU 1080p 不可行** —— 系统内存仅 15.8 GB，而 CPU 模式 net 缓存上限 `cap=1024`
  （≈全权重常驻）+ `fcache` 永久驻留 bin 字节，实测在第 20~24 层 OOM 崩溃（rc=132）。

**关键瓶颈：graph chunk=1 的块间搬运占 DiT 的 ~70%**（1080p 每块 3.3 s 中 extract/下载 ≈2.3 s）。
原因：每块末要把 vid(5440×2560×4B≈56MB) + txt 下载回 CPU，下一块再上传，32 块合计搬 ~3.5 GB 走 PCIe。

**按收益排序的优化方向**：

1. **chunk 合并（首选，零风险）**：块图从 1 层/块 → 2~4 层/块，块数 32→16→8，块间搬运直接减半/减 3/4
   → DiT 估再快 35-45%。已有 fp32 chunk=2 先例，只需重导出 bf16/f16 块图 + 引擎确认每块输出 blob 名随 l1 变化即可。
2. **CPU 内存修复（几行代码）**：CPU 模式 `cap` 改小 + `fcache` 不长期驻留 bin 字节，
   即可让 15.8GB 机器跑通 CPU 1080p。
3. **减少下载量**：块间实际只需 vid/txt 残差流，可尝试半精度传输（省一半 PCIe 带宽，注意累积精度）。
4. **fp32 路径块化**：fp32 也走 graph + 逐块释放（现 chunk=2 1080p OOM），需解决显存或接受 forward_latent。
5. **int8 量化：不建议**。ncnn 的 int8 只对**已量化模型**生效（无运行期开关），且 AWA/AdaCompose/
   RMSNorm/Cast 均无 int8 变体，只能 fp32 包夹 GEMM；扩散模型激活对量化敏感，收益窄、风险高。

---

## 7. 验证体系（保证「改 GPU 不坏 CPU / 改低精度不坏 fp32」）

| 层级 | 手段 | 硬指标 |
|---|---|---|
| AWA 单测 | `test_awa_layer`（SEEDVR_TEST_LP=0/1/2 三精度） | fp32 cos=1.0；fp16/bf16 cos≥0.99999 |
| cast 单测 | cast shader 读写侧自测 | 逐位/近逐位 |
| 块级对拍 | `SEEDVR_DUMP_B0MID`（GPU extract 版）逐 blob dump vs fp32 图 | 量级一致 |
| 端到端图级 | 图 vs forward_latent vs CPU、vs 官方 fp16 latent | cos/PSNR/SSIM/mean\|d\| 多指标 |
| **fp32 零影响红线** | 修低精度前后 fp32 PNG 逐字节 cmp | SAME（本次已验证） |

历史图级结论：CPU 与 GPU forward_latent fp32 互比 cos 0.99999988/80.5dB；graph fp32 vs forward cos 0.99994/42.5dB（graph 融合路径的固有微小差异，源自块内连续执行与逐层路径的算子融合次序不同）；tail 修复后 vs 官方 fp16 全流程 cos 0.999905 / 46.6dB。

---

## 8. 复现命令速查

```bash
# 单图 GPU（默认 fp32 逐位；360p 用 --resolution 360）
./build-cmake/seedvr2_run.exe input/椎名真白.png out.png --resolution 1080 --seed 42

# 低精度 GPU（graph 块路径，bf16 推荐）
./build-cmake/seedvr2_run.exe input/椎名真白.png out_bf16.png --resolution 1080 --bf16

# fp16（已知 32 层累积失真，供对比）
./build-cmake/seedvr2_run.exe input/椎名真白.png out_fp16.png --resolution 1080 --fp16

# CPU（15.8GB RAM 下仅建议 ≤720p；1080p 需先修 §6-4）
./build-cmake/seedvr2_run.exe input/椎名真白.png out_cpu.png --resolution 720 --cpu

# 帧序列（目录）
./build-cmake/seedvr2_run.exe frames/ out_frames/ --resolution 1080 --bf16

# 诊断 env：SEEDVR_DUMP_B0MID[_GPU]=1 / SEEDVR_DUMP_BLOCKOUT=1 / SEEDVR_NO_REUSE=1 /
#          SEEDVR_GRAPH_FP32STORAGE=1（强制 fp32 存储隔离）/ SEEDVR_BLOCK_PREFIX=<前缀>
# 重建：cmake --build build-cmake -j   （shaders 经 glslangValidator 自动编 .spv）
```

---

## 9. 结论

- SeedVR2-ncnn 的 GPU 加速核心 = **合并计算图消除 CPU↔GPU 往返 + 低 16 位存储与 fp32 累加 + Tensor Core**，AWA/AdaCompose/Cast 为三个自研 Vulkan 自定义算子族，全部避开 ncnn 张量层约定、以 flat-float 显式布局实现，并对齐 CPU/官方参考。
- 精度档位选择：**对拍/验证用 fp32（逐位）；生产提速用 bf16（≈fp32 精度）**；fp16 是「全链 16 位存储」的极端档，速度不占优（与 bf16 同受块间 IO 支配）且精度墙明显，仅作研究。
- 下一步最高收益动作：bf16 块图 chunk 1→2/4 合并（预期 DiT 再快 35-50%），随后视需要处理 CPU 内存墙。

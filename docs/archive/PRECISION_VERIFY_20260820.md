# 精度对照进度表 (vs 参考)

对照方法：
- DiT latent: numpy 参考(run_e2e_ref 逻辑, seed=42 同输入 vid_grid.bin) vs engine 输出 latent_sr.bin → cos
- 端到端: ncnn 输出图 vs PyTorch 参考图 bench_pt_480.png → PSNR/cos

## 精度对照结果 (DiT latent cos, 越高越好, 期望 ~0.999+)

| 日期 | 精度 | 图格式 | DiT latent cos | 备注 |
|------|------|--------|----------------|------|
| 2026-08-20 | bf16 | 分块图(旧格式) | -0.138 | 错误, awa.comp online softmax GPU 执行 bug |
| 2026-08-20 | fp32存储 | 分块图(旧格式) | 0.040 | 错误, 同上 |
| 2026-08-20 | (fp32存储) | 整图 single Net | 1.000000 | ✅ 正确 (test_dit_graph) |
| 2026-08-20 | fp32 | 分块图(旧格式) | 0.994 | ✅ **已修复** (awa.comp 改两遍 softmax) |

## 端到端输出 (vs PyTorch 图 bench_pt_480.png)

| 日期 | 精度 | 图格式 | PSNR | cos | 备注 |
|------|------|--------|------|-----|------|
| 2026-08-20 | bf16 | AdaCompose图 | 16.93 | 0.9719 | 模糊 |
| 2026-08-20 | bf16 | 旧格式图 | 17.87 | 0.9772 | 略好仍模糊 |
| 2026-08-20 | fp32 | 分块图(修复后) | 待重测 | 待重测 | 应显著提升 |

## ★ 根因（已定位并修复：awa.comp 的 online softmax GPU 执行 bug）

**经逐层数值对照确认：build_freqs（RoPE 频率）是正确的**（与 mmrope.py cos=1.0）。
之前误判"RoPE 频率错误"，实际根因在 **AWA shader（awa.comp）的 online softmax + 共享内存块加载**。

诊断过程（同输入 vid_grid.bin，权威参考 run_e2e_ref 逻辑）：
1. **build_freqs 正确**：C++ build_freqs vs mmrope.py → cos=1.0，maxdiff=0
2. **qkv 正确**：分块图 b0mid_v_qkv_0 vs 权威 → cos=1.0（同输入）
3. **CPU 复现 shader 逻辑 vs 权威** → cos=1.0（shader 数学逻辑正确）
4. **GPU 执行 shader vs 权威** → cos=0.84（GPU 执行与 CPU 逻辑不一致）
5. **latent 压缩 13 倍 + cos=0.149**（旧 shader 完整输出错误）

修复：awa.comp 从 **online softmax + 共享内存块（sK/sV）** 改为 **两遍 softmax（直接读全局 qkv）**，
与 CPU 正确实现（dit_vk.cpp awa_forward）完全一致。

验证（同输入）：
- block0 attn 输出 vs 权威 → cos=1.0（所有 head）
- 完整 32 层 latent vs 权威 → **cos=0.993777**，range[-8.96, 7.35]（正常，无压缩）

症状: 整体轮廓对、每个 patch 内细节错（latent 压缩 + attention 错位）。

## 待办
- [x] 修复 awa.comp：online softmax → 两遍 softmax（与 CPU 参考一致）
- [x] 修复后重测 DiT latent cos → 0.994 ✅
- [ ] 重测端到端 PSNR → 应显著提升 (目标 25+ dB)
- [ ] 后续扩展 fp16 / fp32 对照

## 最高精度记录
- DiT latent: 整图路径 cos = 1.000000；分块图(修复后) cos = 0.993777
- 端到端: 待重测

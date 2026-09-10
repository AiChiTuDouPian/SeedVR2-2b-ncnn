# 历史归档文档

本目录存放**已被后续工作取代**的历史快照，保留用于追溯，**不要作为当前结论的依据**。

| 文件 | 日期 | 状态 | 说明 |
|---|---|---|---|
| `PERFORMANCE_REPORT_20260820.md` | 2026-08-20 | 已被取代 | 480p 阶段的分块图常驻性能数据（DiT 55.9s → 11.3s）。当时的引擎尚无帧序列模式、无低精度修复、无 emb 槽位修复；当前性能数据见 [`../bench/PERFORMANCE_ANALYSIS.md`](../../bench/PERFORMANCE_ANALYSIS.md) |
| `PRECISION_VERIFY_20260820.md` | 2026-08-20 | 已被取代 | 精度对照**进度表**，当时多项仍标「待重测」，且对拍口径（色彩校正）未对齐。当前对拍结果见 [`../bench/PT_COMPARISON.md`](../../bench/PT_COMPARISON.md) |

## 当前权威文档

- **工程实现与自定义算子**：`docs/GPU加速与自定义算子技术文档.md`
- **性能分析**：`bench/PERFORMANCE_ANALYSIS.md`
- **与 PyTorch 的精度对比**：`bench/PT_COMPARISON.md`

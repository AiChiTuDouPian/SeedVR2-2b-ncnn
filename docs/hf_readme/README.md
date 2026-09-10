# HuggingFace 仓库说明（发布源）

本目录是三个 HuggingFace 权重仓库首页 `README.md` 的**源文件**。
改这里再上传，避免线上内容没有版本记录。

| 本地文件 | HuggingFace 仓库 | 精度 |
|---|---|---|
| `SeedVR2-2b-NCNN.md` | [`xxzigou/SeedVR2-2b-NCNN`](https://huggingface.co/xxzigou/SeedVR2-2b-NCNN) | fp32（基准） |
| `seedVR2-ncnn-bf16.md` | [`xxzigou/seedVR2-ncnn-bf16`](https://huggingface.co/xxzigou/seedVR2-ncnn-bf16) | bf16（推荐） |
| `seedVR2-ncnn-fp16.md` | [`xxzigou/seedVR2-ncnn-fp16`](https://huggingface.co/xxzigou/seedVR2-ncnn-fp16) | fp16 |

三个仓库各自**自包含**（`models/m5` + `models/m6_vae` + 该精度的块图），下载任一个即可运行。

## 更新仓库首页

```bash
hf upload xxzigou/seedVR2-ncnn-bf16 docs/hf_readme/seedVR2-ncnn-bf16.md README.md \
  --repo-type model --commit-message "docs: 更新仓库说明"
```

## 更新权重

权重体积大（合计 71.9 GB），上传脚本在 `diag/hf_upload.sh`（不入库，用法见脚本头注释）：

```bash
HF_TOKEN=hf_xxx bash diag/hf_upload.sh bf16     # fp32 | fp16 | bf16 | all
```

> HuggingFace 的 xet 存储按内容分块去重：基础权重与块图之间、以及不同仓库之间，
> 只要内容有重叠就会跳过实际传输（日志显示 `0.00B transferred`）。
> 因此「每个仓库都带一份基础权重」的实际成本接近 0，不必为了省流量改成增量式。

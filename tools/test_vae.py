#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 VideoAutoencoderKLWrapper 能在 cuda_env 中独立构造并加载真实权重，
且单图 encode/decode 可用（数值有限、roundtrip 合理）。
用法（cuda_env）:
  python tools/test_vae.py
"""
import sys, os
import torch

COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
# 把 ComfyUI 根目录加入 path，使 `src` 作为包可导入（attn_video_vae 内含相对导入 ....common）
sys.path.insert(0, COMFY)
from omegaconf import OmegaConf
from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper

YAML = os.path.join(COMFY, "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml")
SD    = os.path.join(COMFY, "models/SEEDVR2/ema_vae_fp16.safetensors")


def main():
    cfg = OmegaConf.to_container(OmegaConf.load(YAML), resolve=True)
    sdf = cfg.pop("spatial_downsample_factor")
    tdf = cfg.pop("temporal_downsample_factor")
    vae = VideoAutoencoderKLWrapper(
        spatial_downsample_factor=sdf,
        temporal_downsample_factor=tdf,
        freeze_encoder=False,
        **cfg,
    )
    print("VAE constructed OK, params: %.1f M" % (sum(p.numel() for p in vae.parameters()) / 1e6))

    from safetensors.torch import load_file
    sd = load_file(SD, device="cpu")
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print("state_dict: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
    if missing[:5]:
        print("  missing sample:", missing[:5])
    if unexpected[:5]:
        print("  unexpected sample:", unexpected[:5])

    vae = vae.float().eval()
    vae_device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = vae.to(vae_device)
    print("VAE on device:", vae_device)

    # 单图 roundtrip：[B,C,T,H,W]
    # 注意：VAE 把输入映射回自然图像流形，随机噪声输入无法被重建，
    # 故用平滑渐变图（VAE 应当高保真重建）来验证权重确实生效。
    H = W = 64
    yy, xx = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing="ij")
    base = (yy.unsqueeze(0).unsqueeze(0) * 0.6 + xx.unsqueeze(0).unsqueeze(0) * 0.4)
    x = base.repeat(1, 3, 1, 1).unsqueeze(2).to(vae_device)  # (1,3,1,64,64)
    with torch.no_grad():
        z = vae.encode(x).latent          # (1,16,1,8,8)
        print("latent shape:", tuple(z.shape))
        y = vae.decode(z).sample          # (1,3,1,64,64)
        print("recon shape:", tuple(y.shape))
        cos = torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), dim=0).item()
        finite = torch.isfinite(y).all().item()
    print("recon cos(x,y)=%.5f  finite=%s" % (cos, finite))
    assert finite, "decode 输出含 NaN/Inf"
    assert cos > 0.9, "渐变图重建 cos 过低（权重可能未正确加载）"
    print("VAE TEST OK")


if __name__ == "__main__":
    main()

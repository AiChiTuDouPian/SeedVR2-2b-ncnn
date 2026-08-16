#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 VAE 完整 encode/decode 参考（2D 等价），供 C++ verify_vae_full 对比。
输出到 e2e_work_vae/full/：
  in.bin (3,H,W)  mean.bin (16,H/8,W/8)  logvar.bin (16,H/8,W/8)
  latent.bin (16,H/8,W/8)  out.bin (3,H,W)
"""
import sys, os, struct
import numpy as np
import torch
import torch.nn.functional as F

COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from verify_vae_2d import c2d, resnet2d, attention2d, upsample2d, EPS

OUT = os.path.join(ROOT, "e2e_work_vae", "full")


def write_raw(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<q", int(d)))
        f.write(arr.tobytes())


def encoder2d(enc, x):
    x = c2d(enc.conv_in, x)
    for block in enc.down_blocks:
        for rn in block.resnets:
            x = resnet2d(rn, x)
        if block.downsamplers is not None:
            ds = block.downsamplers[0]
            x = F.pad(x, (0, 1, 0, 1))
            x = c2d(ds.conv, x)
    x = resnet2d(enc.mid_block.resnets[0], x)
    x = attention2d(enc.mid_block.attentions[0], x)
    x = resnet2d(enc.mid_block.resnets[1], x)
    x = F.group_norm(x, 32, enc.conv_norm_out.weight, enc.conv_norm_out.bias, EPS)
    x = F.silu(x)
    x = c2d(enc.conv_out, x)
    return x


def decoder2d(dec, z):
    x = c2d(dec.conv_in, z)
    x = resnet2d(dec.mid_block.resnets[0], x)
    x = attention2d(dec.mid_block.attentions[0], x)
    x = resnet2d(dec.mid_block.resnets[1], x)
    for block in dec.up_blocks:
        for rn in block.resnets:
            x = resnet2d(rn, x)
        if block.upsamplers is not None:
            x = upsample2d(block.upsamplers[0], x)
    x = F.group_norm(x, 32, dec.conv_norm_out.weight, dec.conv_norm_out.bias, EPS)
    x = F.silu(x)
    x = c2d(dec.conv_out, x)
    return x


def main():
    from omegaconf import OmegaConf
    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    from safetensors.torch import load_file

    os.makedirs(OUT, exist_ok=True)
    yaml_path = COMFY + "/src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    sdf = cfg.pop("spatial_downsample_factor")
    tdf = cfg.pop("temporal_downsample_factor")
    vae = VideoAutoencoderKLWrapper(spatial_downsample_factor=sdf, temporal_downsample_factor=tdf, freeze_encoder=False, **cfg)
    sd = load_file(COMFY + "/models/SEEDVR2/ema_vae_fp16.safetensors", device="cpu")
    vae.load_state_dict(sd, strict=False)
    vae.eval()

    torch.manual_seed(3)
    H, W = 64, 64
    x = torch.randn(1, 3, H, W) * 0.5  # 小信号，避免溢出
    enc, dec = vae.encoder, vae.decoder

    with torch.no_grad():
        # encode -> 32ch (mean + logvar)
        z32 = encoder2d(enc, x)  # (1,32,8,8)
        mean = z32[:, :16]
        logvar = z32[:, 16:]
        # 采样（固定噪声，seed 固定）
        torch.manual_seed(1000003)
        std = torch.exp(0.5 * logvar)
        latent = mean + std * torch.randn_like(mean)
        # decode
        y = decoder2d(dec, latent)  # (1,3,64,64)

    write_raw(os.path.join(OUT, "in.bin"), x[0].numpy())
    write_raw(os.path.join(OUT, "mean.bin"), mean[0].numpy())
    write_raw(os.path.join(OUT, "logvar.bin"), logvar[0].numpy())
    write_raw(os.path.join(OUT, "latent.bin"), latent[0].numpy())
    write_raw(os.path.join(OUT, "out.bin"), y[0].numpy())
    print(f"full ref: in {tuple(x.shape)} mean {tuple(mean.shape)} latent {tuple(latent.shape)} out {tuple(y.shape)}")
    print("[dump] 完成 ->", OUT)


if __name__ == "__main__":
    main()

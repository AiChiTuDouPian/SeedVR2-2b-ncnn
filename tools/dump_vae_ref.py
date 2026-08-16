#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 VAE 各子图的输入/输出参考（2D 等价），供 C++ verify_vae_vk 对比。
输出到 e2e_work_vae/：
  enc1_in.bin / enc1_out.bin   (3,H,W)->(512,H/8,W/8)
  enc2_in.bin / enc2_out.bin   (512,H/8,W/8)->(32,H/8,W/8)
  dec1_in.bin / dec1_out.bin   (16,H/8,W/8)->(512,H/8,W/8)
  dec2_in.bin / dec2_out.bin   (512,H/8,W/8)->(3,H,W)
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

OUT = os.path.join(ROOT, "e2e_work_vae")


def write_raw(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<q", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<q", int(d)))
        f.write(arr.tobytes())


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

    torch.manual_seed(42)
    H, W = 64, 64
    x = torch.randn(1, 3, H, W)
    enc, dec = vae.encoder, vae.decoder

    with torch.no_grad():
        # ===== encoder =====
        h = c2d(enc.conv_in, x)
        for block in enc.down_blocks:
            for rn in block.resnets:
                h = resnet2d(rn, h)
            if block.downsamplers is not None:
                h = F.pad(h, (0, 1, 0, 1))
                h = c2d(block.downsamplers[0].conv, h)
        h = resnet2d(enc.mid_block.resnets[0], h)  # enc1 输出 = attention 前
        write_raw(os.path.join(OUT, "enc1_in.bin"), x[0].numpy())
        write_raw(os.path.join(OUT, "enc1_out.bin"), h[0].numpy())
        print(f"enc1: {tuple(x.shape)} -> {tuple(h.shape)}")

        h2 = attention2d(enc.mid_block.attentions[0], h)  # attention 后 = enc2 输入
        write_raw(os.path.join(OUT, "enc2_in.bin"), h2[0].numpy())
        h3 = resnet2d(enc.mid_block.resnets[1], h2)
        h3 = F.group_norm(h3, 32, enc.conv_norm_out.weight, enc.conv_norm_out.bias, EPS)
        h3 = F.silu(h3)
        h3 = c2d(enc.conv_out, h3)
        write_raw(os.path.join(OUT, "enc2_out.bin"), h3[0].numpy())
        print(f"enc2: {tuple(h2.shape)} -> {tuple(h3.shape)}")

        # ===== decoder =====
        z = torch.randn(1, 16, H // 8, W // 8)
        d = c2d(dec.conv_in, z)
        d = resnet2d(dec.mid_block.resnets[0], d)  # dec1 输出 = attention 前
        write_raw(os.path.join(OUT, "dec1_in.bin"), z[0].numpy())
        write_raw(os.path.join(OUT, "dec1_out.bin"), d[0].numpy())
        print(f"dec1: {tuple(z.shape)} -> {tuple(d.shape)}")

        d2 = attention2d(dec.mid_block.attentions[0], d)  # attention
        write_raw(os.path.join(OUT, "dec2_in.bin"), d2[0].numpy())
        d3 = resnet2d(dec.mid_block.resnets[1], d2)
        for block in dec.up_blocks:
            for rn in block.resnets:
                d3 = resnet2d(rn, d3)
            if block.upsamplers is not None:
                d3 = upsample2d(block.upsamplers[0], d3)
        d3 = F.group_norm(d3, 32, dec.conv_norm_out.weight, dec.conv_norm_out.bias, EPS)
        d3 = F.silu(d3)
        d3 = c2d(dec.conv_out, d3)
        write_raw(os.path.join(OUT, "dec2_out.bin"), d3[0].numpy())
        print(f"dec2: {tuple(d2.shape)} -> {tuple(d3.shape)}")

    print("[dump] 完成 ->", OUT)


if __name__ == "__main__":
    main()

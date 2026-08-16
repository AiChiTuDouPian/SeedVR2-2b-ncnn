#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 VAE mid attention 的输入/输出参考（mid 分辨率 512 x 136 x 202），供 C++ verify_attn 对比。"""
import sys, os, struct
import numpy as np
import torch

COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from verify_vae_2d import attention2d

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

    torch.manual_seed(7)
    H, W = 136, 202  # mid 分辨率（1080p 输入/8）
    x = torch.randn(1, 512, H, W)

    with torch.no_grad():
        enc = vae.encoder
        dec = vae.decoder
        y_enc = attention2d(enc.mid_block.attentions[0], x)
        y_dec = attention2d(dec.mid_block.attentions[0], x)
    write_raw(os.path.join(OUT, "attn_in.bin"), x[0].numpy())
    write_raw(os.path.join(OUT, "attn_enc_out.bin"), y_enc[0].numpy())
    write_raw(os.path.join(OUT, "attn_dec_out.bin"), y_dec[0].numpy())
    print(f"attn: input {tuple(x.shape)} -> enc {tuple(y_enc.shape)} / dec {tuple(y_dec.shape)}")
    print("[dump] 完成 ->", OUT)


if __name__ == "__main__":
    main()

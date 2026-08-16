#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 run_e2e.py 已生成的 e2e_work/vid_grid.bin + sr_latent.bin 中提取中间结果：
  - noise.bin  : vid_grid 前 16 通道
  - cond.bin   : vid_grid 第 16~31 通道
  - upscaled.bin : noise - sr (x0)
  - raw.png    : VAE decode(upscaled / 0.9152) 后 [-1,1]->[0,1]

用于和官方 run_official_intermediates.py 导出的结果逐阶段对拍。
"""
import sys, os, struct, argparse
import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
SCALING_FACTOR = 0.9152


def read_raw(path):
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = [struct.unpack("<q", fp.read(8))[0] for _ in range(ndim)]
        total = int(np.prod(shape))
        data = np.frombuffer(fp.read(total * 4), dtype=np.float32).reshape(shape)
    return data


def write_raw(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<q", arr.ndim))
        for d in arr.shape:
            fp.write(struct.pack("<q", int(d)))
        fp.write(arr.tobytes())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.join(ROOT, "e2e_work"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "e2e_work", "ncnn_intermediates"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    vid_grid = read_raw(os.path.join(args.workdir, "vid_grid.bin"))  # (1,H,W,33)
    sr = read_raw(os.path.join(args.workdir, "sr_latent.bin"))       # (1,H,W,16)
    print(f"[ncnn] vid_grid={vid_grid.shape} sr={sr.shape}")

    noise = vid_grid[..., :16]
    cond = vid_grid[..., 16:32]
    print(f"[ncnn] noise={noise.shape} cond={cond.shape}")
    write_raw(os.path.join(args.out_dir, "noise.bin"), noise)
    write_raw(os.path.join(args.out_dir, "cond.bin"), cond)

    upscaled = noise - sr
    print(f"[ncnn] upscaled={upscaled.shape}")
    write_raw(os.path.join(args.out_dir, "upscaled.bin"), upscaled)

    # VAE decode
    sys.path.insert(0, COMFY)
    from omegaconf import OmegaConf
    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    from safetensors.torch import load_file

    yaml_path = os.path.join(COMFY, "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    sdf = cfg.pop("spatial_downsample_factor")
    tdf = cfg.pop("temporal_downsample_factor")
    vae = VideoAutoencoderKLWrapper(
        spatial_downsample_factor=sdf, temporal_downsample_factor=tdf,
        freeze_encoder=False, **cfg)
    sd = load_file(os.path.join(COMFY, "models/SEEDVR2/ema_vae_fp16.safetensors"), device="cpu")
    vae.load_state_dict(sd, strict=False)
    vae = vae.eval().to(args.device)

    z_dec = (torch.from_numpy(upscaled).permute(0, 3, 1, 2).to(args.device) / SCALING_FACTOR).half()
    y = vae.decode(z_dec).sample  # (1,3,H,W) [-1,1] 纯 fp16，与官方一致
    y = y.float()
    y = y.clamp(-1, 1).mul(0.5).add_(0.5).clamp(0, 1)
    # 裁回 pad 前真实分辨率（与 official_raw.png 同尺寸便于对比）
    pre_h, pre_w = 1080, 1620
    y = y[:, :, :pre_h, :pre_w]
    out = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(out).save(os.path.join(args.out_dir, "raw.png"))
    print(f"[ncnn] raw 已保存: shape={out.shape[:2]}")


if __name__ == "__main__":
    main()

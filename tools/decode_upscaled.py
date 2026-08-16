#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立 VAE 解码：读取 Vulkan DiT 已产出的 upscaled.bin（SR 潜变量 x0），
做 VAE decode(÷0.9152) -> SR 图(裁剪 padding) -> PNG。
无需重跑 Vulkan DiT（最慢的一步）。

用法（cuda_env）:
  python tools/decode_upscaled.py <upscaled.bin> <output.png>
        [--lr_image <lr.png>]   # 给则额外输出带 LAB 色彩校正的 *_lab.png
        [--pre_h 1080] [--pre_w 1620] [--resolution 1080]
"""
import sys, os, struct, argparse
import numpy as np
import torch
from torchvision.transforms import functional as TVF
from torchvision.transforms import InterpolationMode
from PIL import Image

COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
SCALING_FACTOR = 0.9152


def read_raw(path):
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = [struct.unpack("<q", fp.read(8))[0] for _ in range(ndim)]
        total = int(np.prod(shape))
        data = np.frombuffer(fp.read(total * 4), dtype=np.float32).reshape(shape)
    return data


def resize_shortest_edge(img, size, interpolation=InterpolationMode.BICUBIC):
    h, w = img.shape[-2:]
    if min(h, w) == size:
        return img
    scale = size / float(min(h, w))
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    return TVF.resize(img, (new_h, new_w), interpolation, antialias=True)


def divisible_pad(img, factor=(16, 16)):
    h, w = img.shape[-2:]
    pad_h = (factor[0] - (h % factor[0])) % factor[0]
    pad_w = (factor[1] - (w % factor[1])) % factor[1]
    if pad_h == 0 and pad_w == 0:
        return img, h, w
    img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return img, h, w


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("upscaled_bin")
    ap.add_argument("output")
    ap.add_argument("--lr_image", default="")
    ap.add_argument("--pre_h", type=int, default=1080)
    ap.add_argument("--pre_w", type=int, default=1620)
    ap.add_argument("--resolution", type=int, default=1080)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sys.path.insert(0, COMFY)
    from omegaconf import OmegaConf
    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    yaml_path = os.path.join(COMFY, "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    sdf = cfg.pop("spatial_downsample_factor")
    tdf = cfg.pop("temporal_downsample_factor")
    vae = VideoAutoencoderKLWrapper(
        spatial_downsample_factor=sdf, temporal_downsample_factor=tdf,
        freeze_encoder=False, **cfg)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(COMFY, "models/SEEDVR2/ema_vae_fp16.safetensors"), device="cpu")
    vae.load_state_dict(sd, strict=False)
    vae = vae.eval().half().to(args.device)

    up = read_raw(args.upscaled_bin)                      # (1,H_lat,W_lat,16)
    z_dec = torch.from_numpy(up).permute(0, 3, 1, 2).to(args.device) / SCALING_FACTOR
    z_dec = z_dec.half()
    y = vae.decode(z_dec).sample                          # (1,3,[1,]H_px,W_px)
    y = y.float()
    if y.dim() == 5:
        y = y.squeeze(2)

    # 纯扩散输出（直接对应 Vulkan DiT 产物）
    yr = y.clamp(-1, 1).mul(0.5).add_(0.5).clamp(0, 1)
    yr = yr[:, :, :args.pre_h, :args.pre_w]
    out = (yr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(out).save(args.output)
    print(f"[decode] 输出(raw) {args.output}  shape={out.shape[:2]}")

    # 可选 LAB 色彩校正（对齐官方 Phase4 默认）
    if args.lr_image:
        img = Image.open(args.lr_image).convert("RGB")
        x = TVF.to_tensor(img)
        x = resize_shortest_edge(x, args.resolution)
        x = torch.clamp(x, 0.0, 1.0)
        x, _, _ = divisible_pad(x, (16, 16))
        x = TVF.normalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        x = x.unsqueeze(0).to(args.device)
        sys.path.insert(0, COMFY)
        from src.utils.color_fix import lab_color_transfer
        from src.utils.debug import Debug
        yl = lab_color_transfer(y, x, Debug(enabled=False), luminance_weight=0.8)
        yl = yl.clamp(-1, 1).mul(0.5).add_(0.5).clamp(0, 1)
        yl = yl[:, :, :args.pre_h, :args.pre_w]
        outl = (yl.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        lab_path = args.output.rsplit(".", 1)[0] + "_lab.png"
        Image.fromarray(outl).save(lab_path)
        print(f"[decode] 输出(lab) {lab_path}  shape={outl.shape[:2]}")


if __name__ == "__main__":
    main()

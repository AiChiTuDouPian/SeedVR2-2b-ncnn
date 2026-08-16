#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
checkerboard 去卷积验证：读 workdir 的 upscaled.bin，做 2x2 patch 对角低通（去棋盘格），
再 VAE decode + LAB 校正出图。alpha=1 完全消除对角 Nyquist 分量。
"""
import sys, os, struct, argparse
import numpy as np
import torch
from torchvision.transforms import functional as TVF
from torchvision.transforms import InterpolationMode
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
sys.path.insert(0, COMFY)
SCALING_FACTOR = 0.9152


def read_raw(path):
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = [struct.unpack("<q", fp.read(8))[0] for _ in range(ndim)]
        return np.frombuffer(fp.read(int(np.prod(shape)) * 4), dtype=np.float32).reshape(shape)


def decheckerboard(x, alpha=1.0):
    """x: (T, H, W, C) channel-last。对每个 2x2 patch 做对角低通消除棋盘格。"""
    x = x.copy()
    a = x[:, 0::2, 0::2, :]  # (0,0)
    b = x[:, 0::2, 1::2, :]  # (0,1)
    c = x[:, 1::2, 0::2, :]  # (1,0)
    d = x[:, 1::2, 1::2, :]  # (1,1)
    checker = (a + d - b - c) / 4.0  # 对角 Nyquist 分量
    x[:, 0::2, 0::2, :] = a - alpha * checker
    x[:, 0::2, 1::2, :] = b + alpha * checker
    x[:, 1::2, 0::2, :] = c + alpha * checker
    x[:, 1::2, 1::2, :] = d - alpha * checker
    return x


def bilateral_latent(x, d=3, sigma_color=1.0, sigma_space=1.5):
    """逐通道双边滤波（保边去颗粒）。x: (T,H,W,C) channel-last。"""
    import cv2
    out = x.copy()
    for c in range(x.shape[-1]):
        out[0, :, :, c] = cv2.bilateralFilter(x[0, :, :, c], d, sigma_color, sigma_space)
    return out


def unsharp_latent(x, amount=0.5, sigma=1.0):
    """unsharp mask 锐化（在 latent 空间）。"""
    from scipy.ndimage import gaussian_filter
    blur = gaussian_filter(x, sigma=(0, sigma, sigma, 0))
    return x + amount * (x - blur)


def resize_shortest_edge(img, size, interpolation=InterpolationMode.BICUBIC):
    h, w = img.shape[-2:]
    if min(h, w) == size:
        return img
    scale = size / float(min(h, w))
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    return TVF.resize(img, (new_h, new_w), interpolation, antialias=True)


def divisible_pad(x, block):
    h, w = x.shape[-2], x.shape[-1]
    ph = (block[0] - h % block[0]) % block[0]
    pw = (block[1] - w % block[1]) % block[1]
    return torch.nn.functional.pad(x, (0, pw, 0, ph)), h, w


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lr_image")
    ap.add_argument("output_image")
    ap.add_argument("--workdir", default=os.path.join(ROOT, "e2e_work_sadhu"))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gaussian-sigma", type=float, default=0.0, help="对 latent 做空间高斯低通（模拟官方 bf16 抹平高频）")
    ap.add_argument("--bilateral", action="store_true", help="对 latent 逐通道做双边滤波（保边去颗粒）")
    ap.add_argument("--bil-d", type=int, default=3)
    ap.add_argument("--bil-color", type=float, default=1.0)
    ap.add_argument("--bil-space", type=float, default=1.5)
    ap.add_argument("--sharpen", type=float, default=0.0, help="unsharp mask 锐化强度（0=关闭）")
    ap.add_argument("--post-sharpen", type=float, default=0.0, help="decode 后图像空间 unsharp 锐化强度（0=关闭）")
    ap.add_argument("--resolution", type=int, default=1080)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # ---- 读 upscaled 并去 checkerboard ----
    upscaled = read_raw(os.path.join(args.workdir, "upscaled.bin"))  # (1,H,W,16)
    print(f"[dechecker] upscaled {upscaled.shape} -> alpha={args.alpha} gaussian={args.gaussian_sigma} "
          f"bilateral={args.bilateral} sharpen={args.sharpen}")
    upscaled = decheckerboard(upscaled, args.alpha)
    if args.bilateral:
        upscaled = bilateral_latent(upscaled, args.bil_d, args.bil_color, args.bil_space)
    if args.gaussian_sigma > 0:
        from scipy.ndimage import gaussian_filter
        upscaled = gaussian_filter(upscaled, sigma=(0, args.gaussian_sigma, args.gaussian_sigma, 0))
    if args.sharpen > 0:
        upscaled = unsharp_latent(upscaled, args.sharpen)

    # ---- 载入 VAE ----
    from omegaconf import OmegaConf
    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    yaml_path = os.path.join(COMFY, "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    sdf = cfg.pop("spatial_downsample_factor"); tdf = cfg.pop("temporal_downsample_factor")
    vae = VideoAutoencoderKLWrapper(spatial_downsample_factor=sdf, temporal_downsample_factor=tdf, freeze_encoder=False, **cfg)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(COMFY, "models/SEEDVR2/ema_vae_fp16.safetensors"), device="cpu")
    vae.load_state_dict(sd, strict=False)
    vae = vae.eval().half().to(args.device)

    # ---- 预处理 LR（用于 LAB 校正） ----
    img = Image.open(args.lr_image).convert("RGB")
    x = TVF.to_tensor(img)
    x = resize_shortest_edge(x, args.resolution)
    x = torch.clamp(x, 0.0, 1.0)
    x, pre_h, pre_w = divisible_pad(x, (16, 16))
    x = TVF.normalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).unsqueeze(0).to(args.device)

    # ---- VAE decode ----
    z_dec = (torch.from_numpy(upscaled).permute(0, 3, 1, 2).to(args.device) / SCALING_FACTOR).half()
    y = vae.decode(z_dec).sample.float()

    # ---- LAB 色彩校正 ----
    from src.utils.color_fix import lab_color_transfer
    from src.utils.debug import Debug
    y = lab_color_transfer(y, x, Debug(enabled=False), luminance_weight=0.8)
    y = y.clamp(-1, 1).mul(0.5).add_(0.5).clamp(0, 1)
    y = y[:, :, :pre_h, :pre_w]
    out = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    if args.post_sharpen > 0:
        import cv2
        blur = cv2.GaussianBlur(out, (0, 0), 2.0)
        out = np.clip(out.astype(np.float32) + args.post_sharpen * (out.astype(np.float32) - blur), 0, 255).astype(np.uint8)
    Image.fromarray(out).save(args.output_image)
    print(f"[dechecker] 输出已保存: {args.output_image}  shape={out.shape[:2]}")


if __name__ == "__main__":
    sys.exit(main())

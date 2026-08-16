#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断：用【官方】的 cond + noise 喂给 ncnn DiT exe，看产出的 upscaled latent / 解码图
是否与官方 x0 / raw 对齐。
  - 若对齐（cos~0.999）：说明 DiT 引擎本身正确，问题仅在 ncnn 的 cond 生成
    （预处理或 shift 项），只需修正 run_e2e.py。
  - 若不对齐：说明 exe 内部对 cond 的处理还有 bug，需进一步查 dit_vk.cpp。
"""
import sys, os, struct, argparse, subprocess
import numpy as np
import torch
from torchvision.transforms import functional as TVF
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
EXE = os.path.join(ROOT, "seedvr2_dit_vk_run.exe")
MODELDIR = os.path.join(ROOT, "models", "m5")
OFF = os.path.join(ROOT, "e2e_work", "official_intermediates")
NC = os.path.join(ROOT, "e2e_work", "ncnn_intermediates")
WORK = os.path.join(ROOT, "e2e_work")
SCALING_FACTOR = 0.9152


def read_raw(p):
    with open(p, "rb") as f:
        nd = struct.unpack("<q", f.read(8))[0]
        sh = [struct.unpack("<q", f.read(8))[0] for _ in range(nd)]
        return np.frombuffer(f.read(int(np.prod(sh)) * 4), dtype=np.float32).reshape(sh)


def write_raw(p, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(p, "wb") as f:
        f.write(struct.pack("<q", arr.ndim))
        for d in arr.shape:
            f.write(struct.pack("<q", int(d)))
        f.write(arr.tobytes())


def load_vae(device):
    sys.path.insert(0, COMFY)
    from omegaconf import OmegaConf
    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    yaml_path = os.path.join(COMFY, "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    sdf = cfg.pop("spatial_downsample_factor"); tdf = cfg.pop("temporal_downsample_factor")
    vae = VideoAutoencoderKLWrapper(spatial_downsample_factor=sdf, temporal_downsample_factor=tdf,
                                    freeze_encoder=False, **cfg)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(COMFY, "models/SEEDVR2/ema_vae_fp16.safetensors"), device="cpu")
    vae.load_state_dict(sd, strict=False)
    return vae.eval().to(device)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_ncnn_cond", action="store_true",
                    help="默认用官方 cond+noise；加此 flag 用 ncnn 的（复现原问题）")
    args = ap.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    noise = read_raw(os.path.join(OFF, "noise.bin"))          # (1,H,W,16)
    if args.use_ncnn_cond:
        cond = read_raw(os.path.join(NC, "cond.bin"))
        tag = "ncnn"
    else:
        cond = read_raw(os.path.join(OFF, "cond.bin"))
        tag = "official"

    mask = np.ones((1, cond.shape[1], cond.shape[2], 1), dtype=np.float32)
    cond17 = np.concatenate([cond, mask], axis=-1)
    vid_grid = np.concatenate([noise, cond17], axis=-1)        # (1,H,W,33)
    write_raw(os.path.join(WORK, "vid_grid.bin"), vid_grid)
    print(f"[diag] 用 {tag} cond 构造 vid_grid {vid_grid.shape}")

    cmd = [EXE, WORK, MODELDIR]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[diag][FAIL]\n", r.stderr); sys.exit(1)

    sr = read_raw(os.path.join(WORK, "sr_latent.bin"))
    upscaled = noise - sr
    write_raw(os.path.join(WORK, "upscaled_from_%s.bin" % tag), upscaled)

    # ---- 与官方 x0 对比（latent 级） ----
    x0 = read_raw(os.path.join(OFF, "x0.bin"))
    a = upscaled.astype(np.float64).ravel(); b = x0.astype(np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mse = float(np.mean((a - b) ** 2))
    maxd = float(np.max(np.abs(a - b)))
    print(f"[diag][{tag}] upscaled vs official x0: cos={cos:.6f} MSE={mse:.6e} max|diff|={maxd:.4f}")

    # ---- 解码并对比 raw ----
    vae = load_vae(device)
    z_dec = torch.from_numpy(upscaled).permute(0, 3, 1, 2).to(device) / SCALING_FACTOR
    with torch.autocast(device, dtype=torch.float16):
        y = vae.decode(z_dec).sample.float()
    y = y.clamp(-1, 1).mul(0.5).add_(0.5).clamp(0, 1)
    y = y[:, :, :1080, :1620]
    out = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    raw_path = os.path.join(WORK, "diag_raw_from_%s.png" % tag)
    Image.fromarray(out).save(raw_path)
    print(f"[diag] 解码图已保存: {raw_path}")

    oraw = np.asarray(Image.open(os.path.join(OFF, "raw.png")), dtype=np.float32) / 255.0
    nr = out.astype(np.float32) / 255.0
    aa = oraw.astype(np.float64).ravel(); bb = nr.astype(np.float64).ravel()
    cos2 = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))
    mse2 = float(np.mean((aa - bb) ** 2))
    print(f"[diag][{tag}] decoded raw vs official raw: cos={cos2:.6f} MSE={mse2:.6e}")


if __name__ == "__main__":
    main()

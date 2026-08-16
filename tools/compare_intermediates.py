#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐阶段比较官方与 ncnn 的中间结果。
"""
import sys, os, struct, argparse
import numpy as np
from PIL import Image


def read_raw(path):
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = [struct.unpack("<q", fp.read(8))[0] for _ in range(ndim)]
        total = int(np.prod(shape))
        data = np.frombuffer(fp.read(total * 4), dtype=np.float32).reshape(shape)
    return data


def stats(a, b, name):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    psnr = float(20 * np.log10(2.0 / (np.sqrt(mse) + 1e-12)))  # [-1,1] range => max=2
    if psnr > 100:
        psnr = float('inf')
    cos = float(np.dot(a.ravel(), b.ravel()) /
                (np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()) + 1e-12))
    maxdiff = float(np.max(np.abs(a - b)))
    print(f"[{name}] shape={tuple(a.shape)} MSE={mse:.6e} PSNR={psnr:.3f} cos={cos:.6f} max|diff|={maxdiff:.6f}")
    return mse, psnr, cos, maxdiff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("official_dir")
    ap.add_argument("ncnn_dir")
    args = ap.parse_args()

    print(f"=== 官方: {args.official_dir} ===")
    print(f"=== ncnn: {args.ncnn_dir} ===")

    # ---- latent 阶段 ----
    o_cond = read_raw(os.path.join(args.official_dir, "cond.bin"))
    n_cond = read_raw(os.path.join(args.ncnn_dir, "cond.bin"))
    stats(o_cond, n_cond, "cond")

    o_noise = read_raw(os.path.join(args.official_dir, "noise.bin"))
    n_noise = read_raw(os.path.join(args.ncnn_dir, "noise.bin"))
    stats(o_noise, n_noise, "noise")

    o_x0 = read_raw(os.path.join(args.official_dir, "x0.bin"))
    n_x0 = read_raw(os.path.join(args.ncnn_dir, "upscaled.bin"))
    stats(o_x0, n_x0, "x0/upscaled")

    # ---- decode raw 图像阶段（裁到相同尺寸） ----
    o_raw = np.asarray(Image.open(os.path.join(args.official_dir, "raw.png")), dtype=np.float32) / 255.0
    n_raw_full = np.asarray(Image.open(os.path.join(args.ncnn_dir, "raw.png")), dtype=np.float32) / 255.0
    h, w = o_raw.shape[:2]
    n_raw = n_raw_full[:h, :w]
    stats(o_raw, n_raw, "decoded raw")

    # 同时输出一个差值图便于目视
    diff = np.abs(o_raw - n_raw)
    diff_img = (diff / (diff.max() + 1e-8) * 255).astype(np.uint8)
    out_path = os.path.join(args.ncnn_dir, "diff_vs_official_raw.png")
    Image.fromarray(diff_img).save(out_path)
    print(f"差值图已保存: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简单图像对比：打印两张图的 MSE / PSNR / cosine，并输出叠加差值图。

用法：
  python tools/compare_images.py <img_a> <img_b> [--diff <diff_out.png>]
"""
import argparse
import numpy as np
from PIL import Image


def load(p):
    img = Image.open(p).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("img_a")
    ap.add_argument("img_b")
    ap.add_argument("--diff", default=None)
    args = ap.parse_args()

    a = load(args.img_a)
    b = load(args.img_b)
    if a.shape != b.shape:
        print(f"[warn] 形状不同 {a.shape} vs {b.shape}, resize b")
        b = np.array(Image.fromarray((b * 255).astype(np.uint8)).resize((a.shape[1], a.shape[0])))
        b = b.astype(np.float32) / 255.0

    mse = np.mean((a - b) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 999.0
    cos = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
    maxd = np.max(np.abs(a - b))
    print(f"[{args.img_a.split('/')[-1]}] vs [{args.img_b.split('/')[-1]}]")
    print(f"  shape={a.shape}  MSE={mse:.6f}  PSNR={psnr:.3f}  cos={cos:.6f}  max|diff|={maxd:.4f}")

    if args.diff:
        d = np.abs(a - b)
        d = np.clip(d * 8.0, 0, 1)
        out = (d * 255).astype(np.uint8)
        Image.fromarray(out).save(args.diff)
        print(f"  差值图已保存: {args.diff}")


if __name__ == "__main__":
    main()

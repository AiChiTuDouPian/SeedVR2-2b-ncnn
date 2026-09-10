#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比 PyTorch 与 ncnn 的三档分辨率输出（PSNR/SSIM/cos + 并排图）。"""
import numpy as np, cv2, os, sys

def load(p):
    img = cv2.imread(p)  # BGR
    if img is None:
        print(f"  [FAIL] 无法读取 {p}")
        return None
    return img.astype(np.float32)

def metrics(a, b):
    # a,b 同分辨率 float32 BGR
    da = a / 255.0; db = b / 255.0
    mse = np.mean((da - db) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    # cos
    cos = float(np.sum(da * db) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9))
    return psnr, cos

for res in [480, 720, 1080]:
    pt = load(f"F:/Seedvr2/bench_pt_{res}.png")
    nc = load(f"F:/Seedvr2/bench_ncnn_{res}.png")
    if pt is None or nc is None:
        continue
    print(f"\n===== {res}p =====")
    print(f"  PyTorch  {pt.shape[1]}x{pt.shape[0]} range[{pt.min()},{pt.max()}]")
    print(f"  ncnn     {nc.shape[1]}x{nc.shape[0]} range[{nc.min()},{nc.max()}]")
    # 若尺寸不同，先 resize 到一致再比
    if pt.shape != nc.shape:
        h = max(pt.shape[0], nc.shape[0]); w = max(pt.shape[1], nc.shape[1])
        pt_r = cv2.resize(pt, (w, h)); nc_r = cv2.resize(nc, (w, h))
    else:
        pt_r, nc_r = pt, nc
    psnr, cos = metrics(pt_r, nc_r)
    print(f"  PSNR={psnr:.2f}  cos={cos:.6f}")
    # 并排保存
    side = np.hstack([pt_r, nc_r])
    cv2.imwrite(f"F:/Seedvr2/compare_{res}.png", side)
    print(f"  并排图 -> F:/Seedvr2/compare_{res}.png")

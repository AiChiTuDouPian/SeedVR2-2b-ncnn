#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查 ncnn 各 dump 文件的尺寸关系，验证 Lv / H8 / W8 是否一致。"""
import struct, numpy as np, os

def read_raw(p):
    with open(p, 'rb') as f:
        ndim = struct.unpack('<q', f.read(8))[0]
        sh = struct.unpack('<%dq' % ndim, f.read(8 * ndim))
        d = np.frombuffer(f.read(), np.float32)
    return sh, d

def read_vidgrid(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        H = struct.unpack('<i', f.read(4))[0]
        W = struct.unpack('<i', f.read(4))[0]
        d = np.frombuffer(f.read(), np.float32)
    return n, H, W, d

print("===== vid_grid.bin =====")
n, H, W, vg = read_vidgrid('vid_grid.bin')
print(f"header n={n} H={H} W={W} floats={len(vg)} expect={H*W*33}")
print(f"H8*W8 = {H*W}   patch后 Lv=(H/2)*(W/2) = {(H//2)*(W//2)}")

print("\n===== block0_vid.bin (raw float32) =====")
b0 = np.fromfile('block0_vid.bin', dtype=np.float32)
print(f"floats={len(b0)}  → 若 DIM=2560: Lv={len(b0)/2560}")

print("\n===== latent_sr.bin =====")
ls = np.fromfile('latent_sr.bin', dtype=np.float32)
print(f"floats={len(ls)}  → 若 16ch: H8*W8={len(ls)/16}")

print("\n===== txt.bin (engine reads) =====")
sh, txt = read_raw('models/m5/txt.bin')
print(f"shape={sh} floats={len(txt)}")

print("\n===== 结论核对 =====")
print(f"vid_grid H*W={H*W} 但 patch 后 Lv 应={(H//2)*(W//2)}")
print(f"block0 dump Lv={len(b0)/2560}")
print(f"两者是否一致: {(H//2)*(W//2) == len(b0)/2560}")

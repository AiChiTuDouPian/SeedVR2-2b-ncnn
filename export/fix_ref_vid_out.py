#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重建 m5/ref_vid_out.bin（正确 vid_out_ada 槽位 = attn 组 d*6+{0,1}）。

背景：m5_export.py 旧代码用 emb[:, :D*3]（d*3+g）生成 ref_vid_out.bin，
在 6 槽 emb 布局下是错的 → verify_dit 的 FINAL 检查与官方语义不符。
此脚本只重算尾部（输入用导出时保存的 vid_b31.bin），不改任何运行时权重文件。
运行：python export/fix_ref_vid_out.py   （cwd = seedvr2-ncnn 根）
"""
import os, sys, struct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "export"))
import ncnn_io as io

M5 = "models/m5"
EPS = 1e-5
DIM = 2560

def rd_raw(p):
    b = open(p, "rb").read()
    nd = struct.unpack("<q", b[:8])[0]
    dims = struct.unpack("<%dq" % nd, b[8:8 + 8 * nd])
    return np.frombuffer(b[8 + 8 * nd:], dtype="<f4").reshape(dims)

def save_raw(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<q", arr.ndim))
        for d in arr.shape:
            fp.write(struct.pack("<q", int(d)))
        fp.write(arr.tobytes())

# 尾部输入 = 导出链路 block31 之后的隐藏态（与 m5_export 相同源）
vid = rd_raw(os.path.join(M5, "vid_b31.bin"))            # (Lv,2560)
Lv = vid.shape[0]
Wvon    = rd_raw(os.path.join(M5, "vid_out_norm_w.bin")).reshape(-1)
Wvoa_s  = rd_raw(os.path.join(M5, "vid_out_ada_shift.bin")).reshape(-1)
Wvoa_sc = rd_raw(os.path.join(M5, "vid_out_ada_scale.bin")).reshape(-1)
emb = rd_raw(os.path.join(M5, "emb.bin")).reshape(-1)     # (15360,)
print("vid_b31", vid.shape, "Wvon", Wvon.shape, "emb", emb.shape)

with io.open_sd() as f:
    Wvout = io.get_top_weight(f, "vid_out.proj", "weight")  # (64,2560)
    Bvout = io.get_top_weight(f, "vid_out.proj", "bias")    # (64,)
print("Wvout", Wvout.shape, "Bvout", Bvout.shape)

# ---- vid_out_norm (affine) ----
ms = np.mean(vid * vid, axis=-1, keepdims=True)
vn = vid / np.sqrt(ms + EPS) * Wvon

# ---- vid_out_ada (in): 取 attn 组 l=0；emb (15360,)->(DIM,2,3) -> [d,0,0]=emb[d*6], [d,0,1]=emb[d*6+1]
emb3 = emb.reshape(DIM, 2, 3)
sAo  = emb3[:, 0, 0]            # shift  = emb[d*6+0]
scAo = emb3[:, 0, 1]            # scale  = emb[d*6+1]
vn = vn * (scAo + Wvoa_sc) + (sAo + Wvoa_s)

# ---- vid_out.proj (64)
out = vn @ Wvout.T + Bvout       # (Lv,64)
assert out.shape == (Lv, 64)
save_raw(os.path.join(M5, "ref_vid_out.bin"), out)
print("ref_vid_out.bin 已重建: shape=%s  (原文件已覆盖；若要备份请从 git/历史取)" % (out.shape,))

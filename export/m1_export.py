#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M1 导出：NaPatchIn/Out、txt_in、TimeEmbedding(emb_in)、adaLN 参数导出为 ncnn(fp16)；
并生成 C++ 对拍参考（raw 格式：8字节 magic? 不，简单头）：
  raw 文件格式 = int64 ndim + int64[ndim] shape + float32 数据(row-major)
参考文件:
  vid_in_ref.bin  : vid_seq(L,33) + vid_shape(3,) + expected(L',2560)
  vid_out_ref.bin : hid(L',2560) + expected(L,16)
  rmsnorm_ref.bin : x(N,128) + weight(128,) + expected(N,128) + eps(float32 scalar)
"""
import os
import numpy as np
import struct
from einops import rearrange
from safetensors import safe_open

import ncnn_io as io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M1 = os.path.join(ROOT, "models", "m1")
os.makedirs(M1, exist_ok=True)


def save_raw(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<q", arr.ndim))
        for d in arr.shape:
            fp.write(struct.pack("<q", int(d)))
        fp.write(arr.tobytes())


def load_ref(path):
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = tuple(struct.unpack("<q", fp.read(8))[0] for _ in range(ndim))
        data = fp.read()
        arr = np.frombuffer(data, dtype=np.float32).reshape(shape)
    return arr


def export_top_linear(name, module, with_bias=True):
    with io.open_sd() as f:
        w = io.get_top_weight(f, module, "weight")
        b = io.get_top_weight(f, module, "bias") if with_bias else None
    out = os.path.join(M1, name)
    io.export_linear(out, w, b, fp16=True)
    print(f"[ok] 导出 {name}  Linear {w.shape} -> {out}.param/.bin")
    return w, b


def main():
    # ---------- 1) NaPatchIn: patchify(2x2) + vid_in.proj ----------
    T, H, W = 1, 4, 4
    c_in = 33
    with io.open_sd() as f:
        w_lin = io.get_top_weight(f, "vid_in.proj", "weight")   # (2560, 132)
        b_lin = io.get_top_weight(f, "vid_in.proj", "bias")     # (2560,)
    rng = np.random.RandomState(0)
    vid = rng.randn(T * H * W, c_in).astype(np.float32)  # (16, 33)
    vid3 = vid.reshape(T, H, W, c_in)
    patches = (vid3
               .reshape(T, H // 2, 2, W // 2, 2, c_in)
               .transpose(0, 1, 3, 2, 4, 5)
               .reshape(T * (H // 2) * (W // 2), 2 * 2 * c_in))  # (4, 132)
    expected_in = patches @ w_lin.T + b_lin  # (4, 2560)
    io.export_linear(os.path.join(M1, "vid_in_proj"), w_lin, b_lin, fp16=True)
    save_raw(os.path.join(M1, "vid_in_vidseq.bin"), vid)
    save_raw(os.path.join(M1, "vid_in_shape.bin"), np.array([T, H, W], dtype=np.int64).view(np.float32))
    save_raw(os.path.join(M1, "vid_in_expected.bin"), expected_in)
    print(f"[ok] vid_in: patchify+linear 参考 (patch={patches.shape}, out={expected_in.shape})")

    # ---------- 2) NaPatchOut: vid_out.proj + unpatchify ----------
    c_out = 16
    with io.open_sd() as f:
        w_out = io.get_top_weight(f, "vid_out.proj", "weight")   # (64, 2560)
        b_out = io.get_top_weight(f, "vid_out.proj", "bias")     # (64,)
    hid = rng.randn(T * (H // 2) * (W // 2), 2560).astype(np.float32)
    out4 = hid @ w_out.T + b_out  # (4, 64)
    # proj 输出是 2D (L', t*h*w*c_out)，先 reshape 成 5D (b,T,H',W',t*h*w*c_out)
    out4_5 = out4.reshape(1, T, H // 2, W // 2, c_out * 1 * 2 * 2)  # (1,1,2,2,64)
    out3 = rearrange(out4_5, "b T H W (t h w c) -> b c (T t) (H h) (W w)",
                     t=1, h=2, w=2, c=c_out)
    expected_out = out3.reshape(T * H * W, c_out)  # (16, 16)
    io.export_linear(os.path.join(M1, "vid_out_proj"), w_out, b_out, fp16=True)
    save_raw(os.path.join(M1, "vid_out_hid.bin"), hid)
    save_raw(os.path.join(M1, "vid_out_expected.bin"), expected_out)
    print(f"[ok] vid_out: linear+unpatchify 参考 (out={expected_out.shape})")

    # ---------- 3) txt_in ----------
    export_top_linear("txt_in", "txt_in", with_bias=True)

    # ---------- 4) TimeEmbedding emb_in ----------
    with io.open_sd() as f:
        for sub in ["proj_in", "proj_hid", "proj_out"]:
            w = io.get_top_weight(f, f"emb_in.{sub}", "weight")
            b = io.get_top_weight(f, f"emb_in.{sub}", "bias")
            io.export_linear(os.path.join(M1, f"emb_in_{sub}"), w, b, fp16=True)
            print(f"[ok] 导出 emb_in.{sub}  Linear {w.shape}")

    # ---------- 5) adaLN params ----------
    with io.open_sd() as f:
        out_scale = io.get_top_weight(f, "vid_out_ada", "out_scale")
        out_shift = io.get_top_weight(f, "vid_out_ada", "out_shift")
        vnorm_w = io.get_top_weight(f, "vid_out_norm", "weight")
    save_raw(os.path.join(M1, "ada_out_scale.bin"), out_scale)
    save_raw(os.path.join(M1, "ada_out_shift.bin"), out_shift)
    save_raw(os.path.join(M1, "vnorm_w.bin"), vnorm_w)
    print(f"[ok] vid_out_ada / vid_out_norm 参数已存")

    # ---------- 6) fusedrms RMSNorm 参考 ----------
    with io.open_sd() as f:
        rn_w = io.get_top_weight(f, "blocks.0.attn.norm_q.vid", "weight")  # (128,)
    x = rng.randn(7, 128).astype(np.float32)
    eps = np.float32(1e-6)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    expected_rn = x / np.sqrt(ms + eps) * rn_w
    save_raw(os.path.join(M1, "rmsnorm_x.bin"), x)
    save_raw(os.path.join(M1, "rmsnorm_w.bin"), rn_w)
    save_raw(os.path.join(M1, "rmsnorm_expected.bin"), expected_rn)
    save_raw(os.path.join(M1, "rmsnorm_eps.bin"), eps.reshape(1))
    print(f"[ok] fusedrms RMSNorm 参考 (x={x.shape}, w={rn_w.shape})")

    print("\n[M1 导出完成] 文件在 models/m1/")


if __name__ == "__main__":
    main()

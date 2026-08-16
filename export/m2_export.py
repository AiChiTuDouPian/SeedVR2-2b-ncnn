#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M2 导出：单个 block 的 attn(qkv/qk_norm/out) + mlp(SwiGLU) 子图权重；
生成 C++ 对拍参考（非窗口 attention，window=(1,1,1) 退化 -> 全局 attention，不含 RoPE）。
验证 block 0(dual/vid) 与 block 20(shared/.all) 两套。

attn 参考（无 RoPE，隔离 Linear+attention 正确性）：
  qkv = X @ Wqkv.T                       # (L, 7680)
  q,k,v = split 3x2560
  q = rmsnorm_per_head(q, wq); k = rmsnorm_per_head(k, wk)   # head_dim=128
  for h: scores = q_h @ k_h^T / sqrt(128); softmax; out_h = scores @ v_h
  attn = out.reshape(L,2560) @ Wout.T + bout

mlp 参考（SwiGLU）：
  h = silu(X @ Win_gate.T) * (X @ Win.T); out = h @ Wout.T + bout
"""
import os
import numpy as np
import struct
from safetensors import safe_open

import ncnn_io as io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M2 = os.path.join(ROOT, "models", "m2")
os.makedirs(M2, exist_ok=True)

HEADS = 20
HEAD_D = 128
EPS = 1e-6


def save_raw(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<q", arr.ndim))
        for d in arr.shape:
            fp.write(struct.pack("<q", int(d)))
        fp.write(arr.tobytes())


def rmsnorm(x, w, eps=EPS):
    # x: (...,128), w:(128,)
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * w


def add_bias(y, B):
    """线性层加偏置；SeedVR2 多数投影无 bias(B=None) 时直接返回 y。"""
    return y + B if B is not None else y


def export_block(block, branch):
    tag = f"block{block}"
    with io.open_sd() as f:
        # attn 模块：branch 在 sub 之后 (attn.proj_qkv.vid.weight)
        Wqkv = io.get_block_weight(f, block, "attn.proj_qkv", "weight", branch)
        Bqkv = io.get_block_weight(f, block, "attn.proj_qkv", "bias", branch)
        Wout = io.get_block_weight(f, block, "attn.proj_out", "weight", branch)
        Bout = io.get_block_weight(f, block, "attn.proj_out", "bias", branch)
        wq = io.get_block_weight(f, block, "attn.norm_q", "weight", branch)
        wk = io.get_block_weight(f, block, "attn.norm_k", "weight", branch)
        # mlp 模块：branch 在 sub 之前 (mlp.vid.proj_in.weight)
        Win = io.load_key(f, io.block_key(block, "proj_in", branch, "weight", "mlp"))
        Bin = io.load_key(f, io.block_key(block, "proj_in", branch, "bias", "mlp"))
        Wing = io.load_key(f, io.block_key(block, "proj_in_gate", branch, "weight", "mlp"))
        Bing = io.load_key(f, io.block_key(block, "proj_in_gate", branch, "bias", "mlp"))
        Wmout = io.load_key(f, io.block_key(block, "proj_out", branch, "weight", "mlp"))
        Bmout = io.load_key(f, io.block_key(block, "proj_out", branch, "bias", "mlp"))

    p = os.path.join(M2, tag)
    io.export_linear(p + "_attn_qkv", Wqkv, Bqkv, fp16=True)
    io.export_linear(p + "_attn_out", Wout, Bout, fp16=True)
    io.export_linear(p + "_mlp_in", Win, Bin, fp16=True)
    io.export_linear(p + "_mlp_in_gate", Wing, Bing, fp16=True)
    io.export_linear(p + "_mlp_out", Wmout, Bmout, fp16=True)
    save_raw(p + "_normq.bin", wq)
    save_raw(p + "_normk.bin", wk)

    # ---- attn 参考 ----
    rng = np.random.RandomState(block + 1)
    L = 8
    X = rng.randn(L, 2560).astype(np.float32)
    qkv = add_bias(X @ Wqkv.T, Bqkv)        # (L, 7680)
    q, k, v = qkv[:, :2560], qkv[:, 2560:5120], qkv[:, 5120:]
    q = rmsnorm(q.reshape(L, HEADS, HEAD_D), wq).reshape(L, 2560)
    k = rmsnorm(k.reshape(L, HEADS, HEAD_D), wk).reshape(L, 2560)
    q = q.reshape(L, HEADS, HEAD_D); k = k.reshape(L, HEADS, HEAD_D); v = v.reshape(L, HEADS, HEAD_D)
    out = np.empty((L, HEADS, HEAD_D), dtype=np.float32)
    for h in range(HEADS):
        scores = (q[:, h, :] @ k[:, h, :].T) / np.sqrt(HEAD_D)
        scores = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(scores); p_ = e / e.sum(axis=-1, keepdims=True)
        out[:, h, :] = p_ @ v[:, h, :]
    attn = add_bias(out.reshape(L, 2560) @ Wout.T, Bout)
    save_raw(p + "_attn_x.bin", X)
    save_raw(p + "_attn_expected.bin", attn)

    # ---- mlp 参考 ----
    Xm = rng.randn(L, 2560).astype(np.float32)
    gate = add_bias(Xm @ Wing.T, Bing)
    up = add_bias(Xm @ Win.T, Bin)
    h = (gate / (1 + np.exp(-gate))) * up     # silu(gate)*up
    mlp_out = add_bias(h @ Wmout.T, Bmout)
    save_raw(p + "_mlp_x.bin", Xm)
    save_raw(p + "_mlp_expected.bin", mlp_out)

    print(f"[ok] block {block} ({'dual/vid' if block<10 else 'shared/all'}): "
          f"attn+mlp 导出 & 参考生成 (L={L})")


def main():
    export_block(0, "vid")    # dual
    export_block(20, "vid")   # shared (.all)
    print("\n[M2 导出完成] 文件在 models/m2/")


if __name__ == "__main__":
    main()

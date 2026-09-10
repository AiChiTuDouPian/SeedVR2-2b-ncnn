#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比 ncnn AdaCompose 输出的 ada_0_v_a_sc 与 numpy 参考 scA+scB。"""
import os, sys, numpy as np, struct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import ncnn_io as io

# numpy 参考：scA+scB = emb3[0,:,0,1] + attn_scale
def sinusoidal_embedding(t, dim):
    half = dim//2
    freqs = np.exp(-np.log(10000.0)*np.arange(half,dtype=np.float64)/half)
    args = np.array([t],dtype=np.float64)[:,None]*freqs[None,:]
    return np.concatenate([np.sin(args),np.cos(args)],axis=-1).astype(np.float32)

def time_embedding(t):
    with io.open_sd() as f:
        Wpi=io.get_top_weight(f,"emb_in.proj_in","weight");Bpi=io.get_top_weight(f,"emb_in.proj_in","bias")
        Wph=io.get_top_weight(f,"emb_in.proj_hid","weight");Bph=io.get_top_weight(f,"emb_in.proj_hid","bias")
        Wpo=io.get_top_weight(f,"emb_in.proj_out","weight");Bpo=io.get_top_weight(f,"emb_in.proj_out","bias")
    e=sinusoidal_embedding(t,256)
    e=e@Wpi.T+Bpi; e=e/(1+np.exp(-e))
    e=e@Wph.T+Bph; e=e/(1+np.exp(-e))
    e=e@Wpo.T+Bpo
    return e.reshape(-1)

DIM=2560; TIMESTEP=1000.0
emb = time_embedding(TIMESTEP)
emb3 = emb.reshape(1, DIM, 2, 3)
sA = emb3[0,:,0,0]; scA = emb3[0,:,0,1]; gA = emb3[0,:,0,2]

with io.open_sd() as f:
    scB = np.asarray(io.load_key(f, "blocks.0.ada.vid.attn_scale"), dtype=np.float32)
    shB = np.asarray(io.load_key(f, "blocks.0.ada.vid.attn_shift"), dtype=np.float32)
    gB  = np.asarray(io.load_key(f, "blocks.0.ada.vid.attn_gate"), dtype=np.float32)

ref_sc = (scA + scB).astype(np.float16).astype(np.float32)
ref_sh = (sA + shB).astype(np.float16).astype(np.float32)
ref_g  = (gA + gB).astype(np.float16).astype(np.float32)

# ncnn dump 的 ada_0_v_a_sc（提前定义以便分解检查）
nc = np.fromfile("block0_ada_0_v_a_sc.f32", np.float32)

print("=== numpy 参考 ada 向量（fp16 量化后） ===")
print(f"  attn_scale (scA+scB): range[{ref_sc.min():.4f},{ref_sc.max():.4f}] mean={ref_sc.mean():.4f}")
print(f"  attn_shift (sA+shB): range[{ref_sh.min():.4f},{ref_sh.max():.4f}] mean={ref_sh.mean():.4f}")
print(f"  attn_gate  (gA+gB): range[{ref_g.min():.4f},{ref_g.max():.4f}] mean={ref_g.mean():.4f}")
# 分解 scA(emb通道1) 与 scB(权重)
scA = emb3[0,:,0,1].astype(np.float16).astype(np.float32)
print(f"  [分解] scA=emb[d*6+1]: range[{scA.min():.4f},{scA.max():.4f}]")
print(f"  [分解] scB=attn_scale权重: range[{scB.min():.4f},{scB.max():.4f}] mean={scB.mean():.4f}")
# 检查 ncnn 结果是否接近某个单独分量
print(f"  [检查] ncnn ada_0_v_a_sc 是否≈ scA? cos={float(np.sum(nc*scA)/(np.linalg.norm(nc)*np.linalg.norm(scA))):.4f}")
print(f"  [检查] ncnn ada_0_v_a_sc 是否≈ scB? cos={float(np.sum(nc*scB)/(np.linalg.norm(nc)*np.linalg.norm(scB))):.4f}")

print("\n=== ncnn AdaCompose ada_0_v_a_sc ===")
print(f"  range[{nc.min():.4f},{nc.max():.4f}] mean={nc.mean():.4f} n={nc.size}")
if nc.size == ref_sc.size:
    cos = float(np.sum(nc*ref_sc)/(np.linalg.norm(nc)*np.linalg.norm(ref_sc)))
    print(f"  cos(attn_scale) = {cos:.6f}")
    print(f"  max|diff| = {np.abs(nc-ref_sc).max():.4f}")
    # 检查是否只是符号/顺序问题
    print("  ncnn[0:8] =", nc[:8])
    print("  numpy[0:8] =", ref_sc[:8])

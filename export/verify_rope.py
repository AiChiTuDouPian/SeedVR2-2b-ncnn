#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐元素对比官方 RoPE freq / 旋转结果与 mmrope（修正版）是否一致。"""
import sys, os
import numpy as np
import torch

from rotary_embedding_torch import RotaryEmbedding
from rotary_embedding_torch.rotary_embedding_torch import apply_rotary_emb as lib_apply

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmrope as mm

HEADS, HEAD_D = 20, 128
win_t, win_h, win_w, TXT_LEN = 1, 4, 4, 8
Lv = win_t * win_h * win_w

torch.manual_seed(0)
q_vid = torch.randn(HEADS, Lv, HEAD_D, dtype=torch.float32)


def official_freqs(l, wt, wh, ww):
    rope = RotaryEmbedding(dim=42, freqs_for="lang", theta=10000)
    vf = rope.get_axial_freqs(min(l + wt + 16, 1024), min(wh + 4, 128), min(ww + 4, 128))
    tf = rope.get_axial_freqs(min(l + 16, 1024))
    vid_freq = vf[l:l + wt, :wh, :ww].reshape(-1, vf.shape[-1])
    txt_freq = tf[:l].repeat(1, 3).reshape(-1, vf.shape[-1])  # 用 vid 的 last dim=126
    return vid_freq, txt_freq


def official_rotate(q, freq):
    out = torch.empty_like(q)
    for h in range(HEADS):
        out[h] = lib_apply(freq, q[h].float()).to(q.dtype)
    return out


vfreq, tfreq = official_freqs(TXT_LEN, win_t, win_h, win_w)
print(f"[official] vid_freq {tuple(vfreq.shape)} txt_freq {tuple(tfreq.shape)} (rot_dim={vfreq.shape[-1]})")
q_vid_off = official_rotate(q_vid, vfreq).numpy()


def mm_rotate(qh, freq):
    q = np.transpose(qh, (1, 0, 2))
    out = np.empty_like(q)
    for i in range(q.shape[0]):
        out[i] = mm.apply_rotary_emb(freq[i], q[i])
    return np.transpose(out, (1, 0, 2))


# ---- mmrope 修正版 ----
mm.ROPE_DIM = 42
mm.ROT_DIM = 2 * ((mm.ROPE_DIM + 1) // 2)
mm.FREQ_LAST = mm.N_AXES * mm.ROT_DIM

def rotate_half_fixed(x):
    x1 = x[..., 0::2]; x2 = x[..., 1::2]
    return np.concatenate([-x2, x1], axis=-1)

def apply_fixed(freqs, x):
    rot_dim = freqs.shape[-1]
    cos = np.cos(freqs).astype(np.float32); sin = np.sin(freqs).astype(np.float32)
    x_mid = x[..., :rot_dim]; x_left = x[..., rot_dim:]
    xr = rotate_half_fixed(x_mid)
    out = x_mid.astype(np.float32) * cos + xr.astype(np.float32) * sin
    return np.concatenate([out, x_left], axis=-1).astype(np.float32)

mm.apply_rotary_emb = apply_fixed

# 直接用 mmrope 当前 build_window_freqs（已 patch ROPE_DIM=42）
mvf, mtf = mm.build_window_freqs([(win_t, win_h, win_w)], TXT_LEN)
print(f"[mmrope FIXED] FREQ_LAST={mm.FREQ_LAST}")
print(f"  vid_freq max|diff| vs official = {np.abs(mvf - vfreq.numpy()).max():.3e}")
print(f"  txt_freq max|diff| vs official = {np.abs(mtf - tfreq.numpy()).max():.3e}")
qv_fix = mm_rotate(q_vid.numpy(), mvf)

# 手工旋转（直接用官方库 rotate_half + 官方 freq），与两者对比，隔离差异来源
from rotary_embedding_torch.rotary_embedding_torch import rotate_half as lib_rh
def manual_rotate(qh, freq):
    out = np.empty_like(qh)
    for h in range(HEADS):
        x = qh[h]
        cos = np.cos(freq).astype(np.float32); sin = np.sin(freq).astype(np.float32)
        xm = x[..., :126]; xl = x[..., 126:]
        xr = lib_rh(torch.from_numpy(xm)).numpy()
        out[h] = np.concatenate([xm.astype(np.float32)*cos + xr.astype(np.float32)*sin, xl], axis=-1)
    return out
qv_man = manual_rotate(q_vid.numpy(), vfreq.numpy())

print(f"  q_vid manual vs official max|diff| = {np.abs(qv_man - q_vid_off).max():.3e}")
print(f"  q_vid fixed  vs official max|diff| = {np.abs(qv_fix - q_vid_off).max():.3e}")
print(f"  q_vid fixed  vs manual  max|diff| = {np.abs(qv_fix - qv_man).max():.3e}")
print(f"  q_vid cos(fixed, official) = {float(np.dot(qv_fix.ravel(), q_vid_off.ravel())/(np.linalg.norm(qv_fix)*np.linalg.norm(q_vid_off)+1e-12)):.6f}")
print(f"  sample official[0,0,:6] = {q_vid_off[0,0,:6]}")
print(f"  sample fixed  [0,0,:6] = {qv_fix[0,0,:6]}")
print(f"  sample manual [0,0,:6] = {qv_man[0,0,:6]}")
print(f"  sample freq   [0,:6]   = {vfreq.numpy()[0,:6]}")

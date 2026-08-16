#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mmrope3d 的 freq 生成 + 应用（从 rotary_embedding_torch 源码忠实移植到 numpy）。
模型使用 NaMMRotaryEmbedding3d(dim=64) -> RotaryEmbedding(dim=64//3=21,
freqs_for="lang", theta=10000)。per-axis freq dim = 2*ceil(21/2)=22，三维 concat -> 66。
apply_rotary_emb 旋转 x 的最后维前 66 个通道，其余不变。
"""
import numpy as np


# 模型 config 中 rope_dim=128 -> NaMMRotaryEmbedding3d(dim=128) ->
# RotaryEmbedding(dim=128//3=42, freqs_for="lang", theta=10000)。
# get_axial_freqs 把 3 个轴 concat：每头实际旋转 42*3=126 通道（非 66！）。
# 注意 rotary_embedding_torch 的 freq 布局为 [f0,f0,f1,f1,...]，故 rotate_half 必须
# 按「相邻配对」(2i,2i+1) 交织，而不能用「半分」。
ROPE_DIM = 42          # = 模型 rope_dim(128) // 3
THETA = 10000.0
N_AXES = 3
ROT_DIM = 2 * ((ROPE_DIM + 1) // 2)   # 42
FREQ_LAST = N_AXES * ROT_DIM           # 126


def _base_freqs():
    # self.freqs = 1/(theta ** (arange(0, dim, 2)/dim))
    a = np.arange(0, ROPE_DIM, 2, dtype=np.float64)
    return 1.0 / (THETA ** (a / ROPE_DIM))


def _axis_freqs(dim_axis, base):
    # pos = arange(dim_axis) (lang); freqs = einsum('i,f->if', pos, base) -> (dim_axis, ROPE_DIM//2)
    # repeat r=2 -> (dim_axis, ROT_DIM)
    pos = np.arange(dim_axis, dtype=np.float64)
    f = np.einsum("i,f->if", pos, base)            # (dim_axis, base_n)
    f = np.repeat(f, 2, axis=-1)                   # (dim_axis, ROT_DIM)
    return f


def get_axial_freqs(*dims):
    base = _base_freqs()
    axis_list = []
    for ind, d in enumerate(dims):
        f = _axis_freqs(d, base)                   # (d, ROT_DIM)
        shape = [1] * len(dims) + [ROT_DIM]
        shape[ind] = d
        axis_list.append(f.reshape(shape))
    # broadcast over spatial dims then concat last dim
    b = np.broadcast_arrays(*axis_list)
    return np.concatenate(b, axis=-1)              # (*dims, FREQ_LAST)


def rotate_half(x):
    # rotary_embedding_torch 的 freq 布局为 [f0,f0,f1,f1,...]（每对相邻通道共享同一频率），
    # 因此必须按相邻配对 (2i,2i+1) 交织旋转：out[2i]=-x[2i+1], out[2i+1]=x[2i]。
    # 不能用「前半/后半」切分（那仅适用于 freq 为 [前半,后半] 布局的传统 RoPE）。
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return np.stack([-x_odd, x_even], axis=-1).reshape(np.shape(x))


def apply_rotary_emb(freqs, x):
    """
    freqs: (L, FREQ_LAST) 或 (*dims, FREQ_LAST)
    x:    (heads, L, 128)   —— 与模型 forward 的 rearrange 'L h d -> h L d' 一致
    返回 (heads, L, 128)
    """
    rot_dim = freqs.shape[-1]
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    x_mid = x[..., :rot_dim]
    x_left = x[..., rot_dim:]
    xr = rotate_half(x_mid)
    out_mid = x_mid.astype(np.float32) * cos + xr.astype(np.float32) * sin
    return np.concatenate([out_mid, x_left], axis=-1).astype(np.float32)


def build_window_freqs(win_shapes, txt_len):
    """忠实复现 NaMMRotaryEmbedding3d.get_freqs 的窗口分支。

    win_shapes: list of (wt_i, wh_i, ww_i) 每一窗口的实际尺寸（变长！）
    返回 vid_freq (sum(f_i), 66) 与 txt_freq (Nwin*txt_len, 66)。

    关键点（来自 rope.py get_freqs）：
    - max_temporal = max_i (txt_len + f_i), max_height = max_i wh_i, max_width = max_i ww_i
    - vid_freqs = get_axial_freqs(min(max_t+16,1024), min(max_h+4,128), min(max_w+4,128))
    - 每窗口 vid_freq = vid_freqs[txt_len : txt_len+f_i, :wh_i, :ww_i].reshape(-1,66)
      （视频时序 freq 从偏移 txt_len 处开始，文本占据前 txt_len 个时序槽）
    - txt_freq = get_axial_freqs(min(txt_len+16,1024))[:txt_len].repeat(1,3).reshape(-1,66)
      逐窗口重复（文本 freq 与窗口无关，故每窗口相同）
    """
    l = txt_len
    # 注意：模型 get_freqs 里 vid_shape = (wt_i, wh_i, ww_i)，第一个分量 f 即窗口时序尺寸 wt
    # （不是 token 数 wt*wh*ww）。capping: max_temporal = max_i(l + wt_i)
    max_t = max(l + wt for (wt, wh, ww) in win_shapes)
    max_h = max(wh for (_, wh, _) in win_shapes)
    max_w = max(ww for (_, _, ww) in win_shapes)
    capped_t = min(max_t + 16, 1024)
    capped_h = min(max_h + 4, 128)
    capped_w = min(max_w + 4, 128)
    vid_full = get_axial_freqs(capped_t, capped_h, capped_w)     # (Tc,Hc,Wc,66)
    vid_list = []
    for (wt, wh, ww) in win_shapes:
        # 视频时序 freq 从偏移 l 处开始；每窗口 freq 形状 (wt,wh,ww,66) -> (wt*wh*ww,66)
        vid_list.append(vid_full[l:l + wt, :wh, :ww].reshape(-1, FREQ_LAST))
    vid_freq = np.concatenate(vid_list, axis=0).astype(np.float32)  # (sum wt*wh*ww, 66)

    capped_l = min(l + 16, 1024)
    txt_full = get_axial_freqs(capped_l)                          # (lc, ROT_DIM=42)
    # 官方 rope.py: txt_freqs[:l].repeat(1, 3) 是 torch repeat = 整块平铺 3 次，
    # 等价 np.tile((1,3))，而非逐元素交织的 np.repeat(..., axis=-1)。
    txt_freq = np.tile(txt_full[0:l], (1, 3))                     # (l, 126)
    txt_freq = np.tile(txt_freq, (len(win_shapes), 1)).astype(np.float32)  # (Nwin*l,66)
    return vid_freq, txt_freq

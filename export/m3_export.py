#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M3 导出 + 参考：单个 NaSwinAttention（非 shifted，720pwin_by_size_bysize）的
window partition -> RoPE(mmrope3d) -> varlen concat(文本逐窗口重复) -> 窗口内 SDPA
-> window reverse -> proj_out。

用真实 checkpoint 的 block0(dual) 的 vid/txt 分支权重，随机输入，生成参考输出；
同时导出 C++ 需要的权重/输入/freqs/窗口定义。
"""
import os
import sys
import math
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncnn_io as io
import awa_window
import mmrope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M3 = os.path.join(ROOT, "models", "m3")
os.makedirs(M3, exist_ok=True)

HEADS = 20
HEAD_D = 128
DIM = 2560
QKV = HEADS * HEAD_D * 3  # 7680
EPS = 1e-6
BLOCK = 0          # dual 层
NUM_WINDOWS = (2, 3, 3)
WIN_METHOD = "720pwin_by_size_bysize"
TXT_LEN = 5


def save_raw(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<q", arr.ndim))
        for d in arr.shape:
            fp.write(struct.pack("<q", int(d)))
        fp.write(arr.tobytes())


def rmsnorm(x, w):
    # x: (...,128), w:(128,)  ->  fusedrms: x/sqrt(mean(x^2)+eps)*w
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + EPS) * w


def window_partition(vid, windows, t, h, w):
    """vid: (t,h,w, hh, d) -> (Nwin*f, hh, d)，窗口顺序 iw,ih,it。"""
    parts = []
    for (st, sh, sw) in windows:
        sl = vid[st, sh, sw]              # (wt',wh',ww', hh, d)
        parts.append(sl.reshape(-1, sl.shape[-2], sl.shape[-1]))  # (f, hh, d)
    return np.concatenate(parts, axis=0)  # (Nwin*f, hh, d)


def window_reverse(part, windows, t, h, w):
    hh, d = part.shape[-2], part.shape[-1]
    out = np.zeros((t, h, w, hh, d), dtype=np.float32)
    i = 0
    for (st, sh, sw) in windows:
        wt = st.stop - st.start; wh = sh.stop - sh.start; ww = sw.stop - sw.start
        f = wt * wh * ww
        out[st, sh, sw] = part[i:i + f].reshape(wt, wh, ww, hh, d)
        i += f
    return out.reshape(t * h * w, hh, d)


def main():
    with io.open_sd() as f:
        Wqkv_v = io.block_key(BLOCK, "proj_qkv", "vid", "weight")
        Wqkv_t = io.block_key(BLOCK, "proj_qkv", "txt", "weight")
        Wout_v = io.block_key(BLOCK, "proj_out", "vid", "weight")
        Wout_t = io.block_key(BLOCK, "proj_out", "txt", "weight")
        nq_v = io.load_key(f, io.block_key(BLOCK, "norm_q", "vid", "weight"))
        nk_v = io.load_key(f, io.block_key(BLOCK, "norm_k", "vid", "weight"))
        nq_t = io.load_key(f, io.block_key(BLOCK, "norm_q", "txt", "weight"))
        nk_t = io.load_key(f, io.block_key(BLOCK, "norm_k", "txt", "weight"))
        Wqkv_v = io.load_key(f, Wqkv_v); Wqkv_t = io.load_key(f, Wqkv_t)
        Wout_v = io.load_key(f, Wout_v); Wout_t = io.load_key(f, Wout_t)

    # ---- 导出权重 (ncnn fp16 Linear) ----
    io.export_linear(os.path.join(M3, "b0_qkv_vid"), Wqkv_v, None, fp16=True)
    io.export_linear(os.path.join(M3, "b0_qkv_txt"), Wqkv_t, None, fp16=True)
    io.export_linear(os.path.join(M3, "b0_out_vid"), Wout_v, None, fp16=True)
    io.export_linear(os.path.join(M3, "b0_out_txt"), Wout_t, None, fp16=True)
    save_raw(os.path.join(M3, "normq_vid.bin"), nq_v)
    save_raw(os.path.join(M3, "normk_vid.bin"), nk_v)
    save_raw(os.path.join(M3, "normq_txt.bin"), nq_t)
    save_raw(os.path.join(M3, "normk_txt.bin"), nk_t)

    # ---- 随机输入（固定种子，可复现）----
    rng = np.random.RandomState(20240713)
    t, h, w = 4, 40, 40
    Lv = t * h * w
    Xv = rng.randn(Lv, DIM).astype(np.float32)
    Xt = rng.randn(TXT_LEN, DIM).astype(np.float32)
    save_raw(os.path.join(M3, "x_vid.bin"), Xv)
    save_raw(os.path.join(M3, "x_txt.bin"), Xt)

    # ---- 窗口 ----
    windows = awa_window.make_720Pwindows_bysize((t, h, w), NUM_WINDOWS)
    # 直接由公式复算 nt,nh,nw,wt,wh,ww 供 C++ 校验
    resized_nt, resized_nh, resized_nw = NUM_WINDOWS
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh = math.ceil(resized_h / resized_nh); ww = math.ceil(resized_w / resized_nw)
    wt = math.ceil(min(t, 30) / resized_nt)
    nnt, nnh, nnw = math.ceil(t / wt), math.ceil(h / wh), math.ceil(w / ww)
    # 导出窗口定义
    win_def = np.array([t, h, w, wt, wh, ww, nnt, nnh, nnw, len(windows)], dtype=np.int64)
    slices_flat = []
    for (st, sh, sw) in windows:
        slices_flat.append([st.start, st.stop, sh.start, sh.stop, sw.start, sw.stop])
    slices_flat = np.array(slices_flat, dtype=np.int64).reshape(-1)
    # 注意：C++ load_raw_i64 读的是 "count + count个值"，count 必须是 总数量
    # （header 10 + slices len(windows)*6），否则 slices 读不到
    total = win_def.size + slices_flat.size
    with open(os.path.join(M3, "windows.bin"), "wb") as fp:
        fp.write(struct.pack("<q", total))
        fp.write(win_def.tobytes())
        fp.write(slices_flat.tobytes())

    # ---- proj_qkv ----
    vid_qkv = Xv @ Wqkv_v.T            # (Lv, 7680)
    txt_qkv = Xt @ Wqkv_t.T            # (l, 7680)
    vid_qkv = vid_qkv.reshape(Lv, 3, HEADS, HEAD_D)
    txt_qkv = txt_qkv.reshape(TXT_LEN, 3, HEADS, HEAD_D)
    vq, vk, vv = vid_qkv[:, 0], vid_qkv[:, 1], vid_qkv[:, 2]
    tq, tk, tv = txt_qkv[:, 0], txt_qkv[:, 1], txt_qkv[:, 2]

    # ---- qk_norm ----
    vq = rmsnorm(vq, nq_v); vk = rmsnorm(vk, nk_v)
    tq = rmsnorm(tq, nq_t); tk = rmsnorm(tk, nk_t)

    # ---- window partition (video) ----
    vqw = window_partition(vq.reshape(t, h, w, HEADS, HEAD_D), windows, t, h, w)  # (Nwin*f, hh, d)
    vkw = window_partition(vk.reshape(t, h, w, HEADS, HEAD_D), windows, t, h, w)
    vvw = window_partition(vv.reshape(t, h, w, HEADS, HEAD_D), windows, t, h, w)

    # ---- RoPE（窗口分支）----
    per_win_f = wt * wh * ww
    win_shapes = [(wt, wh, ww)] * len(windows)
    vid_freq_all, txt_freq_all = mmrope.build_window_freqs(win_shapes, TXT_LEN)
    # 注意 build_window_freqs 用 capped 维度计算，需与模型一致：这里复算使用真实公式
    # apply: rearrange 'L h d -> h L d'
    vq_w = np.transpose(vqw, (1, 0, 2))   # (hh, Nwin*f, d)
    save_raw(os.path.join(M3, "vqw_norope.bin"), vqw.reshape(-1, DIM).astype(np.float32))  # (Nwin*f,2560)
    save_raw(os.path.join(M3, "tq_norope.bin"), tq.astype(np.float32).reshape(-1, DIM))     # (l,2560)
    vk_w = np.transpose(vkw, (1, 0, 2))
    vv_w = np.transpose(vvw, (1, 0, 2))
    vq_w = mmrope.apply_rotary_emb(vid_freq_all, vq_w)
    vk_w = mmrope.apply_rotary_emb(vid_freq_all, vk_w)

    # 文本逐窗口重复
    tq_rep = np.tile(tq, (len(windows), 1, 1))   # (Nwin*l, hh, d)
    tk_rep = np.tile(tk, (len(windows), 1, 1))
    tv_rep = np.tile(tv, (len(windows), 1, 1))
    tq_w = np.transpose(tq_rep, (1, 0, 2))
    tk_w = np.transpose(tk_rep, (1, 0, 2))
    tq_w = mmrope.apply_rotary_emb(txt_freq_all, tq_w)
    tk_w = mmrope.apply_rotary_emb(txt_freq_all, tk_w)
    tv_w = np.transpose(tv_rep, (1, 0, 2))   # 文本 value（不做 RoPE）

    # ---- varlen concat + 窗口内 SDPA ----
    cu = [0]
    q_seg, k_seg, v_seg = [], [], []
    # 按窗口顺序拼接 [vid_win, txt]
    off = 0
    for wi in range(len(windows)):
        f0 = wi * per_win_f; f1 = f0 + per_win_f
        t0 = wi * TXT_LEN; t1 = t0 + TXT_LEN
        q_seg.append(np.concatenate([vq_w[:, f0:f1, :], tq_w[:, t0:t1, :]], axis=1))  # (hh, f+l, d)
        k_seg.append(np.concatenate([vk_w[:, f0:f1, :], tk_w[:, t0:t1, :]], axis=1))
        v_seg.append(np.concatenate([vv_w[:, f0:f1, :], tv_w[:, t0:t1, :]], axis=1))
        cu.append(cu[-1] + per_win_f + TXT_LEN)
    cu = np.array(cu, dtype=np.int64)

    out_seg = []
    for wi in range(len(windows)):
        q = q_seg[wi]; k = k_seg[wi]; v = v_seg[wi]   # (hh, S, d), S=f+l
        S = q.shape[1]
        o = np.empty((HEADS, S, HEAD_D), dtype=np.float32)
        scale = 1.0 / math.sqrt(HEAD_D)
        for hh in range(HEADS):
            qq = q[hh]; kk = k[hh]; vv = v[hh]          # (S,d)
            scores = (qq @ kk.T) * scale                 # (S,S)
            scores = scores - scores.max(axis=-1, keepdims=True)
            e = np.exp(scores); p = e / e.sum(axis=-1, keepdims=True)
            o[hh] = p @ vv
        out_seg.append(o)
    out_all = np.concatenate(out_seg, axis=1)           # (hh, Nwin*(f+l), d)

    # ---- unconcat：拆回 vid / txt 每段 ----
    vout_parts = []; tout_parts = []
    for wi in range(len(windows)):
        seg = out_all[:, wi * (per_win_f + TXT_LEN): (wi + 1) * (per_win_f + TXT_LEN), :]
        vout_parts.append(seg[:, :per_win_f, :])         # (hh, f, d)
        tout_parts.append(seg[:, per_win_f:, :])         # (hh, l, d)
    vout_win = np.concatenate(vout_parts, axis=1)        # (hh, Nwin*f, d)
    tout_win = np.concatenate(tout_parts, axis=1)        # (hh, Nwin*l, d)
    # 转回 (L, hh, d)
    vout_win = np.transpose(vout_win, (1, 0, 2))   # (Nwin*f, hh, d)
    tout_win = np.transpose(tout_win, (1, 0, 2))   # (Nwin*l, hh, d)
    # window reverse video
    vout_rev = window_reverse(vout_win, windows, t, h, w)   # (Lv, hh, d)
    vout_rev = vout_rev.reshape(Lv, DIM)
    # txt: 逐窗口结果相同（同一文本重复），取第一段即可
    tout = tout_win[:TXT_LEN].reshape(TXT_LEN, DIM)

    # ---- proj_out ----
    vid_out = vout_rev @ Wout_v.T
    txt_out = tout @ Wout_t.T
    save_raw(os.path.join(M3, "ref_vid_out.bin"), vid_out)
    save_raw(os.path.join(M3, "ref_txt_out.bin"), txt_out)
    save_raw(os.path.join(M3, "vid_freq.bin"), vid_freq_all)
    save_raw(os.path.join(M3, "txt_freq.bin"), txt_freq_all)
    save_raw(os.path.join(M3, "cu_seqlens.bin"), cu.astype(np.int64))

    print(f"[ok] M3 导出完成: t={t} h={h} w={w} windows={len(windows)} "
          f"(wt,wh,ww=({wt},{wh},{ww}) n=({nnt},{nnh},{nnw})) txt_len={TXT_LEN}")
    print(f"     vid_out cos-self-check skip; ref saved. Lv={Lv}")


if __name__ == "__main__":
    main()

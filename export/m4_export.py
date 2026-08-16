#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M4 导出 + 参考：shifted NaSwinAttention（720pswin_by_size_bysize）的 AWA。

与 M3 的区别：
1. 窗口用 make_shifted_720Pwindows_bysize -> 27 个窗口，且窗口尺寸可变（边界窗口更小）。
2. 视频 reverse 仍是精确逆（shifted 窗口非重叠、完美平铺，M=L）。
3. 文本输出在每个窗口重复，attention 后需跨窗口「平均」(coalesce) 得到单个文本输出。
4. RoPE 为窗口局部坐标；freq 用真实公式（时序偏移 l，capping 取 batch 内各窗口最大尺寸）。

复用 block0(dual) 权重、随机输入（与 M3 同种子，便于对照）。
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
M4 = os.path.join(ROOT, "models", "m4")
os.makedirs(M4, exist_ok=True)

HEADS = 20
HEAD_D = 128
DIM = 2560
QKV = HEADS * HEAD_D * 3  # 7680
EPS = 1e-6
BLOCK = 0
NUM_WINDOWS = (2, 3, 3)
WIN_METHOD = "720pswin_by_size_bysize"
TXT_LEN = 5


def save_raw(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<q", arr.ndim))
        for d in arr.shape:
            fp.write(struct.pack("<q", int(d)))
        fp.write(arr.tobytes())


def rmsnorm(x, w):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + EPS) * w


def main():
    with io.open_sd() as f:
        Wqkv_v = io.load_key(f, io.block_key(BLOCK, "proj_qkv", "vid", "weight"))
        Wqkv_t = io.load_key(f, io.block_key(BLOCK, "proj_qkv", "txt", "weight"))
        Wout_v = io.load_key(f, io.block_key(BLOCK, "proj_out", "vid", "weight"))
        Wout_t = io.load_key(f, io.block_key(BLOCK, "proj_out", "txt", "weight"))
        nq_v = io.load_key(f, io.block_key(BLOCK, "norm_q", "vid", "weight"))
        nk_v = io.load_key(f, io.block_key(BLOCK, "norm_k", "vid", "weight"))
        nq_t = io.load_key(f, io.block_key(BLOCK, "norm_q", "txt", "weight"))
        nk_t = io.load_key(f, io.block_key(BLOCK, "norm_k", "txt", "weight"))

    io.export_linear(os.path.join(M4, "b0_qkv_vid"), Wqkv_v, None, fp16=True)
    io.export_linear(os.path.join(M4, "b0_qkv_txt"), Wqkv_t, None, fp16=True)
    io.export_linear(os.path.join(M4, "b0_out_vid"), Wout_v, None, fp16=True)
    io.export_linear(os.path.join(M4, "b0_out_txt"), Wout_t, None, fp16=True)
    save_raw(os.path.join(M4, "normq_vid.bin"), nq_v)
    save_raw(os.path.join(M4, "normk_vid.bin"), nk_v)
    save_raw(os.path.join(M4, "normq_txt.bin"), nq_t)
    save_raw(os.path.join(M4, "normk_txt.bin"), nk_t)

    rng = np.random.RandomState(20240713)
    t, h, w = 4, 40, 40
    Lv = t * h * w
    Xv = rng.randn(Lv, DIM).astype(np.float32)
    Xt = rng.randn(TXT_LEN, DIM).astype(np.float32)
    save_raw(os.path.join(M4, "x_vid.bin"), Xv)
    save_raw(os.path.join(M4, "x_txt.bin"), Xt)

    # ---- shifted 窗口 ----
    windows = awa_window.make_shifted_720Pwindows_bysize((t, h, w), NUM_WINDOWS)
    nwin = len(windows)
    # 每窗口实际尺寸 (wt_i, wh_i, ww_i) 与 f_i
    win_shapes = []
    f_list = []
    for (st, sh, sw) in windows:
        wt = st.stop - st.start; wh = sh.stop - sh.start; ww = sw.stop - sw.start
        win_shapes.append((wt, wh, ww))
        f_list.append(wt * wh * ww)
    M = sum(f_list)
    print(f"[info] shifted windows={nwin} M(总视频token)={M} Lv={Lv} (应相等)")

    # 导出窗口定义（与 M3 同格式：count + header(10) + nwin*6 slices）
    resized_nt, resized_nh, resized_nw = NUM_WINDOWS
    scale = math.sqrt((45 * 80) / (h * w))
    resized_h, resized_w = round(h * scale), round(w * scale)
    wh0 = math.ceil(resized_h / resized_nh); ww0 = math.ceil(resized_w / resized_nw)
    wt0 = math.ceil(min(t, 30) / resized_nt)
    win_def = np.array([t, h, w, wt0, wh0, ww0, nwin, 0, 0, nwin], dtype=np.int64)
    slices_flat = []
    for (st, sh, sw) in windows:
        slices_flat.append([st.start, st.stop, sh.start, sh.stop, sw.start, sw.stop])
    slices_flat = np.array(slices_flat, dtype=np.int64).reshape(-1)
    total = win_def.size + slices_flat.size
    with open(os.path.join(M4, "windows.bin"), "wb") as fp:
        fp.write(struct.pack("<q", total))
        fp.write(win_def.tobytes())
        fp.write(slices_flat.tobytes())

    # ---- proj_qkv + qk_norm ----
    vid_qkv = (Xv @ Wqkv_v.T).reshape(Lv, 3, HEADS, HEAD_D)
    txt_qkv = (Xt @ Wqkv_t.T).reshape(TXT_LEN, 3, HEADS, HEAD_D)
    vq, vk, vv = vid_qkv[:, 0], vid_qkv[:, 1], vid_qkv[:, 2]
    tq, tk, tv = txt_qkv[:, 0], txt_qkv[:, 1], txt_qkv[:, 2]
    vq = rmsnorm(vq, nq_v); vk = rmsnorm(vk, nk_v)
    tq = rmsnorm(tq, nq_t); tk = rmsnorm(tk, nk_t)

    # ---- window partition（视频，变长窗口）----
    vq_win = vq.reshape(t, h, w, HEADS, HEAD_D)
    vk_win = vk.reshape(t, h, w, HEADS, HEAD_D)
    vv_win = vv.reshape(t, h, w, HEADS, HEAD_D)
    vqw_list, vkw_list, vvw_list = [], [], []
    vout_target = np.zeros((t, h, w, HEADS, HEAD_D), dtype=np.float32)  # 用于 reverse 写回
    for wi, (st, sh, sw) in enumerate(windows):
        vqw_list.append(vq_win[st, sh, sw].reshape(-1, HEADS, HEAD_D))
        vkw_list.append(vk_win[st, sh, sw].reshape(-1, HEADS, HEAD_D))
        vvw_list.append(vv_win[st, sh, sw].reshape(-1, HEADS, HEAD_D))
    vqw = np.concatenate(vqw_list, axis=0)   # (M, hh, d)
    vkw = np.concatenate(vkw_list, axis=0)
    vvw = np.concatenate(vvw_list, axis=0)
    save_raw(os.path.join(M4, "vqw_norope.bin"), vqw.reshape(-1, DIM).astype(np.float32))

    # ---- RoPE（窗口局部坐标，真实公式）----
    vid_freq_all, txt_freq_all = mmrope.build_window_freqs(win_shapes, TXT_LEN)
    # apply: rearrange 'L h d -> h L d'
    vq_w = np.transpose(vqw, (1, 0, 2))   # (hh, M, d)
    vk_w = np.transpose(vkw, (1, 0, 2))
    vv_w = np.transpose(vvw, (1, 0, 2))
    vq_w = mmrope.apply_rotary_emb(vid_freq_all, vq_w)
    vk_w = mmrope.apply_rotary_emb(vid_freq_all, vk_w)

    # 文本逐窗口重复（RoPE 与窗口无关，逐窗口相同）
    tq_rep = np.tile(tq, (nwin, 1, 1))     # (Nwin*l, hh, d)
    tk_rep = np.tile(tk, (nwin, 1, 1))
    tv_rep = np.tile(tv, (nwin, 1, 1))
    tq_w = np.transpose(tq_rep, (1, 0, 2))
    tk_w = np.transpose(tk_rep, (1, 0, 2))
    tq_w = mmrope.apply_rotary_emb(txt_freq_all, tq_w)
    tk_w = mmrope.apply_rotary_emb(txt_freq_all, tk_w)
    tv_w = np.transpose(tv_rep, (1, 0, 2))

    # ---- varlen concat + 每窗口 SDPA（变长）----
    cu = [0]
    out_v_seg = []   # 每窗口 vid 输出 (f_i, hh, d)
    out_t_seg = []   # 每窗口 txt 输出 (l, hh, d)
    vg = 0   # 视频 token 累计（vq_w/vk_w/vv_w 中无文本间隔）
    for wi in range(nwin):
        f_i = f_list[wi]
        q = np.concatenate([vq_w[:, vg:vg + f_i, :],
                            tq_w[:, wi * TXT_LEN:(wi + 1) * TXT_LEN, :]], axis=1)  # (hh, f+l, d)
        k = np.concatenate([vk_w[:, vg:vg + f_i, :],
                            tk_w[:, wi * TXT_LEN:(wi + 1) * TXT_LEN, :]], axis=1)
        v = np.concatenate([vv_w[:, vg:vg + f_i, :],
                            tv_w[:, wi * TXT_LEN:(wi + 1) * TXT_LEN, :]], axis=1)
        vg += f_i
        cu.append(cu[-1] + f_i + TXT_LEN)
        S = q.shape[1]
        o = np.empty((HEADS, S, HEAD_D), dtype=np.float32)
        scale = 1.0 / math.sqrt(HEAD_D)
        for hh in range(HEADS):
            qq = q[hh]; kk = k[hh]; vv = v[hh]
            scores = (qq @ kk.T) * scale
            scores = scores - scores.max(axis=-1, keepdims=True)
            e = np.exp(scores); p = e / e.sum(axis=-1, keepdims=True)
            o[hh] = p @ vv
        out_v_seg.append(o[:, :f_i, :])          # (hh, f_i, d)
        out_t_seg.append(o[:, f_i:, :])          # (hh, l, d)
    cu = np.array(cu, dtype=np.int64)
    save_raw(os.path.join(M4, "cu_seqlens.bin"), cu)
    save_raw(os.path.join(M4, "vid_freq.bin"), vid_freq_all)
    save_raw(os.path.join(M4, "txt_freq.bin"), txt_freq_all)

    # ---- 视频 reverse（写回，非重叠精确逆）----
    g = 0
    for wi, (st, sh, sw) in enumerate(windows):
        wt = st.stop - st.start; wh = sh.stop - sh.start; ww = sw.stop - sw.start
        blk = out_v_seg[wi]                       # (hh, f_i, d)
        blk = np.transpose(blk, (1, 0, 2))        # (f_i, hh, d)
        tgt = (st.stop - st.start) * (sh.stop - sh.start) * (sw.stop - sw.start) * HEADS * HEAD_D
        if blk.size != tgt or wi == 25:
            print(f"[DBG] wi={wi} blk.shape={blk.shape} f_i={f_list[wi]} tgt={tgt} "
                  f"(wt,wh,ww)=({wt},{wh},{ww})")
        if blk.size != tgt:
            print(f"[DBG] wi={wi} f_i={f_list[wi]} blk.size={blk.size} tgt={tgt} "
                  f"(wt,wh,ww)=({wt},{wh},{ww}) cu[{wi}:{wi+2}]={cu[wi]},{cu[wi+1]}")
        vout_target[st, sh, sw] = blk.reshape(wt, wh, ww, HEADS, HEAD_D)
        g += f_list[wi]
    vout_rev = vout_target.reshape(Lv, DIM)       # (Lv, hh*d)

    # ---- 文本 coalesce 平均（跨窗口）----
    tout_stacked = np.stack(out_t_seg, axis=0)    # (Nwin, hh, l, d)
    tout_avg = tout_stacked.mean(axis=0)          # (hh, l, d)
    tout = np.transpose(tout_avg, (1, 0, 2)).reshape(TXT_LEN, DIM)  # (l, hh*d)

    # ---- proj_out ----
    vid_out = vout_rev @ Wout_v.T
    txt_out = tout @ Wout_t.T
    save_raw(os.path.join(M4, "ref_vid_out.bin"), vid_out)
    save_raw(os.path.join(M4, "ref_txt_out.bin"), txt_out)

    print(f"[ok] M4 导出完成: shifted windows={nwin} M={M} txt_len={TXT_LEN}")
    print(f"     cu_seqlens={cu.tolist()}")


if __name__ == "__main__":
    main()

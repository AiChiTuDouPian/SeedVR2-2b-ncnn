#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比 C++ build_freqs 与 mmrope(PyTorch参考) 生成的 RoPE 频率。用 480p 真实窗口。"""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import awa_window, mmrope

TXT = 58
# 480p token 网格 (1, 30, 45)
GRID = (1, 30, 45)
WINDOW = (4, 3, 3)

# ---- PyTorch/mmrope 参考 ----
wns = awa_window.make_720Pwindows_bysize(GRID, WINDOW)   # 非移位
wsh = awa_window.make_shifted_720Pwindows_bysize(GRID, WINDOW)
for tag, windows in (("ns", wns), ("sh", wsh)):
    win_shapes = []
    for (st, sh, sw) in windows:
        win_shapes.append((st.stop-st.start, sh.stop-sh.start, sw.stop-sw.start))
    vid_freq_pt, txt_freq_pt = mmrope.build_window_freqs(win_shapes, TXT)
    # ---- 复现 C++ build_freqs ----
    def build_freqs(wt, wh, ww, l):
        nwin = len(wt)
        ROPE_DIM=42; ROT_DIM=42; FREQ_LAST=126; THETA=10000.0
        base = np.array([1.0/(THETA**((j*2)/ROPE_DIM)) for j in range(21)], dtype=np.float64)
        max_t=max(l+wt[i] for i in range(nwin)); max_h=max(wh); max_w=max(ww)
        total = sum(wt[i]*wh[i]*ww[i] for i in range(nwin))
        vf = np.zeros(total*FREQ_LAST, np.float32)
        off=0
        for i in range(nwin):
            for it in range(wt[i]):
                tt = l+it
                for ih in range(wh[i]):
                    for iw in range(ww[i]):
                        for k in range(FREQ_LAST):
                            if k<ROT_DIM: v=tt*base[k//2]
                            elif k<2*ROT_DIM: v=ih*base[(k-ROT_DIM)//2]
                            else: v=iw*base[(k-2*ROT_DIM)//2]
                            vf[off]=v; off+=1
        tf = np.zeros(nwin*l*FREQ_LAST, np.float32)
        off=0
        for i in range(nwin):
            for it in range(l):
                for k in range(FREQ_LAST):
                    kk=k%ROT_DIM
                    tf[off]=it*base[kk//2]; off+=1
        return vf, tf
    wt=[sh.stop-sh.start for (_,sh,_) in windows]
    wh=[sh.stop-sh.start for (st,sh,_) in windows]
    ww=[sw.stop-sw.start for (_,_,sw) in windows]
    vf_cpp, tf_cpp = build_freqs(wt, wh, ww, TXT)
    print(f"[{tag}] 窗口 nwin={len(windows)} 尺寸={win_shapes[:3]}")
    print(f"  vid_freq: PT={vid_freq_pt.shape} n={vid_freq_pt.size}  C++={vf_cpp.shape} n={vf_cpp.size}")
    print(f"  txt_freq: PT={txt_freq_pt.shape} n={txt_freq_pt.size}  C++={tf_cpp.shape} n={tf_cpp.size}")
    if vid_freq_pt.size == vf_cpp.size:
        a=vid_freq_pt.ravel().astype(np.float64); b=vf_cpp.astype(np.float64)
        cos = float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
        print(f"  vid_freq cos(C++ vs PT) = {cos:.8f}  maxdiff={np.abs(a-b).max():.6f}")
        if cos < 0.999:
            # 找第一个不同位置
            d = np.abs(a-b)
            idx = np.argmax(d)
            print(f"  最大差异位置 {idx}: C++={b[idx]:.6f} PT={a[idx]:.6f}")
            print(f"  窗口 token 数 vs 全局网格: max_h={max(wh)} max_w={max(ww)} max_t={max(wt)}")

#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import awa_window, mmrope
TXT=58; GRID=(1,30,45); WINDOW=(4,3,3)

def build_freqs_cpp(win_tuples, l):
    # win_tuples: list of (st_t,en_t,sh_t,eh_t,sw_t,ew_t) 即完整 slice
    nwin=len(win_tuples)
    ROPE_DIM=42; ROT_DIM=42; FREQ_LAST=126; THETA=10000.0
    base=np.array([1.0/(THETA**((j*2)/ROPE_DIM)) for j in range(21)],dtype=np.float64)
    wt=[en-st for (st,en,_,_,_,_) in win_tuples]
    wh=[eh-sh for (_,_,sh,eh,_,_) in win_tuples]
    ww=[ew-sw for (_,_,_,_,sw,ew) in win_tuples]
    total=sum(wt[i]*wh[i]*ww[i] for i in range(nwin))
    vf=np.zeros(total*FREQ_LAST,np.float32); off=0
    for i in range(nwin):
        for it in range(wt[i]):
            tt=l+it
            for ih in range(wh[i]):
                for iw in range(ww[i]):
                    for k in range(FREQ_LAST):
                        if k<ROT_DIM: v=tt*base[k//2]
                        elif k<2*ROT_DIM: v=ih*base[(k-ROT_DIM)//2]
                        else: v=iw*base[(k-2*ROT_DIM)//2]
                        vf[off]=v; off+=1
    return vf, total, wt, wh, ww

for tag, windows in (("ns", awa_window.make_720Pwindows_bysize(GRID,WINDOW)),
                     ("sh", awa_window.make_shifted_720Pwindows_bysize(GRID,WINDOW))):
    win_shapes=[]; win_tuples=[]
    for (st,sh,sw) in windows:
        win_shapes.append((st.stop-st.start,sh.stop-sh.start,sw.stop-sw.start))
        win_tuples.append((st.start,st.stop,sh.start,sh.stop,sw.start,sw.stop))
    vf_pt, tf_pt = mmrope.build_window_freqs(win_shapes, TXT)
    vf_cpp, total_cpp, wt, wh, ww = build_freqs_cpp(win_tuples, TXT)
    print(f"[{tag}] nwin={len(windows)} C++ total={total_cpp} PT={vf_pt.shape[0]}  wt={wt} wh={wh} ww={ww}")
    if vf_pt.size==vf_cpp.size:
        a=vf_pt.ravel().astype(np.float64); b=vf_cpp.astype(np.float64)
        cos=float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
        print(f"  vid_freq cos(C++ vs PT)={cos:.10f} maxdiff={np.abs(a-b).max():.8f}")
        if cos<0.999:
            idx=np.argmax(np.abs(a-b))
            print(f"  最大差异 idx={idx}: C++={b[idx]:.6f} PT={a[idx]:.6f}  rel={np.abs(a[idx]-b[idx])/max(abs(a[idx]),1e-9):.4f}")
    else:
        print(f"  尺寸不同: C++={vf_cpp.size} PT={vf_pt.size}")

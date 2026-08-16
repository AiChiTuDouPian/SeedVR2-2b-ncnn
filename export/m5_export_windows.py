#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M5 辅助：导出两套窗口几何 + mmrope3d freq 供 C++ verify_dit 加载。
窗口几何只取决于 (grid, window_method)，与 block 无关 -> 每个 method 只导出一份。
文件格式（与 verify_awa.cpp / m4 的 windows.bin 兼容）：
  header: 10 x int64 [t,h,w,wt,wh,ww,nnt,nnh,nnw,nwin]
  slices: nwin x 6 x int64 [st,en,sh,eh,sw,ew]
  vid_freq: (sum(f_i), 66) float32   (f_i = wt*wh*ww)
  txt_freq: (nwin*TXT, 66) float32
"""
import os, struct
import numpy as np
sys = __import__("sys")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import awa_window as aw
import mmrope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M5 = os.path.join(ROOT, "models", "m5")
os.makedirs(M5, exist_ok=True)

GRID = (2, 40, 40)
WINDOW = (4, 3, 3)
TXT_LEN = 8


def dump(path, method):
    windows = aw.window_op(method)(GRID, WINDOW)
    nwin = len(windows)
    win_shapes = []
    slices = []
    for (st, sh, sw) in windows:
        wt = st.stop - st.start; wh = sh.stop - sh.start; ww = sw.stop - sw.start
        win_shapes.append((wt, wh, ww))
        slices.append([st.start, st.stop, sh.start, sh.stop, sw.start, sw.stop])
    vid_freq, txt_freq = mmrope.build_window_freqs(win_shapes, TXT_LEN)
    with open(path, "wb") as fp:
        hdr = [GRID[0], GRID[1], GRID[2], WINDOW[0], WINDOW[1], WINDOW[2],
               len([w for w in windows if w[0].stop-w[0].start>0]),
               0, 0, nwin]
        # nnt/nh/nw 这里不需要精确，仅占位；C++ 只用 nwin + slices
        hdr = [GRID[0], GRID[1], GRID[2], WINDOW[0], WINDOW[1], WINDOW[2], 0, 0, 0, nwin]
        for v in hdr: fp.write(struct.pack("<q", int(v)))
        sl = np.array(slices, dtype=np.int64)
        fp.write(sl.tobytes())
        fp.write(np.ascontiguousarray(vid_freq, dtype=np.float32).tobytes())
        fp.write(np.ascontiguousarray(txt_freq, dtype=np.float32).tobytes())
    print(f"[ok] {method}: nwin={nwin} vid_freq={vid_freq.shape} txt_freq={txt_freq.shape} -> {path}")


if __name__ == "__main__":
    dump(os.path.join(M5, "win_nonshifted.bin"), "720pwin_by_size_bysize")
    dump(os.path.join(M5, "win_shifted.bin"), "720pswin_by_size_bysize")

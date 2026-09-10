#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import awa_window, mmrope
TXT=58; GRID=(1,30,45); WINDOW=(4,3,3)
for tag, windows in (("ns", awa_window.make_720Pwindows_bysize(GRID,WINDOW)),
                     ("sh", awa_window.make_shifted_720Pwindows_bysize(GRID,WINDOW))):
    print(f"== {tag} nwin={len(windows)} ==")
    total=0
    for w in windows:
        st,sh,sw=w
        wt=st.stop-st.start; wh=sh.stop-sh.start; ww=sw.stop-sw.start
        total+=wt*wh*ww
        print(f"  slice t[{st.start},{st.stop}] h[{sh.start},{sh.stop}] w[{sw.start},{sw.stop}] -> {wt}x{wh}x{ww}={wt*wh*ww}")
    print(f"  sum(窗口token)={total}  Lv={GRID[1]*GRID[2]}")

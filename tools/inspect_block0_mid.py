#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np, os
files = ['block0_v_cur_0.f32','block0_v_rn_0.f32','block0_v_m2_0.f32',
         'block0_v_qkv_0.f32','block0_v_attn_0.f32','block0_v_cur_0_a.f32']
for f in files:
    if not os.path.exists(f):
        print(f, 'MISSING'); continue
    d = np.fromfile(f, np.float32)
    fin = np.isfinite(d)
    print(f"{f}: n={d.size} range[{np.nanmin(d) if d.size else '-'},{np.nanmax(d) if d.size else '-'}] "
          f"nan={np.isnan(d).sum()} inf={np.isinf(d).sum()} finite_mean={d[fin].mean() if fin.any() else '-'}")

# 对比 numpy 参考的 vid_in
print("\n--- numpy 参考 (r_dbgr_*) ---")
for f in ['models/m5/r_dbgr_vid_an.bin','models/m5/r_dbgr_vid_b0_attn.bin','models/m5/r_dbgr_vid_b0_full.bin']:
    if not os.path.exists(f): print(f, 'MISSING'); continue
    d = np.fromfile(f, np.float32)[16:]   # skip int64 header
    print(f"{f}: n={d.size} range[{d.min()},{d.max()}]")

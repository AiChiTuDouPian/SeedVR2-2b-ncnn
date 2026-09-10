#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐层对比 engine 分块图输出 vs numpy 参考 (run_e2e_ref 逻辑)。找到第一个分叉层。"""
import os, sys, struct
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "export"))
import run_e2e_ref as ref

# ---- 参考输入 ----
wd = os.path.join(ROOT, "e2e_work")
vid_grid = ref.read_raw(os.path.join(wd, "vid_grid.bin"))
txt = ref.read_raw(os.path.join(wd, "txt.bin"))
with open(os.path.join(wd, "params.txt")) as fp:
    T, Hg, Wg, TXT_LEN, ts = fp.read().split()
    T, Hg, Wg, TXT_LEN = int(T), int(Hg), int(Wg), int(TXT_LEN)
    ts = float(ts)
H2, W2 = Hg // 2, Wg // 2
Lv = T * H2 * W2

# ---- 参考前向 ----
x_patch = ref.patchify(vid_grid, T, Hg, Wg)
x_txt = txt.astype(np.float32)
with ref.io.open_sd() as f:
    Wpin = ref.io.get_top_weight(f, "vid_in.proj", "weight"); Bpin = ref.io.get_top_weight(f, "vid_in.proj", "bias")
    Wtxt = ref.io.get_top_weight(f, "txt_in", "weight"); Btxt = ref.io.get_top_weight(f, "txt_in", "bias")
vid = x_patch @ Wpin.T + Bpin
txt_t = x_txt @ Wtxt.T + Btxt
emb = ref.time_embedding(ts)
emb3 = emb.reshape(1, ref.HEAD_D * ref.HEADS, 2, 3)

def cosim(a, b):
    return float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

# engine 块输出文件
blk_files = {}
for i in range(ref.NUM_LAYERS):
    p = os.path.join(ROOT, f"blkout_{i}_vid.f32")
    if os.path.exists(p):
        blk_files[i] = np.fromfile(p, np.float32)
    else:
        blk_files[i] = None

print(f"参考 vid: Lv={Lv} DIM={ref.DIM}")
first_bad = None
for i in range(ref.NUM_LAYERS):
    window_method = "720pwin_by_size_bysize" if i % 2 == 0 else "720pswin_by_size_bysize"
    Wb = ref.block_weights(i)
    Adv = Wb["vid"]["ada"]
    vid_an = ref.rmsnorm(vid); txt_an = ref.rmsnorm(txt_t)
    sA, scA, gA = emb3[0, :, 0, 0], emb3[0, :, 0, 1], emb3[0, :, 0, 2]
    shB = Adv["attn_shift"]; scB = Adv["attn_scale"]; gB = Adv["attn_gate"]
    thB = Wb["txt"]["ada"]["attn_shift"]; tcB = Wb["txt"]["ada"]["attn_scale"]; tgB = Wb["txt"]["ada"]["attn_gate"]
    vid_an = vid_an * (scA + scB) + (sA + shB)
    txt_an = txt_an * (scA + tcB) + (sA + thB)
    vid_at, txt_at = ref.awa_forward(vid_an, txt_an, (T, H2, W2), window_method, Wb["vid"], Wb["txt"], TXT_LEN)
    vid_at = vid_at * (gA + gB); txt_at = txt_at * (gA + tgB)
    vid = vid_at + vid; txt_t = txt_at + txt_t
    vid_mn = ref.rmsnorm(vid); txt_mn = ref.rmsnorm(txt_t)
    sAm, scAm, gAm = emb3[0, :, 1, 0], emb3[0, :, 1, 1], emb3[0, :, 1, 2]
    shBm = Adv["mlp_shift"]; scBm = Adv["mlp_scale"]; gBm = Adv["mlp_gate"]
    thBm = Wb["txt"]["ada"]["mlp_shift"]; tcBm = Wb["txt"]["ada"]["mlp_scale"]; tgBm = Wb["txt"]["ada"]["mlp_gate"]
    vid_mn = vid_mn * (scAm + scBm) + (sAm + shBm)
    txt_mn = txt_mn * (scAm + tcBm) + (sAm + thBm)
    vid_m = ref.swiglu(vid_mn, Wb["vid"]["mlp_in"], Wb["vid"]["mlp_g"], Wb["vid"]["mlp_out"])
    txt_m = ref.swiglu(txt_mn, Wb["txt"]["mlp_in"], Wb["txt"]["mlp_g"], Wb["txt"]["mlp_out"])
    vid_m = vid_m * (gAm + gBm); txt_m = txt_m * (gAm + tgBm)
    vid = vid_m + vid; txt_t = txt_m + txt_t
    # 对比 engine 块输出
    eng = blk_files.get(i)
    if eng is None:
        continue
    eng = eng.reshape(Lv, ref.DIM)
    c = cosim(vid, eng)
    d = np.abs(vid - eng).max()
    flag = "  <-- 分叉" if c < 0.99 and first_bad is None else ""
    print(f"层{i:2d}: cos={c:.6f} maxdiff={d:.4f}{flag}")
    if c < 0.99 and first_bad is None:
        first_bad = i
print(f"\n>>> 第一个分叉层: {first_bad}")

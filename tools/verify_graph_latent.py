#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比 engine(graph 路径) 的 DiT latent vs numpy 参考（复用 run_e2e_ref 的参考实现）。
engine latent: latent_sr.bin (Q header + H8 int + W8 int + H8*W8*16 f32, channel-last)
参考输入: e2e_work/vid_grid.bin + txt.bin + params.txt (seed=42, 与 engine 一致)
"""
import os, sys, struct, importlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "export"))

import run_e2e_ref as ref

# ---- 读取 engine 的 latent_sr.bin ----
with open(os.path.join(ROOT, "latent_sr.bin"), "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    H8 = struct.unpack("<i", f.read(4))[0]
    W8 = struct.unpack("<i", f.read(4))[0]
    sr_eng = np.frombuffer(f.read(), np.float32).reshape(1, H8, W8, 16)
print(f"engine latent_sr: {sr_eng.shape} range[{sr_eng.min():.4f},{sr_eng.max():.4f}]")

# ---- 参考输入 ----
wd = os.path.join(ROOT, "e2e_work")
vid_grid = ref.read_raw(os.path.join(wd, "vid_grid.bin"))
txt = ref.read_raw(os.path.join(wd, "txt.bin"))
with open(os.path.join(wd, "params.txt")) as fp:
    T, Hg, Wg, TXT_LEN, ts = fp.read().split()
    T, Hg, Wg, TXT_LEN = int(T), int(Hg), int(Wg), int(TXT_LEN)
    ts = float(ts)
print(f"参考 grid({T},{Hg},{Wg}) TXT={TXT_LEN} ts={ts}")

# ---- 重算 numpy 参考 sr_latent ----
H2, W2 = Hg // 2, Wg // 2
x_patch = ref.patchify(vid_grid, T, Hg, Wg)
x_txt = txt.astype(np.float32)

with ref.io.open_sd() as f:
    Wpin = ref.io.get_top_weight(f, "vid_in.proj", "weight"); Bpin = ref.io.get_top_weight(f, "vid_in.proj", "bias")
    Wtxt = ref.io.get_top_weight(f, "txt_in", "weight"); Btxt = ref.io.get_top_weight(f, "txt_in", "bias")
    Wvon = ref.io.get_top_weight(f, "vid_out_norm", "weight")
    Wvoa_s = ref.io.load_key(f, "vid_out_ada.out_shift")
    Wvoa_sc = ref.io.load_key(f, "vid_out_ada.out_scale")
    Wvout = ref.io.get_top_weight(f, "vid_out.proj", "weight"); Bvout = ref.io.get_top_weight(f, "vid_out.proj", "bias")

vid = x_patch @ Wpin.T + Bpin
txt_t = x_txt @ Wtxt.T + Btxt
emb = ref.time_embedding(ts)
emb3 = emb.reshape(1, ref.HEAD_D * ref.HEADS, 2, 3)

for i in range(ref.NUM_LAYERS):
    window_method = "720pwin_by_size_bysize" if i % 2 == 0 else "720pswin_by_size_bysize"
    Wb = ref.block_weights(i)
    dual = i < ref.MM_LAYERS
    Adv = Wb["vid"]["ada"]; Adt = Wb["txt"]["ada"]
    vid_an = ref.rmsnorm(vid); txt_an = ref.rmsnorm(txt_t)
    sA, scA, gA = emb3[0, :, 0, 0], emb3[0, :, 0, 1], emb3[0, :, 0, 2]
    shB = Adv["attn_shift"]; scB = Adv["attn_scale"]; gB = Adv["attn_gate"]
    thB = Adt["attn_shift"]; tcB = Adt["attn_scale"]; tgB = Adt["attn_gate"]
    vid_an = vid_an * (scA + scB) + (sA + shB)
    txt_an = txt_an * (scA + tcB) + (sA + thB)
    vid_at, txt_at = ref.awa_forward(vid_an, txt_an, (T, H2, W2), window_method, Wb["vid"], Wb["txt"], TXT_LEN)
    vid_at = vid_at * (gA + gB); txt_at = txt_at * (gA + tgB)
    vid = vid_at + vid; txt_t = txt_at + txt_t
    vid_mn = ref.rmsnorm(vid); txt_mn = ref.rmsnorm(txt_t)
    sAm, scAm, gAm = emb3[0, :, 1, 0], emb3[0, :, 1, 1], emb3[0, :, 1, 2]
    shBm = Adv["mlp_shift"]; scBm = Adv["mlp_scale"]; gBm = Adv["mlp_gate"]
    thBm = Adt["mlp_shift"]; tcBm = Adt["mlp_scale"]; tgBm = Adt["mlp_gate"]
    vid_mn = vid_mn * (scAm + scBm) + (sAm + shBm)
    txt_mn = txt_mn * (scAm + tcBm) + (sAm + thBm)
    vid_m = ref.swiglu(vid_mn, Wb["vid"]["mlp_in"], Wb["vid"]["mlp_g"], Wb["vid"]["mlp_out"])
    txt_m = ref.swiglu(txt_mn, Wb["txt"]["mlp_in"], Wb["txt"]["mlp_g"], Wb["txt"]["mlp_out"])
    vid_m = vid_m * (gAm + gBm); txt_m = txt_m * (gAm + tgBm)
    vid = vid_m + vid; txt_t = txt_m + txt_t

vid = ref.rmsnorm(vid, Wvon)
emb_out3 = emb[:, :ref.HEAD_D * ref.HEADS * 3].reshape(1, ref.HEAD_D * ref.HEADS, 1, 3)
sAo = emb_out3[0, :, 0, 0]; scAo = emb_out3[0, :, 0, 1]
vid = vid * (scAo + Wvoa_sc) + (sAo + Wvoa_s)
ref_latent = vid @ Wvout.T + Bvout
ref_grid = ref.unpatchify(ref_latent, T, Hg, Wg)
print(f"numpy 参考 sr_latent: {ref_grid.shape} range[{ref_grid.min():.4f},{ref_grid.max():.4f}]")

# ---- 对比 ----
c = float(np.dot(ref_grid.ravel(), sr_eng.ravel()) /
          (np.linalg.norm(ref_grid.ravel()) * np.linalg.norm(sr_eng.ravel()) + 1e-12))
d = np.abs(ref_grid - sr_eng)
print(f"\n>>> cos(engine graph latent, numpy ref) = {c:.6f}")
print(f"    max|diff| = {d.max():.4f}  mean = {d.mean():.6f}")
print(f"    PSNR(latent) = {10*np.log10((np.max(np.abs(ref_grid))**2+1e-12)/(d**2).mean()):.2f} dB")

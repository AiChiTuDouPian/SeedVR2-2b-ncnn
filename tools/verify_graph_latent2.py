#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比 engine(graph 路径) 的 DiT latent vs numpy 参考。
关键：用 engine 自己的 vid_grid.bin 作为参考输入，保证输入一致，才能判断引擎对错。
engine vid_grid: Q header + H8 int + W8 int + H8*W8*33 f32 channel-last
engine latent: latent_sr.bin (Q + H8 + W8 + H8*W8*16 f32)
"""
import os, sys, struct
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "export"))
import run_e2e_ref as ref

# ---- 读 engine 的 vid_grid.bin ----
with open(os.path.join(ROOT, "vid_grid.bin"), "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    H8 = struct.unpack("<i", f.read(4))[0]
    W8 = struct.unpack("<i", f.read(4))[0]
    vg = np.frombuffer(f.read(), np.float32).reshape(H8, W8, 33)
vg = vg[None, ...]  # (1,H8,W8,33)
print(f"engine vid_grid: {vg.shape}")

# ---- 参考输入：txt 用 engine 的（models/m5/txt.bin）+ params ----
# engine 用与 run_e2e 相同的 txt？读 e2e_work/txt.bin 或 models/m5/txt.bin
wd = os.path.join(ROOT, "e2e_work")
if os.path.exists(os.path.join(wd, "txt.bin")):
    txt = ref.read_raw(os.path.join(wd, "txt.bin"))
else:
    txt = np.fromfile(os.path.join(ROOT, "models/m5/txt.bin"), np.float32)
    txt = txt.reshape(58, 5120)
TXT_LEN = 58; ts = 1000.0
T, Hg, Wg = vg.shape[0], vg.shape[1], vg.shape[2]
H2, W2 = Hg // 2, Wg // 2
print(f"grid(T,H,W)=({T},{Hg},{Wg}) TXT={TXT_LEN}")

# ---- numpy 参考前向 ----
x_patch = ref.patchify(vg, T, Hg, Wg)
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

# ---- engine latent ----
with open(os.path.join(ROOT, "latent_sr.bin"), "rb") as f:
    n2 = struct.unpack("<Q", f.read(8))[0]
    H8b = struct.unpack("<i", f.read(4))[0]; W8b = struct.unpack("<i", f.read(4))[0]
    sr_eng = np.frombuffer(f.read(), np.float32).reshape(1, H8b, W8b, 16)
print(f"engine latent: {sr_eng.shape} range[{sr_eng.min():.4f},{sr_eng.max():.4f}]")

c = float(np.dot(ref_grid.ravel(), sr_eng.ravel()) /
          (np.linalg.norm(ref_grid.ravel()) * np.linalg.norm(sr_eng.ravel()) + 1e-12))
d = np.abs(ref_grid - sr_eng)
print(f"\n>>> cos(engine latent, numpy ref, 同输入) = {c:.6f}")
print(f"    max|diff| = {d.max():.4f}  mean = {d.mean():.6f}")

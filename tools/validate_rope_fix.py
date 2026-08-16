#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用「修正后的 mmrope」（ROPE_DIM=42 / 相邻配对 rotate_half）跑完整 numpy 参考前向，
与官方 DiT 的 x0（upscaled latent）对拍，确认 RoPE bug 修复后数值对齐官方。

用法（cuda_env）:
  python tools/validate_rope_fix.py [--workdir e2e_work] [--x0 official_intermediates/x0.bin]
"""
import sys, os, struct, argparse, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "export"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ncnn_io as io
import awa_window
import mmrope
import run_e2e_ref as R   # 复用其前向函数（time_embedding/block_weights/awa_forward/patchify/unpatchify/rmsnorm/swiglu/sinusoidal_embedding）


def read_raw(path):
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = [struct.unpack("<q", fp.read(8))[0] for _ in range(ndim)]
        total = int(np.prod(shape))
        d = np.frombuffer(fp.read(total * 4), dtype=np.float32)
    return d.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.join(ROOT, "e2e_work"))
    ap.add_argument("--x0", default=os.path.join(ROOT, "e2e_work", "official_intermediates", "x0.bin"))
    args = ap.parse_args()

    wd = args.workdir
    vid_grid = read_raw(os.path.join(wd, "vid_grid.bin"))
    txt = read_raw(os.path.join(wd, "txt.bin"))
    with open(os.path.join(wd, "params.txt")) as fp:
        T, Hg, Wg, TXT_LEN, ts = fp.read().split()
        T, Hg, Wg, TXT_LEN = int(T), int(Hg), int(Wg), int(TXT_LEN)
        ts = float(ts)
    print(f"[val] grid=({T},{Hg},{Wg}) TXT={TXT_LEN} ts={ts}  FREQ_LAST={mmrope.FREQ_LAST} (应为 126)")

    H2, W2 = Hg // 2, Wg // 2
    x_patch = R.patchify(vid_grid, T, Hg, Wg)
    x_txt = txt.astype(np.float32)

    with io.open_sd() as f:
        Wpin = io.get_top_weight(f, "vid_in.proj", "weight"); Bpin = io.get_top_weight(f, "vid_in.proj", "bias")
        Wtxt = io.get_top_weight(f, "txt_in", "weight"); Btxt = io.get_top_weight(f, "txt_in", "bias")
        Wvon = io.get_top_weight(f, "vid_out_norm", "weight")
        Wvoa_s = io.load_key(f, "vid_out_ada.out_shift")
        Wvoa_sc = io.load_key(f, "vid_out_ada.out_scale")
        Wvout = io.get_top_weight(f, "vid_out.proj", "weight"); Bvout = io.get_top_weight(f, "vid_out.proj", "bias")

    vid = x_patch @ Wpin.T + Bpin
    txt_t = x_txt @ Wtxt.T + Btxt
    emb = R.time_embedding(ts)
    emb3 = emb.reshape(1, R.HEAD_D * R.HEADS, 2, 3)

    for i in range(R.NUM_LAYERS):
        window_method = "720pwin_by_size_bysize" if i % 2 == 0 else "720pswin_by_size_bysize"
        Wb = R.block_weights(i)
        dual = i < R.MM_LAYERS
        Adv = Wb["vid"]["ada"]; Adt = Wb["txt"]["ada"]

        vid_an = R.rmsnorm(vid); txt_an = R.rmsnorm(txt_t)
        sA, scA, gA = emb3[0, :, 0, 0], emb3[0, :, 0, 1], emb3[0, :, 0, 2]
        shB = Adv["attn_shift"]; scB = Adv["attn_scale"]; gB = Adv["attn_gate"]
        thB = Adt["attn_shift"]; tcB = Adt["attn_scale"]; tgB = Adt["attn_gate"]
        vid_an = vid_an * (scA + scB) + (sA + shB)
        txt_an = txt_an * (scA + tcB) + (sA + thB)
        vid_at, txt_at = R.awa_forward(vid_an, txt_an, (T, H2, W2), window_method, Wb["vid"], Wb["txt"], TXT_LEN)
        vid_at = vid_at * (gA + gB); txt_at = txt_at * (gA + tgB)
        vid = vid_at + vid; txt_t = txt_at + txt_t

        vid_mn = R.rmsnorm(vid); txt_mn = R.rmsnorm(txt_t)
        sAm, scAm, gAm = emb3[0, :, 1, 0], emb3[0, :, 1, 1], emb3[0, :, 1, 2]
        shBm = Adv["mlp_shift"]; scBm = Adv["mlp_scale"]; gBm = Adv["mlp_gate"]
        thBm = Adt["mlp_shift"]; tcBm = Adt["mlp_scale"]; tgBm = Adt["mlp_gate"]
        vid_mn = vid_mn * (scAm + scBm) + (sAm + shBm)
        txt_mn = txt_mn * (scAm + tcBm) + (sAm + thBm)
        vid_m = R.swiglu(vid_mn, Wb["vid"]["mlp_in"], Wb["vid"]["mlp_g"], Wb["vid"]["mlp_out"])
        txt_m = R.swiglu(txt_mn, Wb["txt"]["mlp_in"], Wb["txt"]["mlp_g"], Wb["txt"]["mlp_out"])
        vid_m = vid_m * (gAm + gBm); txt_m = txt_m * (gAm + tgBm)
        vid = vid_m + vid; txt_t = txt_m + txt_t

    vid = R.rmsnorm(vid, Wvon)
    emb_out3 = emb[:, :R.HEAD_D * R.HEADS * 3].reshape(1, R.HEAD_D * R.HEADS, 1, 3)
    sAo = emb_out3[0, :, 0, 0]; scAo = emb_out3[0, :, 0, 1]
    vid = vid * (scAo + Wvoa_sc) + (sAo + Wvoa_s)
    ref = vid @ Wvout.T + Bvout
    ref_grid = R.unpatchify(ref, T, Hg, Wg)

    x0 = read_raw(args.x0)
    noise = read_raw(os.path.join(os.path.dirname(args.x0), "noise.bin"))
    print(f"[val] ref_grid(velocity v)={ref_grid.shape}  x0={x0.shape}  noise={noise.shape}")

    def stats(a, b):
        a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        mse = float(np.mean((a - b) ** 2)); psnr = 10 * math.log10(1.0 / (mse + 1e-12))
        return cos, mse, psnr

    # DiT 输出是 velocity v；官方 x0 = noise - v。必须相减后再比。
    upscaled_ref = noise - ref_grid
    cos, mse, psnr = stats(upscaled_ref, x0)
    print(f"[val] numpy-ref(upscaled=noise-v) vs OFFICIAL x0: cos={cos:.6f}  MSE={mse:.3e}  PSNR={psnr:.2f}dB")
    if cos >= 0.99:
        print("[PASS] RoPE 修复后 numpy 参考与官方 DiT 对齐 (cos>=0.99)")
    else:
        print("[FAIL] 仍未对齐，需继续排查")
    return 0 if cos >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())

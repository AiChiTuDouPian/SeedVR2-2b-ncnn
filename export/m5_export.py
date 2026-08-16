#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M5 导出 + 参考：完整 32 层 NaDiT 单次前向（1-step）。
- 输入：post-patchify 网格 GRID=(T,H,W)，patchified 随机 token (L, 33*1*2*2=132) -> vid_in.proj -> (L,2560)
        文本随机 token (TXT_LEN, 5120) -> txt_in(Linear) -> (TXT_LEN,2560)
- 逐 block（0-9 dual vid/txt，10-31 shared .all）：
    attn_norm(fusedrms 无affine) -> ada(in) -> AWA(偶数非shifted/奇数shifted) -> ada(out) -> 残差
    mlp_norm(fusedrms 无affine) -> ada(in) -> SwiGLU -> ada(out) -> 残差
- vid_out_norm(fusedrms affine) -> vid_out_ada(in) -> vid_out.proj -> (L,64)
- 对照：ref_vid_out (L,64)，及逐 block vid 输出 vid_b{b}.bin 供 C++ 分层调试。
所有 freq/窗口公式与 M3/M4 一致（窗口局部 RoPE；视频时序 offset l；capping 取 batch 最大）。
"""
import os, sys, math, struct
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncnn_io as io
import awa_window
import mmrope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M5 = os.path.join(ROOT, "models", "m5")
os.makedirs(M5, exist_ok=True)

HEADS = 20; HEAD_D = 128; DIM = 2560; QKV = HEADS*HEAD_D*3
EPS = 1e-5
NUM_LAYERS = 32; MM_LAYERS = 10
WINDOW = (4, 3, 3)
TXT_LEN = 8
TIMESTEP = 500.0
GRID = (2, 40, 40)   # post-patchify 网格 (T,H,W) -> L=3200


def save_raw(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<q", arr.ndim))
        for d in arr.shape: fp.write(struct.pack("<q", int(d)))
        fp.write(arr.tobytes())


def rmsnorm(x, w=None):
    ms = np.mean(x*x, axis=-1, keepdims=True)
    y = x / np.sqrt(ms + EPS)
    return y * w if w is not None else y


def swiglu(x, Win, Wg, Wout):
    h1 = x @ Win.T          # (L, 4D)
    hg = x @ Wg.T           # (L, 4D)
    h = (h1 / (1 + np.exp(-h1))) * hg   # SiLU(h1)*hg
    return h @ Wout.T


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)  # downscale_freq_shift=0 -> /half
    args = t.astype(np.float64)[:, None] * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=-1).astype(np.float32)


def time_embedding(t):
    # 加载 emb_in 权重
    with io.open_sd() as f:
        Wpi = io.get_top_weight(f, "emb_in.proj_in", "weight"); Bpi = io.get_top_weight(f, "emb_in.proj_in", "bias")
        Wph = io.get_top_weight(f, "emb_in.proj_hid", "weight"); Bph = io.get_top_weight(f, "emb_in.proj_hid", "bias")
        Wpo = io.get_top_weight(f, "emb_in.proj_out", "weight"); Bpo = io.get_top_weight(f, "emb_in.proj_out", "bias")
    e = sinusoidal_embedding(np.array([t], dtype=np.float32), 256)       # (1,256)
    e = e @ Wpi.T + Bpi; e = e / (1 + np.exp(-e))      # SiLU
    e = e @ Wph.T + Bph; e = e / (1 + np.exp(-e))      # SiLU
    e = e @ Wpo.T + Bpo                                 # (1,15360)
    return e


def ada_params(f, block, branch):
    out = {}
    for layer in ["attn", "mlp"]:
        for nm in ["shift", "scale", "gate"]:
            key = f"blocks.{block}.ada.{branch}.{layer}_{nm}"
            out[f"{layer}_{nm}"] = io.load_key(f, key)
    return out


def block_weights(block):
    """返回 {'vid':{...}, 'txt':{...}} 两套分支权重。
    dual block(0-9): vid/txt 各自独立；shared block(>=10): 都用 'all'。"""
    dual = block < MM_LAYERS
    bv = "vid" if dual else "all"
    bt = "txt" if dual else "all"
    with io.open_sd() as f:
        def L(sub, branch, name, sp=None):
            return io.load_key(f, io.block_key(block, sub, branch, name, shared_pattern=sp))
        W = {"vid": {}, "txt": {}}
        for br, slot in [(bv, "vid"), (bt, "txt")]:
            W[slot]["qkv"]    = L("proj_qkv", br, "weight")
            W[slot]["out"]    = L("proj_out", br, "weight")
            W[slot]["out_b"]  = L("proj_out", br, "bias")
            W[slot]["nq"]     = L("norm_q", br, "weight")
            W[slot]["nk"]     = L("norm_k", br, "weight")
            W[slot]["mlp_in"]  = L("proj_in", br, "weight", sp="mlp")
            W[slot]["mlp_g"]   = L("proj_in_gate", br, "weight", sp="mlp")
            W[slot]["mlp_out"] = L("proj_out", br, "weight", sp="mlp")
            W[slot]["ada"] = ada_params(f, block, br)
    return W


def awa_forward(vid, txt, grid, window_method, Wv, Wt):
    """单 block 的 AWA。Wv/Wt 为该 block 的 vid/txt 分支权重 dict。
    返回 vid(L,DIM), txt(TXT,DIM)（文本已跨窗口平均）。"""
    T, H, Wd = grid
    Lv = vid.shape[0]
    if window_method == "720pwin_by_size_bysize":
        windows = awa_window.make_720Pwindows_bysize(grid, WINDOW)
    else:
        windows = awa_window.make_shifted_720Pwindows_bysize(grid, WINDOW)
    nwin = len(windows)
    win_shapes, f_list = [], []
    for (st, sh, sw) in windows:
        wt = st.stop-st.start; wh = sh.stop-sh.start; ww = sw.stop-sw.start
        win_shapes.append((wt, wh, ww)); f_list.append(wt*wh*ww)
    M = sum(f_list)
    vid_freq, txt_freq = mmrope.build_window_freqs(win_shapes, TXT_LEN)

    # proj_qkv + qk_norm（分支各自）
    vqkv = (vid @ Wv["qkv"].T).reshape(Lv, 3, HEADS, HEAD_D)
    tqkv = (txt @ Wt["qkv"].T).reshape(TXT_LEN, 3, HEADS, HEAD_D)
    vq, vk, vv = vqkv[:, 0], vqkv[:, 1], vqkv[:, 2]
    tq, tk, tv = tqkv[:, 0], tqkv[:, 1], tqkv[:, 2]
    vq = rmsnorm(vq, Wv["nq"]); vk = rmsnorm(vk, Wv["nk"])
    tq = rmsnorm(tq, Wt["nq"]); tk = rmsnorm(tk, Wt["nk"])

    # partition
    vq_win = vq.reshape(T, H, Wd, HEADS, HEAD_D)
    vk_win = vk.reshape(T, H, Wd, HEADS, HEAD_D)
    vv_win = vv.reshape(T, H, Wd, HEADS, HEAD_D)
    vqw_list, vkw_list, vvw_list = [], [], []
    for (st, sh, sw) in windows:
        vqw_list.append(vq_win[st, sh, sw].reshape(-1, HEADS, HEAD_D))
        vkw_list.append(vk_win[st, sh, sw].reshape(-1, HEADS, HEAD_D))
        vvw_list.append(vv_win[st, sh, sw].reshape(-1, HEADS, HEAD_D))
    vqw = np.concatenate(vqw_list, 0); vkw = np.concatenate(vkw_list, 0); vvw = np.concatenate(vvw_list, 0)

    # RoPE
    vq_w = np.transpose(vqw, (1, 0, 2)); vk_w = np.transpose(vkw, (1, 0, 2)); vv_w = np.transpose(vvw, (1, 0, 2))
    vq_w = mmrope.apply_rotary_emb(vid_freq, vq_w); vk_w = mmrope.apply_rotary_emb(vid_freq, vk_w)
    tq_rep = np.tile(tq, (nwin, 1, 1)); tk_rep = np.tile(tk, (nwin, 1, 1)); tv_rep = np.tile(tv, (nwin, 1, 1))
    tq_w = np.transpose(tq_rep, (1, 0, 2)); tk_w = np.transpose(tk_rep, (1, 0, 2)); tv_w = np.transpose(tv_rep, (1, 0, 2))
    tq_w = mmrope.apply_rotary_emb(txt_freq, tq_w); tk_w = mmrope.apply_rotary_emb(txt_freq, tk_w)

    # varlen per-window SDPA + coalesce
    out_v_seg, out_t_seg = [], []
    vg = 0
    for wi in range(nwin):
        f_i = f_list[wi]
        q = np.concatenate([vq_w[:, vg:vg+f_i, :], tq_w[:, wi*TXT_LEN:(wi+1)*TXT_LEN, :]], 1)
        k = np.concatenate([vk_w[:, vg:vg+f_i, :], tk_w[:, wi*TXT_LEN:(wi+1)*TXT_LEN, :]], 1)
        v = np.concatenate([vv_w[:, vg:vg+f_i, :], tv_w[:, wi*TXT_LEN:(wi+1)*TXT_LEN, :]], 1)
        vg += f_i
        S = q.shape[1]
        o = np.empty((HEADS, S, HEAD_D), np.float32)
        scale = 1.0/math.sqrt(HEAD_D)
        for hh in range(HEADS):
            qq = q[hh]; kk = k[hh]; vv = v[hh]
            sc = (qq @ kk.T) * scale; sc = sc - sc.max(-1, keepdims=True)
            p = np.exp(sc); p = p / p.sum(-1, keepdims=True)
            o[hh] = p @ vv
        out_v_seg.append(o[:, :f_i, :]); out_t_seg.append(o[:, f_i:, :])
    # video reverse（非重叠精确逆）
    vout_target = np.zeros((T, H, Wd, HEADS, HEAD_D), np.float32)
    for wi, (st, sh, sw) in enumerate(windows):
        wt = st.stop-st.start; wh = sh.stop-sh.start; ww = sw.stop-sw.start
        blk = np.transpose(out_v_seg[wi], (1, 0, 2)).reshape(wt, wh, ww, HEADS, HEAD_D)
        vout_target[st, sh, sw] = blk
    vout_rev = vout_target.reshape(Lv, DIM)
    # 文本平均
    tout = np.stack(out_t_seg, 0).mean(0)
    tout = np.transpose(tout, (1, 0, 2)).reshape(TXT_LEN, DIM)
    # proj_out（分支各自，带 bias）
    vid_out = vout_rev @ Wv["out"].T
    if Wv["out_b"] is not None: vid_out = vid_out + Wv["out_b"]
    txt_out = tout @ Wt["out"].T
    if Wt["out_b"] is not None: txt_out = txt_out + Wt["out_b"]
    return vid_out, txt_out


def main():
    T, H, Wd = GRID
    Lv = T * H * Wd
    rng = np.random.RandomState(20240713)
    x_patch = rng.randn(Lv, 33*1*2*2).astype(np.float32)   # patchified 输入 (L,132)
    x_txt = rng.randn(TXT_LEN, 5120).astype(np.float32)

    # ---- 顶层权重 ----
    with io.open_sd() as f:
        Wpin = io.get_top_weight(f, "vid_in.proj", "weight"); Bpin = io.get_top_weight(f, "vid_in.proj", "bias")
        Wtxt = io.get_top_weight(f, "txt_in", "weight"); Btxt = io.get_top_weight(f, "txt_in", "bias")
        Wvon = io.get_top_weight(f, "vid_out_norm", "weight")
        Wvoa_s = io.load_key(f, "vid_out_ada.out_shift")
        Wvoa_sc = io.load_key(f, "vid_out_ada.out_scale")
        Wvout = io.get_top_weight(f, "vid_out.proj", "weight"); Bvout = io.get_top_weight(f, "vid_out.proj", "bias")

    # ---- patch in ----
    vid = x_patch @ Wpin.T + Bpin        # (L,2560)
    txt = x_txt @ Wtxt.T + Btxt          # (TXT,2560)
    emb = time_embedding(TIMESTEP)       # (1,15360)

    # ada emb 切片: emb (1,15360) -> (1,2560,2,3)
    emb3 = emb.reshape(1, HEAD_D*HEADS, 2, 3)   # (1,DIM,2,3)
    # 导出输入/中间
    save_raw(os.path.join(M5, "x_patch.bin"), x_patch)
    save_raw(os.path.join(M5, "x_txt.bin"), x_txt)
    save_raw(os.path.join(M5, "vid0.bin"), vid)
    save_raw(os.path.join(M5, "txt0.bin"), txt)
    save_raw(os.path.join(M5, "emb.bin"), emb)
    # 顶层权重导出（ncnn Linear）
    io.export_linear(os.path.join(M5, "vid_in_proj"), Wpin, Bpin, fp16=True)
    io.export_linear(os.path.join(M5, "txt_in"), Wtxt, Btxt, fp16=True)
    io.export_linear(os.path.join(M5, "vid_out_proj"), Wvout, Bvout, fp16=True)
    save_raw(os.path.join(M5, "vid_out_norm_w.bin"), Wvon)
    save_raw(os.path.join(M5, "vid_out_ada_shift.bin"), Wvoa_s)
    save_raw(os.path.join(M5, "vid_out_ada_scale.bin"), Wvoa_sc)

    # 导出 emb_in 权重（C++ 复现 TimeEmbedding 需要）
    with io.open_sd() as f:
        for nm in ["proj_in", "proj_hid", "proj_out"]:
            w = io.get_top_weight(f, f"emb_in.{nm}", "weight"); b = io.get_top_weight(f, f"emb_in.{nm}", "bias")
            io.export_linear(os.path.join(M5, f"emb_{nm}"), w, b, fp16=True)

    # 导出网格/window 配置
    with open(os.path.join(M5, "grid.txt"), "w") as fp:
        fp.write(f"{T} {H} {Wd} {TXT_LEN} {TIMESTEP}\n")
        fp.write(" ".join(map(str, WINDOW)) + "\n")

    # ---- 逐 block ----
    for i in range(NUM_LAYERS):
        window_method = "720pwin_by_size_bysize" if i % 2 == 0 else "720pswin_by_size_bysize"
        Wb = block_weights(i)                 # 返回 {"vid":{...}, "txt":{...}}
        dual = i < MM_LAYERS
        # 导出该 block 权重：dual(0-9) 导出 vid/txt 两套；shared(>=10) 仅 all 一套
        branches = [("vid", Wb["vid"]), ("txt", Wb["txt"])]
        exported_tags = set()
        for tag, Ws in branches:
            real_tag = tag if dual else "all"
            if real_tag in exported_tags:
                continue
            exported_tags.add(real_tag)
            ep = os.path.join(M5, f"b{i}_{real_tag}")
            io.export_linear(ep + "_qkv", Ws["qkv"], None, fp16=True)
            io.export_linear(ep + "_out", Ws["out"], Ws["out_b"], fp16=True)
            io.export_linear(ep + "_mlp_in", Ws["mlp_in"], None, fp16=True)
            io.export_linear(ep + "_mlp_g", Ws["mlp_g"], None, fp16=True)
            io.export_linear(ep + "_mlp_out", Ws["mlp_out"], None, fp16=True)
            save_raw(ep + "_nq.bin", Ws["nq"])
            save_raw(ep + "_nk.bin", Ws["nk"])
            for nm in ["attn_shift","attn_scale","attn_gate","mlp_shift","mlp_scale","mlp_gate"]:
                save_raw(ep + f"_{nm}.bin", Ws["ada"][nm])

        # vid/txt 分支 ada 参数（shared block 下二者相同，均来自 .all.）
        Adv = Wb["vid"]["ada"]; Adt = Wb["txt"]["ada"]

        # ---- block 计算 ----
        # attn_norm（fusedrms 无 affine）
        vid_an = rmsnorm(vid); txt_an = rmsnorm(txt)
        # ada attn in
        sA, scA, gA = emb3[0, :, 0, 0], emb3[0, :, 0, 1], emb3[0, :, 0, 2]  # (DIM,)
        shB = Adv["attn_shift"]; scB = Adv["attn_scale"]; gB = Adv["attn_gate"]
        thB = Adt["attn_shift"]; tcB = Adt["attn_scale"]; tgB = Adt["attn_gate"]
        vid_an = vid_an * (scA + scB) + (sA + shB)
        txt_an = txt_an * (scA + tcB) + (sA + thB)
        # attn (AWA)：vid/txt 用各自分支权重
        vid_at, txt_at = awa_forward(vid_an, txt_an, GRID, window_method, Wb["vid"], Wb["txt"])
        # ada attn out
        vid_at = vid_at * (gA + gB)
        txt_at = txt_at * (gA + tgB)
        vid = vid_at + vid
        txt = txt_at + txt
        # mlp_norm
        vid_mn = rmsnorm(vid); txt_mn = rmsnorm(txt)
        # ada mlp in
        sAm, scAm, gAm = emb3[0, :, 1, 0], emb3[0, :, 1, 1], emb3[0, :, 1, 2]
        shBm = Adv["mlp_shift"]; scBm = Adv["mlp_scale"]; gBm = Adv["mlp_gate"]
        thBm = Adt["mlp_shift"]; tcBm = Adt["mlp_scale"]; tgBm = Adt["mlp_gate"]
        vid_mn = vid_mn * (scAm + scBm) + (sAm + shBm)
        txt_mn = txt_mn * (scAm + tcBm) + (sAm + thBm)
        # mlp (SwiGLU)
        vid_m = swiglu(vid_mn, Wb["vid"]["mlp_in"], Wb["vid"]["mlp_g"], Wb["vid"]["mlp_out"])
        txt_m = swiglu(txt_mn, Wb["txt"]["mlp_in"], Wb["txt"]["mlp_g"], Wb["txt"]["mlp_out"])
        # ada mlp out
        vid_m = vid_m * (gAm + gBm)
        txt_m = txt_m * (gAm + tgBm)
        vid = vid_m + vid
        txt = txt_m + txt

        save_raw(os.path.join(M5, f"vid_b{i}.bin"), vid)
        if i == 0:
            save_raw(os.path.join(M5, "txt_b0.bin"), txt)

    # ---- vid_out ----
    vid = rmsnorm(vid, Wvon)                       # vid_out_norm (affine)
    # vid_out_ada 的 layers=["out"] -> emb 按 (1,D,1,3) reshape，只取前 D*3 段
    emb_out3 = emb[:, :HEAD_D*HEADS*3].reshape(1, HEAD_D*HEADS, 1, 3)
    sAo = emb_out3[0, :, 0, 0]; scAo = emb_out3[0, :, 0, 1]
    vid = vid * (scAo + Wvoa_sc) + (sAo + Wvoa_s)   # ada(in): scale/shift
    vid = vid @ Wvout.T + Bvout                     # (L,64)
    save_raw(os.path.join(M5, "ref_vid_out.bin"), vid)
    print(f"[ok] M5 导出完成: grid={GRID} Lv={Lv} TXT={TXT_LEN} timestep={TIMESTEP}")
    print(f"     ref_vid_out shape={vid.shape}")


if __name__ == "__main__":
    main()

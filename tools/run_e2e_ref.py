#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
端到端数值校验：用真实 DiT 权重的 numpy 参考实现（m5_export 同款），
对拍 C++ Vulkan 引擎在「真实网格 + TXT_LEN=58」下的输出 sr_latent.bin。

复用 run_e2e.py 写到 workdir 的输入（vid_grid.bin / txt.bin / params.txt），
构建与 C++ forward_grid 完全一致的参考前向，并把参考 (L,64) 反 patchify 后
与 sr_latent.bin 比较 cos。

用法（cuda_env，需在 run_e2e.py 之后）:
  python tools/run_e2e_ref.py [--workdir e2e_work]
"""
import sys, os, struct, argparse, math
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "export"))
import ncnn_io as io
import awa_window
import mmrope

HEADS = 20
HEAD_D = 128
DIM = 2560
QKV = HEADS * HEAD_D * 3
EPS = 1e-5
NUM_LAYERS = 32
MM_LAYERS = 10
WINDOW = (4, 3, 3)


def read_raw(path):
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = [struct.unpack("<q", fp.read(8))[0] for _ in range(ndim)]
        total = int(np.prod(shape))
        d = np.frombuffer(fp.read(total * 4), dtype=np.float32)
    return d.reshape(shape)


def write_raw(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<q", arr.ndim))
        for s in arr.shape:
            fp.write(struct.pack("<q", int(s)))
        fp.write(arr.tobytes())


def rmsnorm(x, w=None):
    ms = np.mean(x * x, axis=-1, keepdims=True)
    y = x / np.sqrt(ms + EPS)
    return y * w if w is not None else y


def swiglu(x, Win, Wg, Wout):
    # 官方 SwiGLUMLP: proj_out( silu(proj_in_gate(x)) * proj_in(x) )
    # 注意 silu 作用在 proj_in_gate（即 Wg），而非 proj_in（Win）。
    h_in = x @ Win.T          # proj_in   （不激活）
    h_gate = x @ Wg.T         # proj_in_gate（silu 激活）
    h = (h_gate / (1 + np.exp(-h_gate))) * h_in
    return h @ Wout.T


def sinusoidal_embedding(t, dim):
    half = dim // 2
    freqs = np.exp(-np.log(10000.0) * np.arange(half, dtype=np.float64) / half)
    args = t.astype(np.float64)[:, None] * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=-1).astype(np.float32)


def time_embedding(t):
    with io.open_sd() as f:
        Wpi = io.get_top_weight(f, "emb_in.proj_in", "weight"); Bpi = io.get_top_weight(f, "emb_in.proj_in", "bias")
        Wph = io.get_top_weight(f, "emb_in.proj_hid", "weight"); Bph = io.get_top_weight(f, "emb_in.proj_hid", "bias")
        Wpo = io.get_top_weight(f, "emb_in.proj_out", "weight"); Bpo = io.get_top_weight(f, "emb_in.proj_out", "bias")
    e = sinusoidal_embedding(np.array([t], dtype=np.float32), 256)
    e = e @ Wpi.T + Bpi; e = e / (1 + np.exp(-e))
    e = e @ Wph.T + Bph; e = e / (1 + np.exp(-e))
    e = e @ Wpo.T + Bpo
    return e


def ada_params(f, block, branch):
    out = {}
    for layer in ["attn", "mlp"]:
        for nm in ["shift", "scale", "gate"]:
            key = f"blocks.{block}.ada.{branch}.{layer}_{nm}"
            out[f"{layer}_{nm}"] = io.load_key(f, key)
    return out


def block_weights(block):
    dual = block < MM_LAYERS
    bv = "vid" if dual else "all"
    bt = "txt" if dual else "all"
    with io.open_sd() as f:
        def L(sub, branch, name, sp=None):
            return io.load_key(f, io.block_key(block, sub, branch, name, shared_pattern=sp))
        W = {"vid": {}, "txt": {}}
        for br, slot in [(bv, "vid"), (bt, "txt")]:
            W[slot]["qkv"] = L("proj_qkv", br, "weight")
            W[slot]["out"] = L("proj_out", br, "weight")
            W[slot]["out_b"] = L("proj_out", br, "bias")
            W[slot]["nq"] = L("norm_q", br, "weight")
            W[slot]["nk"] = L("norm_k", br, "weight")
            W[slot]["mlp_in"] = L("proj_in", br, "weight", sp="mlp")
            W[slot]["mlp_g"] = L("proj_in_gate", br, "weight", sp="mlp")
            W[slot]["mlp_out"] = L("proj_out", br, "weight", sp="mlp")
            W[slot]["ada"] = ada_params(f, block, br)
    return W


def awa_forward(vid, txt, grid, window_method, Wv, Wt, TXT_LEN):
    T, H, Wd = grid
    Lv = vid.shape[0]
    if window_method == "720pwin_by_size_bysize":
        windows = awa_window.make_720Pwindows_bysize(grid, WINDOW)
    else:
        windows = awa_window.make_shifted_720Pwindows_bysize(grid, WINDOW)
    nwin = len(windows)
    win_shapes, f_list = [], []
    for (st, sh, sw) in windows:
        wt = st.stop - st.start; wh = sh.stop - sh.start; ww = sw.stop - sw.start
        win_shapes.append((wt, wh, ww)); f_list.append(wt * wh * ww)
    M = sum(f_list)
    vid_freq, txt_freq = mmrope.build_window_freqs(win_shapes, TXT_LEN)

    vqkv = (vid @ Wv["qkv"].T).reshape(Lv, 3, HEADS, HEAD_D)
    tqkv = (txt @ Wt["qkv"].T).reshape(TXT_LEN, 3, HEADS, HEAD_D)
    vq, vk, vv = vqkv[:, 0], vqkv[:, 1], vqkv[:, 2]
    tq, tk, tv = tqkv[:, 0], tqkv[:, 1], tqkv[:, 2]
    vq = rmsnorm(vq, Wv["nq"]); vk = rmsnorm(vk, Wv["nk"])
    tq = rmsnorm(tq, Wt["nq"]); tk = rmsnorm(tk, Wt["nk"])

    vq_win = vq.reshape(T, H, Wd, HEADS, HEAD_D)
    vk_win = vk.reshape(T, H, Wd, HEADS, HEAD_D)
    vv_win = vv.reshape(T, H, Wd, HEADS, HEAD_D)
    vqw_list, vkw_list, vvw_list = [], [], []
    for (st, sh, sw) in windows:
        vqw_list.append(vq_win[st, sh, sw].reshape(-1, HEADS, HEAD_D))
        vkw_list.append(vk_win[st, sh, sw].reshape(-1, HEADS, HEAD_D))
        vvw_list.append(vv_win[st, sh, sw].reshape(-1, HEADS, HEAD_D))
    vqw = np.concatenate(vqw_list, 0); vkw = np.concatenate(vkw_list, 0); vvw = np.concatenate(vvw_list, 0)

    vq_w = np.transpose(vqw, (1, 0, 2)); vk_w = np.transpose(vkw, (1, 0, 2)); vv_w = np.transpose(vvw, (1, 0, 2))
    vq_w = mmrope.apply_rotary_emb(vid_freq, vq_w); vk_w = mmrope.apply_rotary_emb(vid_freq, vk_w)
    tq_rep = np.tile(tq, (nwin, 1, 1)); tk_rep = np.tile(tk, (nwin, 1, 1)); tv_rep = np.tile(tv, (nwin, 1, 1))
    tq_w = np.transpose(tq_rep, (1, 0, 2)); tk_w = np.transpose(tk_rep, (1, 0, 2)); tv_w = np.transpose(tv_rep, (1, 0, 2))
    tq_w = mmrope.apply_rotary_emb(txt_freq, tq_w); tk_w = mmrope.apply_rotary_emb(txt_freq, tk_w)

    out_v_seg, out_t_seg = [], []
    vg = 0
    for wi in range(nwin):
        f_i = f_list[wi]
        q = np.concatenate([vq_w[:, vg:vg + f_i, :], tq_w[:, wi * TXT_LEN:(wi + 1) * TXT_LEN, :]], 1)
        k = np.concatenate([vk_w[:, vg:vg + f_i, :], tk_w[:, wi * TXT_LEN:(wi + 1) * TXT_LEN, :]], 1)
        v = np.concatenate([vv_w[:, vg:vg + f_i, :], tv_w[:, wi * TXT_LEN:(wi + 1) * TXT_LEN, :]], 1)
        vg += f_i
        S = q.shape[1]
        o = np.empty((HEADS, S, HEAD_D), np.float32)
        scale = 1.0 / math.sqrt(HEAD_D)
        for hh in range(HEADS):
            qq = q[hh]; kk = k[hh]; vv = v[hh]
            sc = (qq @ kk.T) * scale; sc = sc - sc.max(-1, keepdims=True)
            p = np.exp(sc); p = p / p.sum(-1, keepdims=True)
            o[hh] = p @ vv
        out_v_seg.append(o[:, :f_i, :]); out_t_seg.append(o[:, f_i:, :])
    vout_target = np.zeros((T, H, Wd, HEADS, HEAD_D), np.float32)
    for wi, (st, sh, sw) in enumerate(windows):
        wt = st.stop - st.start; wh = sh.stop - sh.start; ww = sw.stop - sw.start
        blk = np.transpose(out_v_seg[wi], (1, 0, 2)).reshape(wt, wh, ww, HEADS, HEAD_D)
        vout_target[st, sh, sw] = blk
    vout_rev = vout_target.reshape(Lv, DIM)
    tout = np.stack(out_t_seg, 0).mean(0)
    tout = np.transpose(tout, (1, 0, 2)).reshape(TXT_LEN, DIM)
    vid_out = vout_rev @ Wv["out"].T
    if Wv["out_b"] is not None:
        vid_out = vid_out + Wv["out_b"]
    txt_out = tout @ Wt["out"].T
    if Wt["out_b"] is not None:
        txt_out = txt_out + Wt["out_b"]
    return vid_out, txt_out


def patchify(vid_grid, T, Hg, Wg):
    H2, W2 = Hg // 2, Wg // 2
    Lv = T * H2 * W2
    vid_patch = np.zeros((Lv, 33 * 1 * 2 * 2), dtype=np.float32)
    for t in range(T):
        for hh in range(H2):
            for ww in range(W2):
                for ph in range(2):
                    for pw in range(2):
                        for c in range(33):
                            gi = ((t * Hg + (hh * 2 + ph)) * Wg + (ww * 2 + pw)) * 33 + c
                            pi = ((t * H2 + hh) * W2 + ww) * 132 + ((ph * 2 + pw) * 33 + c)
                            vid_patch.flat[pi] = vid_grid.flat[gi]
    return vid_patch


def unpatchify(ref, T, Hg, Wg):
    H2, W2 = Hg // 2, Wg // 2
    sr = np.zeros((T, Hg, Wg, 16), dtype=np.float32)
    for t in range(T):
        for hh in range(H2):
            for ww in range(W2):
                for ph in range(2):
                    for pw in range(2):
                        for c in range(16):
                            li = ((t * H2 + hh) * W2 + ww) * 64 + ((ph * 2 + pw) * 16 + c)
                            gi = ((t * Hg + (hh * 2 + ph)) * Wg + (ww * 2 + pw)) * 16 + c
                            sr.flat[gi] = ref.flat[li]
    return sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default=os.path.join(ROOT, "e2e_work"))
    ap.add_argument("--threshold", type=float, default=0.99)
    args = ap.parse_args()

    wd = args.workdir
    vid_grid = read_raw(os.path.join(wd, "vid_grid.bin"))   # (T,H,W,33)
    txt = read_raw(os.path.join(wd, "txt.bin"))             # (TXT,5120)
    sr_cpp = read_raw(os.path.join(wd, "sr_latent.bin"))    # (T,H,W,16)
    with open(os.path.join(wd, "params.txt")) as fp:
        T, Hg, Wg, TXT_LEN, ts = fp.read().split()
        T, Hg, Wg, TXT_LEN = int(T), int(Hg), int(Wg), int(TXT_LEN)
        ts = float(ts)
    print(f"[ref] grid(T,H,W)=({T},{Hg},{Wg}) TXT={TXT_LEN} ts={ts}  vid_grid={vid_grid.shape} sr_cpp={sr_cpp.shape}")

    H2, W2 = Hg // 2, Wg // 2
    x_patch = patchify(vid_grid, T, Hg, Wg)
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
    emb = time_embedding(ts)
    emb3 = emb.reshape(1, HEAD_D * HEADS, 2, 3)

    for i in range(NUM_LAYERS):
        window_method = "720pwin_by_size_bysize" if i % 2 == 0 else "720pswin_by_size_bysize"
        Wb = block_weights(i)
        dual = i < MM_LAYERS
        Adv = Wb["vid"]["ada"]; Adt = Wb["txt"]["ada"]

        vid_an = rmsnorm(vid); txt_an = rmsnorm(txt_t)
        sA, scA, gA = emb3[0, :, 0, 0], emb3[0, :, 0, 1], emb3[0, :, 0, 2]
        shB = Adv["attn_shift"]; scB = Adv["attn_scale"]; gB = Adv["attn_gate"]
        thB = Adt["attn_shift"]; tcB = Adt["attn_scale"]; tgB = Adt["attn_gate"]
        vid_an = vid_an * (scA + scB) + (sA + shB)
        txt_an = txt_an * (scA + tcB) + (sA + thB)
        vid_at, txt_at = awa_forward(vid_an, txt_an, (T, H2, W2), window_method, Wb["vid"], Wb["txt"], TXT_LEN)
        vid_at = vid_at * (gA + gB); txt_at = txt_at * (gA + tgB)
        vid = vid_at + vid; txt_t = txt_at + txt_t

        vid_mn = rmsnorm(vid); txt_mn = rmsnorm(txt_t)
        sAm, scAm, gAm = emb3[0, :, 1, 0], emb3[0, :, 1, 1], emb3[0, :, 1, 2]
        shBm = Adv["mlp_shift"]; scBm = Adv["mlp_scale"]; gBm = Adv["mlp_gate"]
        thBm = Adt["mlp_shift"]; tcBm = Adt["mlp_scale"]; tgBm = Adt["mlp_gate"]
        vid_mn = vid_mn * (scAm + scBm) + (sAm + shBm)
        txt_mn = txt_mn * (scAm + tcBm) + (sAm + thBm)
        vid_m = swiglu(vid_mn, Wb["vid"]["mlp_in"], Wb["vid"]["mlp_g"], Wb["vid"]["mlp_out"])
        txt_m = swiglu(txt_mn, Wb["txt"]["mlp_in"], Wb["txt"]["mlp_g"], Wb["txt"]["mlp_out"])
        vid_m = vid_m * (gAm + gBm); txt_m = txt_m * (gAm + tgBm)
        vid = vid_m + vid; txt_t = txt_m + txt_t

    vid = rmsnorm(vid, Wvon)
    emb_out3 = emb[:, :HEAD_D * HEADS * 3].reshape(1, HEAD_D * HEADS, 1, 3)
    sAo = emb_out3[0, :, 0, 0]; scAo = emb_out3[0, :, 0, 1]
    vid = vid * (scAo + Wvoa_sc) + (sAo + Wvoa_s)
    ref = vid @ Wvout.T + Bvout                       # (L,64)
    write_raw(os.path.join(wd, "sr_latent_py.bin"), ref.astype(np.float32))  # 与 C++ 端 sr_latent 直接对比

    ref_grid = unpatchify(ref, T, Hg, Wg)             # (T,H,W,16)
    c = float(np.dot(ref_grid.ravel(), sr_cpp.ravel()) /
              (np.linalg.norm(ref_grid.ravel()) * np.linalg.norm(sr_cpp.ravel()) + 1e-12))
    print(f"[ref] unpatchify ref_grid={ref_grid.shape}  cos(C++ ref, numpy ref)={c:.6f}")
    if c >= args.threshold:
        print(f"[PASS] 端到端 DiT 数值对拍通过（cos>={args.threshold}）")
        return 0
    print(f"[FAIL] cos 不足阈值 {args.threshold}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

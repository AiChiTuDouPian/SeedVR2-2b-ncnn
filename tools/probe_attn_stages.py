#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分阶段对拍 awa_forward（block 0）：hook 官方 NaSwinAttention 的 norm_q/rope/attn
各级张量，逐层对比我们的 numpy 实现，定位 2% 注意力误差来源。
基于 debug_attn_fp32（模型 fp32，VAE 放 cpu 省显存）。
"""
import sys, os, struct, math
import numpy as np
import torch

COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
sys.path.insert(0, COMFY)
from src.utils.debug import Debug
from src.core.generation_utils import (
    setup_generation_context, prepare_runner, load_text_embeddings, script_directory,
)
from src.utils.model_registry import DEFAULT_VAE
from src.models.dit_3b import na
from src.common.cache import Cache

DEVICE = "cuda:0"
MODEL_DIR = os.path.join(COMFY, "models", "SEEDVR2")
DIT_MODEL = "seedvr2_ema_3b_fp16.safetensors"
ROOT = "F:/Seedvr2/seedvr2-ncnn"
WORK = os.path.join(ROOT, "e2e_work")
sys.path.insert(0, os.path.join(ROOT, "export"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ncnn_io as io
import mmrope
import run_e2e_ref as R


def read_raw(path):
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = [struct.unpack("<q", fp.read(8))[0] for _ in range(ndim)]
        d = np.frombuffer(fp.read(int(np.prod(shape)) * 4), dtype=np.float32)
    return d.reshape(shape)


def cos(a, b):
    a = np.asarray(a).ravel().astype(np.float64); b = np.asarray(b).ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@torch.no_grad()
def main():
    vid_grid = read_raw(os.path.join(WORK, "vid_grid.bin"))
    T, H, W = vid_grid.shape[0], vid_grid.shape[1], vid_grid.shape[2]
    noise_t = torch.from_numpy(vid_grid[..., :16].copy()).to(DEVICE).float()
    cond17 = torch.from_numpy(vid_grid[..., 16:].copy()).to(DEVICE).float()
    vid_input = torch.cat([noise_t, cond17], dim=-1)
    vid_flat, vid_shape = na.flatten([vid_input])
    txt_raw = torch.from_numpy(read_raw(os.path.join(WORK, "txt.bin")).astype(np.float32)).to(DEVICE).float()

    ctx = setup_generation_context(dit_device=DEVICE, vae_device="cpu",
                                   dit_offload_device=None, vae_offload_device=None,
                                   tensor_offload_device=None, debug=Debug(enabled=False))
    runner, cache_context = prepare_runner(
        dit_model=DIT_MODEL, vae_model=DEFAULT_VAE, model_dir=MODEL_DIR, debug=Debug(enabled=False),
        ctx=ctx, dit_cache=False, vae_cache=False, dit_id=None, vae_id=None,
        block_swap_config={'blocks_to_swap': 0, 'swap_io_components': False, 'offload_device': 'none'},
        attention_mode='sdpa')
    ctx['cache_context'] = cache_context
    ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], ctx['compute_dtype'], Debug(enabled=False))
    from src.core.model_loader import materialize_model
    if next(runner.dit.parameters()).device.type == "meta":
        materialize_model(runner, "dit", DEVICE, runner.config, Debug(enabled=False))
    dit = runner.dit.to(DEVICE).float().eval()

    txt_flat, txt_shape = na.flatten([txt_raw])
    txt_proj = dit.txt_in(txt_flat)
    tstept = torch.tensor([1000.0], device=DEVICE)
    emb_t = dit.emb_in(tstept, device=DEVICE, dtype=torch.float)
    patched_vid, patched_vid_shape = dit.vid_in(vid_flat, vid_shape)

    cap = {}
    def h_norm(module, inp, out):
        # inp: (vid_q, txt_q) ; out: (vid_qn, txt_qn)  shape (L, h, d) windowed
        cap['norm_vid'] = out[0].detach().cpu().numpy().astype(np.float32)
        cap['norm_txt'] = out[1].detach().cpu().numpy().astype(np.float32)
    def h_rope(module, inp, out):
        # inp: (vid_q, vid_k, window_shape, txt_q, txt_k, txt_shape_repeat, cache)
        # out: (vid_q, vid_k, txt_q, txt_k)  windowed
        cap['rope_vid_q'] = out[0].detach().cpu().numpy().astype(np.float32)
        cap['rope_vid_k'] = out[1].detach().cpu().numpy().astype(np.float32)
        cap['rope_txt_q'] = out[2].detach().cpu().numpy().astype(np.float32)
        cap['rope_txt_k'] = out[3].detach().cpu().numpy().astype(np.float32)
    def h_attn(module, inp, out):
        cap['attn_out'] = out.detach().cpu().numpy().astype(np.float32)

    blk = dit.blocks[0].attn
    h1 = blk.norm_q.register_forward_hook(h_norm)
    h2 = blk.rope.register_forward_hook(h_rope)
    h3 = blk.attn.register_forward_hook(h_attn)

    cache = Cache(disable=True)
    off_vid, off_txt, _, _ = dit.blocks[0](
        vid=patched_vid, txt=txt_proj, vid_shape=patched_vid_shape, txt_shape=txt_shape,
        emb=emb_t, cache=cache)
    h1.remove(); h2.remove(); h3.remove()

    attn_off_vid = off_vid.detach().cpu().numpy().astype(np.float32)  # post proj_out

    # === 我们的 numpy awa_forward（block 0）===
    our_vid = patched_vid.detach().cpu().numpy().astype(np.float32)
    our_txt = txt_proj.detach().cpu().numpy().astype(np.float32)
    emb_np = R.time_embedding(1000.0).astype(np.float32)
    emb3 = emb_np.reshape(1, R.HEAD_D * R.HEADS, 2, 3)
    Wb = R.block_weights(0)
    Adv = Wb["vid"]["ada"]; Adt = Wb["txt"]["ada"]
    vid_an = R.rmsnorm(our_vid); txt_an = R.rmsnorm(our_txt)
    sA, scA, gA = emb3[0, :, 0, 0], emb3[0, :, 0, 1], emb3[0, :, 0, 2]
    shB = Adv["attn_shift"]; scB = Adv["attn_scale"]; gB = Adv["attn_gate"]
    thB = Adt["attn_shift"]; tcB = Adt["attn_scale"]; tgB = Adt["attn_gate"]
    vid_an = vid_an * (scA + scB) + (sA + shB)
    txt_an = txt_an * (scA + tcB) + (sA + thB)
    window_method = "720pwin_by_size_bysize"
    # 复现 awa_forward 内部到 rope 阶段；直接用 R.awa_forward 不便拆分，这里复制关键段
    grid = (T, H//2, W//2)
    Lv = vid_an.shape[0]
    if window_method == "720pwin_by_size_bysize":
        windows = awa_window_make(grid, R.WINDOW)
    else:
        windows = awa_window_make_shifted(grid, R.WINDOW)
    nwin = len(windows)
    win_shapes = []
    for (st, sh, sw) in windows:
        win_shapes.append((st.stop-st.start, sh.stop-sh.start, sw.stop-sw.start))
    vid_freq, txt_freq = mmrope.build_window_freqs(win_shapes, R.TXT_LEN if hasattr(R,"TXT_LEN") else 58)
    TXT_LEN = 58
    vqkv = (vid_an @ Wb["vid"]["qkv"].T).reshape(Lv, 3, R.HEADS, R.HEAD_D)
    tqkv = (txt_an @ Wb["txt"]["qkv"].T).reshape(TXT_LEN, 3, R.HEADS, R.HEAD_D)
    vq, vk, vv = vqkv[:,0], vqkv[:,1], vqkv[:,2]
    tq, tk, tv = tqkv[:,0], tqkv[:,1], tqkv[:,2]
    vq = R.rmsnorm(vq, Wb["vid"]["nq"]); vk = R.rmsnorm(vk, Wb["vid"]["nk"])
    tq = R.rmsnorm(tq, Wb["txt"]["nq"]); tk = R.rmsnorm(tk, Wb["txt"]["nk"])
    vq_win = vq.reshape(T, H//2, W//2, R.HEADS, R.HEAD_D)
    vk_win = vk.reshape(T, H//2, W//2, R.HEADS, R.HEAD_D)
    vv_win = vv.reshape(T, H//2, W//2, R.HEADS, R.HEAD_D)
    vqw_list, vkw_list, vvw_list = [], [], []
    for (st, sh, sw) in windows:
        vqw_list.append(vq_win[st, sh, sw].reshape(-1, R.HEADS, R.HEAD_D))
        vkw_list.append(vk_win[st, sh, sw].reshape(-1, R.HEADS, R.HEAD_D))
        vvw_list.append(vv_win[st, sh, sw].reshape(-1, R.HEADS, R.HEAD_D))
    vqw = np.concatenate(vqw_list,0); vkw = np.concatenate(vkw_list,0); vvw = np.concatenate(vvw_list,0)
    vq_w = np.transpose(vqw,(1,0,2)); vk_w = np.transpose(vkw,(1,0,2)); vv_w = np.transpose(vvw,(1,0,2))
    vq_w = mmrope.apply_rotary_emb(vid_freq, vq_w); vk_w = mmrope.apply_rotary_emb(vid_freq, vk_w)
    tq_rep = np.tile(tq,(nwin,1,1)); tk_rep = np.tile(tk,(nwin,1,1)); tv_rep = np.tile(tv,(nwin,1,1))
    tq_w = np.transpose(tq_rep,(1,0,2)); tk_w = np.transpose(tk_rep,(1,0,2)); tv_w = np.transpose(tv_rep,(1,0,2))
    tq_w = mmrope.apply_rotary_emb(txt_freq, tq_w); tk_w = mmrope.apply_rotary_emb(txt_freq, tk_w)
    # 我们的 windowed q/k (L_win, h, d)
    our_rope_vid_q = np.transpose(vq_w,(1,0,2))   # (L_win, h, d)
    our_rope_vid_k = np.transpose(vk_w,(1,0,2))
    our_rope_txt_q = np.transpose(tq_w,(1,0,2))
    our_rope_txt_k = np.transpose(tk_w,(1,0,2))

    print("=== 阶段 cos（官方 vs 我们 的 numpy）===")
    # norm 是 windowed vs 我们的 full-seq；仅看分布
    print(f"[norm] vid 官方 windowed mean/std = {cap['norm_vid'].mean():.4f}/{cap['norm_vid'].std():.4f}  我们 full mean/std = {R.rmsnorm(vq,Wb['vid']['nq']).mean():.4f}/{R.rmsnorm(vq,Wb['vid']['nq']).std():.4f}")
    print(f"[norm] txt 官方 windowed mean/std = {cap['norm_txt'].mean():.4f}/{cap['norm_txt'].std():.4f}  我们 full mean/std = {R.rmsnorm(tq,Wb['txt']['nq']).mean():.4f}/{R.rmsnorm(tq,Wb['txt']['nq']).std():.4f}")

    print(f"[rope] vid_q : {cos(cap['rope_vid_q'], our_rope_vid_q):.6f}")
    print(f"[rope] vid_k : {cos(cap['rope_vid_k'], our_rope_vid_k):.6f}")
    print(f"[rope] txt_q : {cos(cap['rope_txt_q'], our_rope_txt_q):.6f}")
    print(f"[rope] txt_k : {cos(cap['rope_txt_k'], our_rope_txt_k):.6f}")

    # 用官方 rope 输出的 q/k/v，自己跑逐窗口 softmax，对比官方 attn_out（pre proj_out）
    # 官方 rope 输出布局: rope_vid_q (L_win, h, d), rope_txt_q (nwin*TXT_LEN, h, d)
    # attn_out (L_concat, h, d), L_concat = sum_i (f_i + TXT_LEN)
    rvq = np.transpose(cap['rope_vid_q'], (1, 0, 2))  # (h, L_win, d)
    rvk = np.transpose(cap['rope_vid_k'], (1, 0, 2))
    # v 不做 rope，用我们的 windowed vv_w（与官方一致：官方 vid_v 也未 rope）
    rvv_full = np.transpose(vv_w, (1, 0, 2))  # (h, L_win, d) 我们的 vid v（无 rope，正确）
    rtq = np.transpose(cap['rope_txt_q'], (1, 0, 2))  # (h, nwin*TXT, d)
    rtk = np.transpose(cap['rope_txt_k'], (1, 0, 2))
    rtv = np.transpose(tv_w, (1, 0, 2))  # (h, nwin*TXT, d) 我们的 txt v（无 rope）
    attn_out = np.transpose(cap['attn_out'], (1, 0, 2))  # (h, L_concat, d)

    HEADS = R.HEADS; HEAD_D = R.HEAD_D; scale = 1.0/math.sqrt(HEAD_D)
    vg = 0; off_vg = 0
    cos_seg_v = []; cos_seg_t = []
    for wi in range(nwin):
        f_i = win_shapes[wi][0]*win_shapes[wi][1]*win_shapes[wi][2]
        # 官方 attn_out 段: [off_vg : off_vg+f_i] vid, [off_vg+f_i : off_vg+f_i+TXT_LEN] txt
        o_v_seg = attn_out[:, off_vg:off_vg+f_i, :]            # (h, f_i, d)
        o_t_seg = attn_out[:, off_vg+f_i:off_vg+f_i+TXT_LEN, :] # (h, TXT, d)
        # 我们的 softmax（用官方 rope q/k + 我们的 v）
        q_seg = np.concatenate([rvq[:, vg:vg+f_i, :], rtq[:, wi*TXT_LEN:(wi+1)*TXT_LEN, :]], 1)
        k_seg = np.concatenate([rvk[:, vg:vg+f_i, :], rtk[:, wi*TXT_LEN:(wi+1)*TXT_LEN, :]], 1)
        v_seg = np.concatenate([rvv_full[:, vg:vg+f_i, :], rtv[:, wi*TXT_LEN:(wi+1)*TXT_LEN, :]], 1)
        o = np.empty((HEADS, f_i+TXT_LEN, HEAD_D), np.float32)
        for hh in range(HEADS):
            qq=q_seg[hh]; kk=k_seg[hh]; vv=v_seg[hh]
            sc=(qq@kk.T)*scale; sc=sc-sc.max(-1,keepdims=True)
            p=np.exp(sc); p=p/p.sum(-1,keepdims=True)
            o[hh]=p@vv
        cos_seg_v.append(cos(o[:, :f_i, :], o_v_seg))
        cos_seg_t.append(cos(o[:, f_i:, :], o_t_seg))
        vg += f_i; off_vg += (f_i + TXT_LEN)
    print(f"[softmax] 我的softmax(官方rope_qk + 我的v) vs 官方attn_out : vid_min={min(cos_seg_v):.6f} vid_mean={float(np.mean(cos_seg_v)):.6f}  txt_min={min(cos_seg_t):.6f} txt_mean={float(np.mean(cos_seg_t)):.6f}")
    print(f"[final] 官方 attn post-proj_out vid cos(ref) = {cos(attn_off_vid, attn_off_vid):.6f}")


def awa_window_make(grid, WINDOW):
    import awa_window as aw
    return aw.make_720Pwindows_bysize(grid, WINDOW)

def awa_window_make_shifted(grid, WINDOW):
    import awa_window as aw
    return aw.make_shifted_720Pwindows_bysize(grid, WINDOW)


if __name__ == "__main__":
    sys.exit(main())

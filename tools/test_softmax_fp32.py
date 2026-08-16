#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
隔离测试：用我们 numpy 算出的 block0 q/k/v（post proj+norm+rope，windowed），
送进 PyTorch fp32 SDPA（强制 fp32，无 autocast），对比我们的 numpy softmax。
≈1.0 -> 我们的注意力数学正确（0.998 纯属官方 bf16 噪声）。
"""
import sys, os, struct, math
import numpy as np
import torch

COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
sys.path.insert(0, COMFY)
from src.models.dit_3b import na
ROOT = "F:/Seedvr2/seedvr2-ncnn"
WORK = os.path.join(ROOT, "e2e_work")
sys.path.insert(0, os.path.join(ROOT, "export"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ncnn_io as io
import mmrope
import awa_window
import run_e2e_ref as R

DEVICE = "cuda:0"

def read_raw(path):
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = [struct.unpack("<q", fp.read(8))[0] for _ in range(ndim)]
        d = np.frombuffer(fp.read(int(np.prod(shape)) * 4), dtype=np.float32)
    return d.reshape(shape)

def cos(a, b):
    a = np.asarray(a).ravel().astype(np.float64); b = np.asarray(b).ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def main():
    vid_grid = read_raw(os.path.join(WORK, "vid_grid.bin"))
    T, H, W = vid_grid.shape[0], vid_grid.shape[1], vid_grid.shape[2]
    with io.open_sd() as f:
        Wpin = io.get_top_weight(f, "vid_in.proj", "weight"); Bpin = io.get_top_weight(f, "vid_in.proj", "bias")
        Wtxt = io.get_top_weight(f, "txt_in", "weight"); Btxt = io.get_top_weight(f, "txt_in", "bias")
    # 用 run_e2e_ref 的 patchify+proj 复现 vid_an 之前的输入
    x_patch = R.patchify(vid_grid, T, H, W)
    txt = read_raw(os.path.join(WORK, "txt.bin")).astype(np.float32)
    vid = x_patch @ Wpin.T + Bpin
    txt_t = txt @ Wtxt.T + Btxt
    emb = R.time_embedding(1000.0).astype(np.float32)
    emb3 = emb.reshape(1, R.HEAD_D * R.HEADS, 2, 3)
    Wb = R.block_weights(0)
    Adv = Wb["vid"]["ada"]; Adt = Wb["txt"]["ada"]
    vid_an = R.rmsnorm(vid); txt_an = R.rmsnorm(txt_t)
    sA, scA, gA = emb3[0, :, 0, 0], emb3[0, :, 0, 1], emb3[0, :, 0, 2]
    shB = Adv["attn_shift"]; scB = Adv["attn_scale"]; gB = Adv["attn_gate"]
    thB = Adt["attn_shift"]; tcB = Adt["attn_scale"]; tgB = Adt["attn_gate"]
    vid_an = vid_an * (scA + scB) + (sA + shB)
    txt_an = txt_an * (scA + tcB) + (sA + thB)

    grid = (T, H//2, W//2); Lv = vid_an.shape[0]; TXT_LEN = 58
    windows = awa_window.make_720Pwindows_bysize(grid, R.WINDOW)
    nwin = len(windows)
    win_shapes = [(st.stop-st.start, sh.stop-sh.start, sw.stop-sw.start) for (st,sh,sw) in windows]
    vid_freq, txt_freq = mmrope.build_window_freqs(win_shapes, TXT_LEN)
    vqkv = (vid_an @ Wb["vid"]["qkv"].T).reshape(Lv,3,R.HEADS,R.HEAD_D)
    tqkv = (txt_an @ Wb["txt"]["qkv"].T).reshape(TXT_LEN,3,R.HEADS,R.HEAD_D)
    vq, vk, vv = vqkv[:,0], vqkv[:,1], vqkv[:,2]
    tq, tk, tv = tqkv[:,0], tqkv[:,1], tqkv[:,2]
    vq = R.rmsnorm(vq, Wb["vid"]["nq"]); vk = R.rmsnorm(vk, Wb["vid"]["nk"])
    tq = R.rmsnorm(tq, Wb["txt"]["nq"]); tk = R.rmsnorm(tk, Wb["txt"]["nk"])
    vq_win = vq.reshape(T,H//2,W//2,R.HEADS,R.HEAD_D)
    vk_win = vk.reshape(T,H//2,W//2,R.HEADS,R.HEAD_D)
    vv_win = vv.reshape(T,H//2,W//2,R.HEADS,R.HEAD_D)
    vqw_l,vkw_l,vvw_l=[],[],[]
    for (st,sh,sw) in windows:
        vqw_l.append(vq_win[st,sh,sw].reshape(-1,R.HEADS,R.HEAD_D))
        vkw_l.append(vk_win[st,sh,sw].reshape(-1,R.HEADS,R.HEAD_D))
        vvw_l.append(vv_win[st,sh,sw].reshape(-1,R.HEADS,R.HEAD_D))
    vqw=np.concatenate(vqw_l,0); vkw=np.concatenate(vkw_l,0); vvw=np.concatenate(vvw_l,0)
    vq_w=np.transpose(vqw,(1,0,2)); vk_w=np.transpose(vkw,(1,0,2)); vv_w=np.transpose(vvw,(1,0,2))
    vq_w=mmrope.apply_rotary_emb(vid_freq,vq_w); vk_w=mmrope.apply_rotary_emb(vid_freq,vk_w)
    tq_rep=np.tile(tq,(nwin,1,1)); tk_rep=np.tile(tk,(nwin,1,1)); tv_rep=np.tile(tv,(nwin,1,1))
    tq_w=np.transpose(tq_rep,(1,0,2)); tk_w=np.transpose(tk_rep,(1,0,2)); tv_w=np.transpose(tv_rep,(1,0,2))
    tq_w=mmrope.apply_rotary_emb(txt_freq,tq_w); tk_w=mmrope.apply_rotary_emb(txt_freq,tk_w)

    scale = 1.0/math.sqrt(R.HEAD_D)
    cos_v_list=[]; cos_t_list=[]
    vg=0
    for wi in range(nwin):
        f_i = win_shapes[wi][0]*win_shapes[wi][1]*win_shapes[wi][2]
        # 我们的 numpy softmax（注意力）
        q = np.concatenate([vq_w[:,vg:vg+f_i,:], tq_w[:,wi*TXT_LEN:(wi+1)*TXT_LEN,:]],1)  # (h, f+txt, d)
        k = np.concatenate([vk_w[:,vg:vg+f_i,:], tk_w[:,wi*TXT_LEN:(wi+1)*TXT_LEN,:]],1)
        v = np.concatenate([vv_w[:,vg:vg+f_i,:], tv_w[:,wi*TXT_LEN:(wi+1)*TXT_LEN,:]],1)
        o = np.empty((R.HEADS, f_i+TXT_LEN, R.HEAD_D), np.float32)
        for hh in range(R.HEADS):
            qq=q[hh]; kk=k[hh]; vv=v[hh]
            sc=(qq@kk.T)*scale; sc=sc-sc.max(-1,keepdims=True)
            p=np.exp(sc); p=p/p.sum(-1,keepdims=True)
            o[hh]=p@vv
        our_v_seg = np.transpose(o[:,:f_i,:],(1,0,2))   # (f_i, h, d)
        our_t_seg = np.transpose(o[:,f_i:,:],(1,0,2))   # (txt, h, d)
        # PyTorch fp32 SDPA（强制 fp32）
        qt = torch.from_numpy(np.transpose(q,(1,0,2))).to(DEVICE).float()  # (f+txt, h, d)
        kt = torch.from_numpy(np.transpose(k,(1,0,2))).to(DEVICE).float()
        vt = torch.from_numpy(np.transpose(v,(1,0,2))).to(DEVICE).float()
        with torch.autocast(DEVICE, enabled=False):
            ot = torch.nn.functional.scaled_dot_product_attention(qt.permute(1,0,2), kt.permute(1,0,2), vt.permute(1,0,2))  # (1,h,L,d)
        ot = ot.squeeze(0).permute(1,0,2).detach().cpu().numpy().astype(np.float32)  # (L, h, d)
        torch_out_v = ot[:f_i]                        # (f_i, h, d) 与 our_v_seg 对齐
        torch_out_t = ot[f_i:]                        # (txt, h, d) 与 our_t_seg 对齐
        cos_v_list.append(cos(our_v_seg, torch_out_v))
        cos_t_list.append(cos(our_t_seg, torch_out_t))
        vg += f_i
    print(f"[softmax] 我们的 numpy softmax vs PyTorch fp32 SDPA:")
    print(f"  vid : min={min(cos_v_list):.6f} mean={float(np.mean(cos_v_list)):.6f}")
    print(f"  txt : min={min(cos_t_list):.6f} mean={float(np.mean(cos_t_list)):.6f}")

if __name__ == "__main__":
    main()

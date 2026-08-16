#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CPU 参考路径「逐子模块 cos 对拍」（官方窗口顺序对齐）：
对若干代表层，捕获官方 attn 模块四个阶段的张量，并用官方 window_op 的真实窗口顺序
重建我们的 q/k/v，逐级比较 cos(vid) / cos(txt)。

阶段：
  proj : attn_in @ Wqkv.T                 (权重 + head 布局)
  norm : 窗内 RMSNorm(q)                  (含 window 顺序 / norm 公式)
  rope : 窗内 RoPE 后 q                  (freq 生成 + 应用，核心嫌疑)
  attn : 完整 awa_forward 输出(post proj_out)  （SDPA 此前已验证=1.0）
"""
import sys, os, struct, math
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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

HEADS = R.HEADS
HEAD_D = R.HEAD_D
DIM = R.DIM
QKV = HEADS * HEAD_D * 3

TARGET = [0, 14, 31]
TXT_LEN = 58


def read_raw(path):
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = [struct.unpack("<q", fp.read(8))[0] for _ in range(ndim)]
        return np.frombuffer(fp.read(int(np.prod(shape)) * 4), dtype=np.float32).reshape(shape)


def cos(a, b):
    a = np.asarray(a).ravel().astype(np.float64); b = np.asarray(b).ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def window_official(v, slices):
    # v: (T,H,Wd, HEADS, HEAD_D)
    parts = []
    for (st, sh, sw) in slices:
        parts.append(v[st, sh, sw].reshape(-1, HEADS, HEAD_D))
    return np.concatenate(parts, axis=0)  # (L_win, HEADS, HEAD_D)


@torch.no_grad()
def main():
    vid_grid = read_raw(os.path.join(WORK, "vid_grid.bin"))
    T, H, W = vid_grid.shape[0], vid_grid.shape[1], vid_grid.shape[2]
    H2, W2 = H // 2, W // 2
    noise_t = torch.from_numpy(vid_grid[..., :16].copy()).to(DEVICE).float()
    cond17 = torch.from_numpy(vid_grid[..., 16:].copy()).to(DEVICE).float()
    vid_input = torch.cat([noise_t, cond17], dim=-1)
    vid_flat, vid_shape = na.flatten([vid_input])
    txt_raw = torch.from_numpy(read_raw(os.path.join(WORK, "txt.bin")).astype(np.float32)).to(DEVICE).float()

    ctx = setup_generation_context(dit_device=DEVICE, vae_device=DEVICE,
                                   dit_offload_device=None, vae_offload_device=None,
                                   tensor_offload_device=None, debug=Debug(enabled=False))
    ctx['compute_dtype'] = torch.float32
    runner, cache_context = prepare_runner(
        dit_model=DIT_MODEL, vae_model=DEFAULT_VAE, model_dir=MODEL_DIR, debug=Debug(enabled=False),
        ctx=ctx, dit_cache=False, vae_cache=False, dit_id=None, vae_id=None,
        block_swap_config={'blocks_to_swap': 0, 'swap_io_components': False, 'offload_device': 'none'},
        attention_mode='sdpa')
    ctx['cache_context'] = cache_context
    ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], torch.float32, Debug(enabled=False))
    from src.core.model_loader import materialize_model
    if next(runner.dit.parameters()).device.type == "meta":
        materialize_model(runner, "dit", DEVICE, runner.config, Debug(enabled=False))
    dit = runner.dit.to(DEVICE).half().eval()
    for blk in dit.blocks:
        blk.attn.attn.compute_dtype = None

    txt_flat, txt_shape = na.flatten([txt_raw])
    dit.vid_in.float(); dit.txt_in.float(); dit.emb_in.float()
    patched_vid, patched_vid_shape = dit.vid_in(vid_flat, vid_shape)
    txt_proj = dit.txt_in(txt_flat)
    tstept = torch.tensor([1000.0], device=DEVICE)
    emb_t = dit.emb_in(tstept, device=DEVICE, dtype=torch.float32)
    dit.vid_in.half(); dit.txt_in.half(); dit.emb_in.half()

    caps = {}
    def make_attnin_hook(li):
        def hook(module, inp, out):
            caps.setdefault(li, {})
            caps[li]["attn_in_vid"] = inp[0].detach().cpu().numpy().astype(np.float32)
            caps[li]["attn_in_txt"] = inp[1].detach().cpu().numpy().astype(np.float32)
        return hook
    def make_proj_hook(li):
        def hook(module, inp, out):
            caps.setdefault(li, {})
            caps[li]["proj_vid"] = out[0].detach().cpu().numpy().astype(np.float32)
            caps[li]["proj_txt"] = out[1].detach().cpu().numpy().astype(np.float32)
        return hook
    def make_norm_hook(li):
        def hook(module, inp, out):
            caps.setdefault(li, {})
            caps[li]["norm_vid"] = out[0].detach().cpu().numpy().astype(np.float32)
            caps[li]["norm_txt"] = out[1].detach().cpu().numpy().astype(np.float32)
        return hook
    def make_rope_hook(li):
        def hook(module, inp, out):
            caps.setdefault(li, {})
            vid_q, vid_k, txt_q, txt_k = out
            caps[li]["rope_vid_q"] = vid_q.detach().cpu().numpy().astype(np.float32)
            caps[li]["rope_vid_k"] = vid_k.detach().cpu().numpy().astype(np.float32)
            caps[li]["rope_txt_q"] = txt_q.detach().cpu().numpy().astype(np.float32)
            caps[li]["rope_txt_k"] = txt_k.detach().cpu().numpy().astype(np.float32)
        return hook
    def make_attn_hook(li):
        def hook(module, inp, out):
            caps.setdefault(li, {})
            caps[li]["attn_out_vid"] = out[0].detach().cpu().numpy().astype(np.float32)
            caps[li]["attn_out_txt"] = out[1].detach().cpu().numpy().astype(np.float32)
        return hook
    for li in TARGET:
        dit.blocks[li].attn.register_forward_hook(make_attnin_hook(li))
        dit.blocks[li].attn.proj_qkv.register_forward_hook(make_proj_hook(li))
        dit.blocks[li].attn.norm_q.register_forward_hook(make_norm_hook(li))
        dit.blocks[li].attn.rope.register_forward_hook(make_rope_hook(li))
        dit.blocks[li].attn.register_forward_hook(make_attn_hook(li))

    cache = Cache(disable=True)
    off_vid = patched_vid.float()
    off_txt = txt_proj.float()
    emb_t = emb_t.float()
    for i in range(R.NUM_LAYERS):
        blk = dit.blocks[i]
        blk.float()
        off_vid, off_txt, _, _ = dit.blocks[i](
            vid=off_vid, txt=off_txt, vid_shape=patched_vid_shape, txt_shape=txt_shape,
            emb=emb_t, cache=cache)
        blk.half()

    for li in TARGET:
        c = caps[li]
        window_method = "720pwin_by_size_bysize" if li % 2 == 0 else "720pswin_by_size_bysize"
        Wb = R.block_weights(li)
        blk = dit.blocks[li].attn
        slices = blk.window_op((T, H2, W2), blk.window)
        win_shapes = [(st.stop - st.start, sh.stop - sh.start, sw.stop - sw.start) for (st, sh, sw) in slices]
        vid_freq_ours, txt_freq_ours = mmrope.build_window_freqs(win_shapes, TXT_LEN)

        av = c["attn_in_vid"]; at = c["attn_in_txt"]   # (L, DIM) / (TXT, DIM)
        Lv = av.shape[0]

        # ---- proj ----
        our_vqkv = av @ Wb["vid"]["qkv"].T            # (L, qkv_dim)
        our_tqkv = at @ Wb["txt"]["qkv"].T
        cv_proj_v = cos(our_vqkv, c["proj_vid"])
        ct_proj_t = cos(our_tqkv, c["proj_txt"])

        # ---- norm (windowed) ----
        vq = our_vqkv.reshape(Lv, 3, HEADS, HEAD_D)[:, 0]
        vk = our_vqkv.reshape(Lv, 3, HEADS, HEAD_D)[:, 1]
        vv = our_vqkv.reshape(Lv, 3, HEADS, HEAD_D)[:, 2]
        vq_w = window_official(vq.reshape(T, H2, W2, HEADS, HEAD_D), slices)
        vk_w = window_official(vk.reshape(T, H2, W2, HEADS, HEAD_D), slices)
        vv_w = window_official(vv.reshape(T, H2, W2, HEADS, HEAD_D), slices)
        our_nv = R.rmsnorm(vq_w, Wb["vid"]["nq"])
        our_nk = R.rmsnorm(vk_w, Wb["vid"]["nk"])
        cv_norm_v = cos(our_nv, c["norm_vid"])

        # txt norm（官方 norm_q 作用在「未重复」txt，repeat 只在 RoPE 阶段做）
        tq = our_tqkv.reshape(TXT_LEN, 3, HEADS, HEAD_D)[:, 0]
        our_nt = R.rmsnorm(tq, Wb["txt"]["nq"])   # (58, HEADS, HEAD_D)
        cv_norm_t = cos(our_nt, c["norm_txt"])

        # ---- rope ----
        our_rv_q = mmrope.apply_rotary_emb(vid_freq_ours, our_nv.transpose(1, 0, 2)).transpose(1, 0, 2)
        our_rv_k = mmrope.apply_rotary_emb(vid_freq_ours, our_nk.transpose(1, 0, 2)).transpose(1, 0, 2)
        our_rt_q = mmrope.apply_rotary_emb(txt_freq_ours, np.tile(our_nt, (len(slices), 1, 1)).transpose(1, 0, 2)).transpose(1, 0, 2)
        cv_rope_v = cos(our_rv_q, c["rope_vid_q"])
        ct_rope_v = cos(our_rv_k, c["rope_vid_k"])
        cv_rope_t = cos(our_rt_q, c["rope_txt_q"])

        # ---- attn (full awa_forward) ----
        our_av, our_at = R.awa_forward(av, at, (T, H2, W2), window_method, Wb["vid"], Wb["txt"], TXT_LEN)
        cv_attn_v = cos(our_av, c["attn_out_vid"])
        cv_attn_t = cos(our_at, c["attn_out_txt"])

        print(f"=== layer {li} ({window_method})  nwin={len(slices)} ===")
        print(f"  proj : vid={cv_proj_v:.6f}  txt={ct_proj_t:.6f}")
        print(f"  norm : vid={cv_norm_v:.6f}  txt={cv_norm_t:.6f}")
        print(f"  rope : vid_q={cv_rope_v:.6f}  vid_k={ct_rope_v:.6f}  txt_q={cv_rope_t:.6f}")
        print(f"  attn : vid={cv_attn_v:.6f}  txt={cv_attn_t:.6f}")


if __name__ == "__main__":
    sys.exit(main())

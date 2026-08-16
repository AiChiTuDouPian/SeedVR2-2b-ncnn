#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
组件级对拍：对指定层集合，分别比较
  - attn 组件（post proj_out, pre ada-out-gate, pre residual）
  - mlp  组件（post mlp,       pre ada-out-gate, pre residual）
对比官方 dit.blocks[i] 的 attn / mlp 子模块 hook 输出，与我们的 awa_forward / swiglu。
用于定位 layer12+ 偏差到底来自注意力还是 MLP/ada。
"""
import sys, os, struct
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
import awa_window
import mmrope
import run_e2e_ref as R


def read_raw(path):
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = [struct.unpack("<q", fp.read(8))[0] for _ in range(ndim)]
        d = np.frombuffer(fp.read(int(np.prod(shape)) * 4), dtype=np.float32).reshape(shape)
    return d


def cos(a, b):
    a = np.asarray(a).ravel().astype(np.float64); b = np.asarray(b).ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@torch.no_grad()
def main():
    layers = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else "8,9,10,11,12,13,14,15")]
    vid_grid = read_raw(os.path.join(WORK, "vid_grid.bin"))
    T, H, W = vid_grid.shape[0], vid_grid.shape[1], vid_grid.shape[2]
    noise_t = torch.from_numpy(vid_grid[..., :16].copy()).to(DEVICE).half()
    cond17 = torch.from_numpy(vid_grid[..., 16:].copy()).to(DEVICE).half()
    vid_input = torch.cat([noise_t, cond17], dim=-1)
    vid_flat, vid_shape = na.flatten([vid_input])
    txt_raw = torch.from_numpy(read_raw(os.path.join(WORK, "txt.bin")).astype(np.float32)).to(DEVICE).half()

    ctx = setup_generation_context(dit_device=DEVICE, vae_device=DEVICE,
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
    dit = runner.dit.to(DEVICE).half().eval()

    txt_flat, txt_shape = na.flatten([txt_raw])
    txt_proj = dit.txt_in(txt_flat)
    tstept = torch.tensor([1000.0], device=DEVICE)
    emb_t = dit.emb_in(tstept, device=DEVICE, dtype=torch.half)
    patched_vid, patched_vid_shape = dit.vid_in(vid_flat, vid_shape)

    our_vid = patched_vid.cpu().numpy().astype(np.float32)
    our_txt = txt_proj.cpu().numpy().astype(np.float32)
    emb_np = R.time_embedding(1000.0).astype(np.float32)
    emb3 = emb_np.reshape(1, R.HEAD_D * R.HEADS, 2, 3)

    cache = Cache(disable=True)
    off_vid = patched_vid
    off_txt = txt_proj
    print(f"[cmp] 对比层: {layers}")
    print(f"[cmp] {'layer':>5} | {'blk':>7} | {'attn':>7} | {'mlp':>7}")
    for i in range(R.NUM_LAYERS):
        if i in layers:
            cap = {}
            ha = dit.blocks[i].attn.register_forward_hook(
                lambda m, inp, out: cap.update({'attn_vid': out[0].detach().clone(), 'attn_txt': out[1].detach().clone()}))
            hm = dit.blocks[i].mlp.register_forward_hook(
                lambda m, inp, out: cap.update({'mlp_vid': out[0].detach().clone(), 'mlp_txt': out[1].detach().clone()}))
        off_vid, off_txt, _, _ = dit.blocks[i](
            vid=off_vid, txt=off_txt, vid_shape=patched_vid_shape, txt_shape=txt_shape,
            emb=emb_t, cache=cache)
        if i in layers:
            ha.remove(); hm.remove()

        window_method = "720pwin_by_size_bysize" if i % 2 == 0 else "720pswin_by_size_bysize"
        Wb = R.block_weights(i)
        Adv = Wb["vid"]["ada"]; Adt = Wb["txt"]["ada"]
        vid_an = R.rmsnorm(our_vid); txt_an = R.rmsnorm(our_txt)
        sA, scA, gA = emb3[0, :, 0, 0], emb3[0, :, 0, 1], emb3[0, :, 0, 2]
        shB = Adv["attn_shift"]; scB = Adv["attn_scale"]; gB = Adv["attn_gate"]
        thB = Adt["attn_shift"]; tcB = Adt["attn_scale"]; tgB = Adt["attn_gate"]
        vid_an = vid_an * (scA + scB) + (sA + shB)
        txt_an = txt_an * (scA + tcB) + (sA + thB)
        vid_at, txt_at = R.awa_forward(vid_an, txt_an, (T, H // 2, W // 2), window_method, Wb["vid"], Wb["txt"], R.TXT_LEN if hasattr(R, "TXT_LEN") else 58)
        # 记录 attn 组件（pre gate, pre residual）
        attn_cmp_vid = vid_at
        vid_at = vid_at * (gA + gB); txt_at = txt_at * (gA + tgB)
        vid = vid_at + our_vid; txt_t = txt_at + our_txt
        vid_mn = R.rmsnorm(vid); txt_mn = R.rmsnorm(txt_t)
        sAm, scAm, gAm = emb3[0, :, 1, 0], emb3[0, :, 1, 1], emb3[0, :, 1, 2]
        shBm = Adv["mlp_shift"]; scBm = Adv["mlp_scale"]; gBm = Adv["mlp_gate"]
        thBm = Adt["mlp_shift"]; tcBm = Adt["mlp_scale"]; tgBm = Adt["mlp_gate"]
        vid_mn = vid_mn * (scAm + scBm) + (sAm + shBm)
        txt_mn = txt_mn * (scAm + tcBm) + (sAm + thBm)
        vid_m = R.swiglu(vid_mn, Wb["vid"]["mlp_in"], Wb["vid"]["mlp_g"], Wb["vid"]["mlp_out"])
        txt_m = R.swiglu(txt_mn, Wb["txt"]["mlp_in"], Wb["txt"]["mlp_g"], Wb["txt"]["mlp_out"])
        # 记录 mlp 组件（pre gate, pre residual）
        mlp_cmp_vid = vid_m
        vid_m = vid_m * (gAm + gBm); txt_m = txt_m * (gAm + tgBm)
        our_vid = (vid_m + vid).astype(np.float32); our_txt = (txt_m + txt_t).astype(np.float32)

        if i in layers:
            c_attn = cos(attn_cmp_vid, cap['attn_vid'].cpu().numpy())
            c_mlp = cos(mlp_cmp_vid, cap['mlp_vid'].cpu().numpy())
            c_blk = cos(our_vid, off_vid.cpu().numpy())
            print(f"[cmp] {i:5d} | {c_blk:.5f} | {c_attn:.5f} | {c_mlp:.5f}")
    print("[cmp] 完成组件对拍")


if __name__ == "__main__":
    sys.exit(main())

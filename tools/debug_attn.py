#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
隔离注意力：hook 官方 NaSwinAttention.forward 的返回（post proj_out，pre ada-out gate），
与我们的 awa_forward 输出对比。输入用 block 0 之前已对齐的 vid_an（attn_norm+ada-in 后）。
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

    # hook block 0 的 attention（NaSwinAttention）返回
    captured = {}
    def hook(module, inp, out):
        captured['vid'] = out[0].detach().clone()
        captured['txt'] = out[1].detach().clone()
    h = dit.blocks[0].attn.register_forward_hook(hook)

    cache = Cache(disable=True)
    off_vid, off_txt, _, _ = dit.blocks[0](
        vid=patched_vid, txt=txt_proj, vid_shape=patched_vid_shape, txt_shape=txt_shape,
        emb=emb_t, cache=cache)
    h.remove()
    attn_off_vid = captured['vid']   # (L_patch, 2560) post proj_out, pre ada-out, pre residual

    # 我们的 awa_forward（block 0）：需要 attn_norm+ada-in 后的 vid_an / txt_an
    # 复现 block 0 之前的输入（用 patched_vid 作为原始 vid，txt_proj 作为 txt）
    with io.open_sd() as f:
        Wpin = io.get_top_weight(f, "vid_in.proj", "weight"); Bpin = io.get_top_weight(f, "vid_in.proj", "bias")
        Wtxt = io.get_top_weight(f, "txt_in", "weight"); Btxt = io.get_top_weight(f, "txt_in", "bias")
    our_vid = patched_vid.cpu().numpy().astype(np.float32)
    our_txt = txt_proj.cpu().numpy().astype(np.float32)
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
    vid_at, txt_at = R.awa_forward(vid_an, txt_an, (T, H // 2, W // 2), window_method, Wb["vid"], Wb["txt"], 58)

    c_vid = cos(vid_at, attn_off_vid.cpu().numpy())
    c_txt = cos(txt_at, captured['txt'].cpu().numpy())
    print(f"[attn] cos(our awa_forward vid, official attn vid) = {c_vid:.6f}")
    print(f"[attn] cos(our awa_forward txt, official attn txt) = {c_txt:.6f}")

    # 进一步：分别比较 q/k/v 生成 + rope 之前/之后，定位
    # 复现我们的 q/k/v + norm + rope，对比官方 attn 内部（需 hook 更细，先报总差异）
    if c_vid < 0.99:
        print("[attn] 注意力输出不一致 -> 需进一步细分 qkv/norm/rope/attn")
    else:
        print("[attn] 注意力输出一致 -> 偏差在 ada/mlp/残差（已核对公式）")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比 checkerboard 强度：官方 bf16 DiT 输出 vs 我们的 fp32 输出。
官方 pipeline 硬编码 compute_dtype=bf16，这里用官方 dit.forward（默认 bf16）跑，
输出 unpatchify 后的 (T,H,W,16)，测 2x2 patch 4 位置的 checkerboard 偏差。
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

DEVICE = "cuda:0"
MODEL_DIR = os.path.join(COMFY, "models", "SEEDVR2")
DIT_MODEL = "seedvr2_ema_3b_fp16.safetensors"
ROOT = "F:/Seedvr2/seedvr2-ncnn"
WORK = os.path.join(ROOT, "e2e_work_sadhu")


def read_raw(path):
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = [struct.unpack("<q", fp.read(8))[0] for _ in range(ndim)]
        return np.frombuffer(fp.read(int(np.prod(shape)) * 4), dtype=np.float32).reshape(shape)


def checkerboard(sr):
    # sr: (1,H,W,16)，测 2x2 patch 4 位置的 checkerboard 偏差
    ch = sr[0, ..., 0]  # (H,W)
    p00 = ch[0::2, 0::2].mean()
    p01 = ch[0::2, 1::2].mean()
    p10 = ch[1::2, 0::2].mean()
    p11 = ch[1::2, 1::2].mean()
    # checkerboard = (p00+p11) - (p01+p10)，对角差
    cb = ((p00 + p11) - (p01 + p10)) / 2
    return cb, [p00, p01, p10, p11]


@torch.no_grad()
def main():
    vid_grid = read_raw(os.path.join(WORK, "vid_grid.bin"))
    T, H, W = vid_grid.shape[0], vid_grid.shape[1], vid_grid.shape[2]
    noise_t = torch.from_numpy(vid_grid[..., :16].copy()).to(DEVICE).half()
    cond17 = torch.from_numpy(vid_grid[..., 16:].copy()).to(DEVICE).half()
    vid_input = torch.cat([noise_t, cond17], dim=-1)
    vid_flat, vid_shape = na.flatten([vid_input])
    txt_raw = torch.from_numpy(read_raw(os.path.join(WORK, "txt.bin")).astype(np.float32)).to(DEVICE).half()

    # 官方 bf16 pipeline：不 override compute_dtype（默认 bf16）
    ctx = setup_generation_context(dit_device=DEVICE, vae_device=DEVICE,
                                   dit_offload_device=None, vae_offload_device=None,
                                   tensor_offload_device=None, debug=Debug(enabled=False))
    runner, cache_context = prepare_runner(
        dit_model=DIT_MODEL, vae_model=DEFAULT_VAE, model_dir=MODEL_DIR, debug=Debug(enabled=False),
        ctx=ctx, dit_cache=False, vae_cache=False, dit_id=None, vae_id=None,
        block_swap_config={'blocks_to_swap': 0, 'swap_io_components': False, 'offload_device': 'none'},
        attention_mode='sdpa')
    ctx['cache_context'] = cache_context
    ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], torch.float16, Debug(enabled=False))
    from src.core.model_loader import materialize_model
    if next(runner.dit.parameters()).device.type == "meta":
        materialize_model(runner, "dit", DEVICE, runner.config, Debug(enabled=False))
    dit = runner.dit.to(DEVICE).half().eval()

    txt_flat, txt_shape = na.flatten([txt_raw])
    from src.common.cache import Cache
    cache = Cache(disable=True)
    timestep = torch.tensor([1000.0], device=DEVICE)
    out = dit(vid_flat, txt_flat, vid_shape, txt_shape, timestep, cache)
    sr_bf16 = out.vid_sample  # (T,H,W,16) 已 unpatchify，bf16
    sr_bf16 = sr_bf16.detach().float().cpu().numpy()

    # 我们的 fp32 sr_latent（C++ 输出，已 unpatchify）
    sr_ours = read_raw(os.path.join(WORK, "sr_latent.bin"))

    cb_bf, p_bf = checkerboard(sr_bf16)
    cb_our, p_our = checkerboard(sr_ours)
    print(f"=== checkerboard 强度对比 (Sadhu) ===")
    print(f"  官方 bf16: 4位置均值={['%.3f'%x for x in p_bf]}  对角差(checkerboard)={cb_bf:.4f}")
    print(f"  我们 fp32: 4位置均值={['%.3f'%x for x in p_our]}  对角差(checkerboard)={cb_our:.4f}")
    print(f"  checkerboard 比值(fp32/bf16) = {abs(cb_our)/max(abs(cb_bf),1e-6):.2f}x")


if __name__ == "__main__":
    sys.exit(main())

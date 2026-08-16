#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
官方 SeedVR2 推理 + 中间结果导出。

在 run_official.py 的基础上，额外保存：
  - cond.bin   : VAE encode + scaling 后的条件 latent (T,H,W,16)
  - noise.bin  : Phase2 中 torch.randn_like(latent) 生成的噪声 (T,H,W,16)
  - x0.bin     : DiT 单步去噪后的 upscaled latent (T,H,W,16)
  - raw.png    : Phase4 之前的解码图（与 run_official.py 的 raw 一致）

用于和 ncnn 的 vid_grid.bin / sr_latent.bin / upscaled.bin 逐阶段对拍。
"""
import sys, os, struct, argparse
import numpy as np
import torch
from PIL import Image

COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
sys.path.insert(0, COMFY)

from src.utils.debug import Debug
from src.core.generation_utils import (
    setup_generation_context, prepare_runner, compute_generation_info,
    load_text_embeddings, script_directory,
)
from src.core.generation_phases import (
    encode_all_batches, upscale_all_batches, decode_all_batches, postprocess_all_batches,
)
from src.utils.model_registry import DEFAULT_VAE

DEVICE = "cuda:0"
MODEL_DIR = os.path.join(COMFY, "models", "SEEDVR2")
DIT_MODEL = "seedvr2_ema_3b_fp16.safetensors"


def load_image_thwc(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr[None, ...]).to(torch.float16)


def write_raw(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<q", arr.ndim))
        for d in arr.shape:
            fp.write(struct.pack("<q", int(d)))
        fp.write(arr.tobytes())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lr_image")
    ap.add_argument("out_dir")
    ap.add_argument("--resolution", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    debug = Debug(enabled=False)

    ctx = setup_generation_context(
        dit_device=DEVICE, vae_device=DEVICE,
        dit_offload_device=None, vae_offload_device=None,
        tensor_offload_device=None, debug=debug,
    )
    runner, cache_context = prepare_runner(
        dit_model=DIT_MODEL, vae_model=DEFAULT_VAE, model_dir=MODEL_DIR,
        debug=debug, ctx=ctx,
        dit_cache=False, vae_cache=False, dit_id=None, vae_id=None,
        block_swap_config={'blocks_to_swap': 0, 'swap_io_components': False, 'offload_device': 'none'},
        attention_mode='sdpa',
    )
    ctx['cache_context'] = cache_context
    ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], ctx['compute_dtype'], debug)

    images = load_image_thwc(args.lr_image)
    print(f"[official] 输入图 {tuple(images.shape)}")

    images, gen_info = compute_generation_info(
        ctx=ctx, images=images, resolution=args.resolution, max_resolution=0,
        batch_size=5, uniform_batch_size=False, seed=args.seed,
        prepend_frames=0, temporal_overlap=0, debug=debug,
    )
    print(f"[official] true_target_dims={ctx.get('true_target_dims')}")

    # ---- Phase 1: encode ----
    ctx = encode_all_batches(runner, ctx=ctx, images=images, debug=debug,
                             batch_size=5, uniform_batch_size=False, seed=args.seed,
                             resolution=args.resolution, max_resolution=0,
                             input_noise_scale=0.0, color_correction="lab")

    cond = ctx['all_latents'][0]  # (T,H,W,16) bf16
    print(f"[official] cond shape={tuple(cond.shape)} dtype={cond.dtype}")
    write_raw(os.path.join(args.out_dir, "cond.bin"), cond.cpu().float().numpy())

    # ---- Phase 2: upscale (保存 noise 和 x0) ----
    # 复现 upscale_all_batches 的 set_seed + randn_like，保证 noise 与官方内部一致
    torch.manual_seed(args.seed)
    noise = torch.randn_like(cond, dtype=ctx['compute_dtype'])
    print(f"[official] noise shape={tuple(noise.shape)} dtype={noise.dtype}")
    write_raw(os.path.join(args.out_dir, "noise.bin"), noise.cpu().float().numpy())

    ctx = upscale_all_batches(runner, ctx=ctx, debug=debug, progress_callback=None,
                              seed=args.seed, latent_noise_scale=0.0, cache_model=False)

    x0 = ctx['all_upscaled_latents'][0]  # (T,H,W,16) bf16
    print(f"[official] x0 shape={tuple(x0.shape)} dtype={x0.dtype}")
    write_raw(os.path.join(args.out_dir, "x0.bin"), x0.cpu().float().numpy())

    # ---- Phase 3: decode ----
    ctx = decode_all_batches(runner, ctx=ctx, debug=debug, progress_callback=None, cache_model=False)

    final = ctx['final_video'].cpu().float()
    true_h, true_w = ctx.get('true_target_dims', (final.shape[1], final.shape[2]))
    raw = final[0, :true_h, :true_w, :3].clamp(-1, 1).mul(0.5).add_(0.5).clamp(0, 1)
    raw_np = (raw.numpy() * 255.0).astype(np.uint8)
    Image.fromarray(raw_np).save(os.path.join(args.out_dir, "raw.png"))
    print(f"[official] raw 已保存: shape={raw_np.shape[:2]}")


if __name__ == "__main__":
    main()

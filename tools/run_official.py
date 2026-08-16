#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
官方 SeedVR2 推理对比驱动（直接复用 ComfyUI-SeedVR2 的 4 阶段管线，本地 fp16 权重，不触发下载）。

目的：用「官方 Python 推理」同一张图，产出两份产物用于和 ncnn 输出对比：
  - <out_raw>  : 跳过 Phase4 色彩校正，纯扩散解码结果（raw，[-1,1]->[0,1]）
  - <out_lab>  : 官方默认 lab 色彩校正后的最终产物（用户实际看到的结果）

用法（cuda_env）:
  python tools/run_official.py <lr_image> <out_raw> <out_lab> [--resolution 1080] [--seed 42]
"""
import sys, os, argparse
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
DIT_MODEL = "seedvr2_ema_3b_fp16.safetensors"   # 本地 fp16（默认 DEFAULT_DIT 是 fp8，这里显式用 fp16）


def load_image_thwc(path):
    """复现 inference_cli.extract_frames_from_image：RGB, [0,1], [1,H,W,3], float16。"""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr[None, ...]).to(torch.float16)   # [1,H,W,3]
    return t


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lr_image")
    ap.add_argument("out_raw")
    ap.add_argument("out_lab")
    ap.add_argument("--resolution", type=int, default=1080)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    debug = Debug(enabled=False)

    # ---- 上下文 + 模型（本地 fp16，不下载） ----
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

    # ---- 载入图（与官方一致） ----
    images = load_image_thwc(args.lr_image)
    print(f"[official] 输入图 {tuple(images.shape)}")

    images, gen_info = compute_generation_info(
        ctx=ctx, images=images, resolution=args.resolution, max_resolution=0,
        batch_size=5, uniform_batch_size=False, seed=args.seed,
        prepend_frames=0, temporal_overlap=0, debug=debug,
    )
    print(f"[official] true_target_dims={ctx.get('true_target_dims')}")

    # ---- Phase 1-3：encode / upscale / decode（color_correction='lab' 仅用于存 batch_metadata） ----
    ctx = encode_all_batches(runner, ctx=ctx, images=images, debug=debug,
                             batch_size=5, uniform_batch_size=False, seed=args.seed,
                             resolution=args.resolution, max_resolution=0,
                             input_noise_scale=0.0, color_correction="lab")
    ctx = upscale_all_batches(runner, ctx=ctx, debug=debug, progress_callback=None,
                              seed=args.seed, latent_noise_scale=0.0, cache_model=False)
    ctx = decode_all_batches(runner, ctx=ctx, debug=debug, progress_callback=None, cache_model=False)

    # ---- 保存 raw（Phase4 之前的 final_video，[-1,1]） ----
    final = ctx['final_video']  # [T,H,W,C] [-1,1]（compute_dtype=bf16）
    if final.is_cuda:
        final = final.cpu()
    final = final.float()   # bf16 不能直接 numpy
    true_h, true_w = ctx.get('true_target_dims', (final.shape[1], final.shape[2]))
    raw = final[0, :true_h, :true_w, :3].clamp(-1, 1).mul(0.5).add_(0.5).clamp(0, 1)
    raw_np = (raw.numpy() * 255.0).astype(np.uint8)
    Image.fromarray(raw_np).save(args.out_raw)
    print(f"[official] raw 已保存: {args.out_raw}  shape={raw_np.shape[:2]}")

    # ---- Phase 4：lab 色彩校正（覆盖写回 final_video） ----
    ctx = postprocess_all_batches(ctx, debug=debug, progress_callback=None,
                                  color_correction="lab", prepend_frames=0,
                                  temporal_overlap=0, batch_size=5)
    final2 = ctx['final_video']
    if final2.is_cuda:
        final2 = final2.cpu()
    final2 = final2.float()
    lab = final2[0, :true_h, :true_w, :3].clamp(0, 1)
    lab_np = (lab.numpy() * 255.0).astype(np.uint8)
    Image.fromarray(lab_np).save(args.out_lab)
    print(f"[official] lab 已保存: {args.out_lab}  shape={lab_np.shape[:2]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
fp32 对照：把官方 DiT 直接加载为 float32（_dit_dtype_override），跑同一份输入，
与我们的 fp32 numpy 参考 ref_grid（velocity）对比。

目的：分离「fp16 精度漂移」与「真实算法 bug」。
  - 若 cos(ref_grid, v_off_fp32) ~ 0.999  -> 0.94 的差距来自 fp16 漂移，算法正确。
  - 若仍 ~0.94           -> 存在真实算法 bug，需继续定位。
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


@torch.no_grad()
def main():
    vid_grid = read_raw(os.path.join(WORK, "vid_grid.bin"))
    T, H, W = vid_grid.shape[0], vid_grid.shape[1], vid_grid.shape[2]

    noise_t = torch.from_numpy(vid_grid[..., :16].copy()).to(DEVICE).float()
    cond17 = torch.from_numpy(vid_grid[..., 16:].copy()).to(DEVICE).float()
    vid_input = torch.cat([noise_t, cond17], dim=-1)
    vid_flat, vid_shape = na.flatten([vid_input])

    ctx = setup_generation_context(dit_device=DEVICE, vae_device=DEVICE,
                                   dit_offload_device=None, vae_offload_device=None,
                                   tensor_offload_device=None, debug=Debug(enabled=False))
    runner, cache_context = prepare_runner(
        dit_model=DIT_MODEL, vae_model=DEFAULT_VAE, model_dir=MODEL_DIR, debug=Debug(enabled=False),
        ctx=ctx, dit_cache=False, vae_cache=False, dit_id=None, vae_id=None,
        block_swap_config={'blocks_to_swap': 0, 'swap_io_components': False, 'offload_device': 'none'},
        attention_mode='sdpa')
    ctx['cache_context'] = cache_context
    ctx['text_embeds'] = load_text_embeddings(script_directory, ctx['dit_device'], torch.float32, Debug(enabled=False))
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(device=DEVICE, dtype=torch.float32)

    # 内存安全 fp32 路径：先正常 fp16 上 CUDA（~6GB），再挪到 CPU 转 fp32，最后回 CUDA。
    # 避免「fp16(6G)+fp32(12G)+模型(12G)」~30GB 峰值 OOM。
    from src.core.model_loader import materialize_model
    if next(runner.dit.parameters()).device.type == "meta":
        materialize_model(runner, "dit", DEVICE, runner.config, Debug(enabled=False))
    torch.cuda.empty_cache()
    runner.dit = runner.dit.cpu().float().to(DEVICE).eval()

    text_pos = ctx['text_embeds']['texts_pos'][0].to(DEVICE).float()
    txt_flat, txt_shape = na.flatten([text_pos])
    txt_flat = txt_flat.to(DEVICE).float()

    dit = (runner.dit.dit_model if hasattr(runner.dit, "dit_model") else runner.dit).eval()

    v_off = dit(vid=vid_flat, txt=txt_flat, vid_shape=vid_shape, txt_shape=txt_shape,
                timestep=torch.tensor([1000.0], device=DEVICE)).vid_sample
    v_off_grid = na.unflatten(v_off, vid_shape)[0].float().cpu().numpy()
    print(f"[fp32] v_off {v_off_grid.shape}")

    # 我们的 numpy 参考 ref_grid（velocity），保持 fp32
    txt = read_raw(os.path.join(WORK, "txt.bin")).astype(np.float32)
    with open(os.path.join(WORK, "params.txt")) as fp:
        _, _, _, TXT_LEN, ts = fp.read().split(); TXT_LEN = int(TXT_LEN); ts = float(ts)
    H2, W2 = H // 2, W // 2
    x_patch = R.patchify(vid_grid, T, H, W)
    x_txt = txt.astype(np.float32)
    with io.open_sd() as f:
        Wpin = io.get_top_weight(f, "vid_in.proj", "weight"); Bpin = io.get_top_weight(f, "vid_in.proj", "bias")
        Wtxt = io.get_top_weight(f, "txt_in", "weight"); Btxt = io.get_top_weight(f, "txt_in", "bias")
        Wvon = io.get_top_weight(f, "vid_out_norm", "weight")
        Wvoa_s = io.load_key(f, "vid_out_ada.out_shift"); Wvoa_sc = io.load_key(f, "vid_out_ada.out_scale")
        Wvout = io.get_top_weight(f, "vid_out.proj", "weight"); Bvout = io.get_top_weight(f, "vid_out.proj", "bias")
    vid = x_patch @ Wpin.T + Bpin
    txt_t = x_txt @ Wtxt.T + Btxt
    emb = R.time_embedding(ts)
    emb3 = emb.reshape(1, R.HEAD_D * R.HEADS, 2, 3)
    for i in range(R.NUM_LAYERS):
        window_method = "720pwin_by_size_bysize" if i % 2 == 0 else "720pswin_by_size_bysize"
        Wb = R.block_weights(i)
        Adv = Wb["vid"]["ada"]; Adt = Wb["txt"]["ada"]
        vid_an = R.rmsnorm(vid); txt_an = R.rmsnorm(txt_t)
        sA, scA, gA = emb3[0, :, 0, 0], emb3[0, :, 0, 1], emb3[0, :, 0, 2]
        shB = Adv["attn_shift"]; scB = Adv["attn_scale"]; gB = Adv["attn_gate"]
        thB = Adt["attn_shift"]; tcB = Adt["attn_scale"]; tgB = Adt["attn_gate"]
        vid_an = vid_an * (scA + scB) + (sA + shB)
        txt_an = txt_an * (scA + tcB) + (sA + thB)
        vid_at, txt_at = R.awa_forward(vid_an, txt_an, (T, H2, W2), window_method, Wb["vid"], Wb["txt"], TXT_LEN)
        vid_at = vid_at * (gA + gB); txt_at = txt_at * (gA + tgB)
        vid = vid_at + vid; txt_t = txt_at + txt_t
        vid_mn = R.rmsnorm(vid); txt_mn = R.rmsnorm(txt_t)
        sAm, scAm, gAm = emb3[0, :, 1, 0], emb3[0, :, 1, 1], emb3[0, :, 1, 2]
        shBm = Adv["mlp_shift"]; scBm = Adv["mlp_scale"]; gBm = Adv["mlp_gate"]
        thBm = Adt["mlp_shift"]; tcBm = Adt["mlp_scale"]; tgBm = Adt["mlp_gate"]
        vid_mn = vid_mn * (scAm + scBm) + (sAm + shBm)
        txt_mn = txt_mn * (scAm + tcBm) + (sAm + thBm)
        vid_m = R.swiglu(vid_mn, Wb["vid"]["mlp_in"], Wb["vid"]["mlp_g"], Wb["vid"]["mlp_out"])
        txt_m = R.swiglu(txt_mn, Wb["txt"]["mlp_in"], Wb["txt"]["mlp_g"], Wb["txt"]["mlp_out"])
        vid_m = vid_m * (gAm + gBm); txt_m = txt_m * (gAm + tgBm)
        vid = vid_m + vid; txt_t = txt_m + txt_t
    vid = R.rmsnorm(vid, Wvon)
    emb_out3 = emb[:, :R.HEAD_D * R.HEADS * 3].reshape(1, R.HEAD_D * R.HEADS, 1, 3)
    sAo = emb_out3[0, :, 0, 0]; scAo = emb_out3[0, :, 0, 1]
    vid = vid * (scAo + Wvoa_sc) + (sAo + Wvoa_s)
    ref = vid @ Wvout.T + Bvout
    ref_grid = R.unpatchify(ref, T, H, W)

    def cos(a, b):
        a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    c_fwd = cos(ref_grid, v_off_grid)
    print(f"\n[fp32] cos(ref_grid, v_off_fp32) = {c_fwd:.6f}   (fp16 官方为 0.942760)")
    if c_fwd >= 0.99:
        print("[fp32] => 0.94 差距来自 fp16 漂移，numpy 参考算法正确")
    else:
        print("[fp32] => 仍存在真实算法 bug，需继续定位")
    return 0 if c_fwd >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())

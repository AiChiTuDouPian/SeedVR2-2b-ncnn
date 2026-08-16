#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接调用官方 DiT（与 run_official 同款 runner/model），用「我们 e2e_work/vid_grid.bin」的
同一份输入（noise+cond+mask）跑前向，dump：
  - v_off   : 官方网络原始输出（velocity，patch token 已 unflatten 回像素空间）
  - x0_off  : 官方一步采样后的 upscaled latent
并与 numpy 参考（run_e2e_ref 同款前向）的 ref_grid 对拍：
  - cos(ref_grid, v_off)      -> DiT 前向保真度
  - 经验反推 x0 = f(ref_grid)  -> 与 x0_off 比较，确定 velocity->x0 公式
用途：定位 DiT 前向是否存在 bug（RoPE 或其他）。
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
    vid_grid = read_raw(os.path.join(WORK, "vid_grid.bin"))   # (1,H,W,33)
    T, H, W = vid_grid.shape[0], vid_grid.shape[1], vid_grid.shape[2]
    print(f"[cmp] vid_grid {vid_grid.shape}")

    noise_t = torch.from_numpy(vid_grid[..., :16].copy()).to(DEVICE).half()
    cond17 = torch.from_numpy(vid_grid[..., 16:].copy()).to(DEVICE).half()  # [cond16, mask1]
    vid_input = torch.cat([noise_t, cond17], dim=-1)            # (1,H,W,33) 同官方 cat([x_t, latents_cond])
    vid_flat, vid_shape = na.flatten([vid_input])
    print(f"[cmp] vid_flat {tuple(vid_flat.shape)} vid_shape {vid_shape.tolist()}")

    # ---- 官方 runner + 文本 ----
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
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion(device=DEVICE, dtype=ctx['compute_dtype'])

    from src.core.model_loader import materialize_model
    if next(runner.dit.parameters()).device.type == "meta":
        materialize_model(runner, "dit", DEVICE, runner.config, Debug(enabled=False))
    runner.dit = runner.dit.to(DEVICE).half()

    text_pos = ctx['text_embeds']['texts_pos'][0]               # (58,5120)
    txt_flat, txt_shape = na.flatten([text_pos])
    txt_flat = txt_flat.to(DEVICE).half()
    print(f"[cmp] txt_flat {tuple(txt_flat.shape)} txt_shape {txt_shape.tolist()}")

    dit = (runner.dit.dit_model if hasattr(runner.dit, "dit_model") else runner.dit).eval()

    # ---- 官方 velocity ----
    v_off = dit(vid=vid_flat, txt=txt_flat, vid_shape=vid_shape, txt_shape=txt_shape,
                timestep=torch.tensor([1000.0], device=DEVICE)).vid_sample
    v_off_grid = na.unflatten(v_off, vid_shape)[0].float().cpu().numpy()   # (1,H,W,16)
    print(f"[cmp] v_off {v_off_grid.shape}")

    # ---- 官方 x0（一步采样） ----
    x0_off = runner.inference(noises=[noise_t], conditions=[cond17],
                              texts_pos=[text_pos.half()],
                              texts_neg=[ctx['text_embeds']['texts_neg'][0].half()],
                              cfg_scale=1.0)
    x0_off_np = x0_off[0].float().cpu().numpy()
    print(f"[cmp] x0_off {x0_off_np.shape}")

    # ---- 我们的 numpy 参考 ref_grid（velocity） ----
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
    ref_grid = R.unpatchify(ref, T, H, W)                       # (1,H,W,16) velocity

    def cos(a, b):
        a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    noise_np = vid_grid[..., :16].astype(np.float32)
    c_fwd = cos(ref_grid, v_off_grid)
    print(f"\n[cmp] cos(ref_grid, v_off)  = {c_fwd:.6f}   <-- DiT 前向保真度")

    # 经验反推 velocity->x0 公式
    cands = {
        "x0 = noise - v": noise_np - ref_grid,
        "x0 = noise + v": noise_np + ref_grid,
        "x0 = v": ref_grid,
        "x0 = -v": -ref_grid,
        "x0 = noise - v (off v)": noise_np - v_off_grid,
    }
    print("[cmp] 与官方 x0_off 比较（确定公式）：")
    best = None
    for name, cand in cands.items():
        c = cos(cand, x0_off_np)
        print(f"   cos({name:24s}) = {c:.6f}")
        if best is None or c > best[1]:
            best = (name, c)
    print(f"[cmp] 最佳公式: {best[0]}  cos={best[1]:.6f}")

    # 保存官方 ground truth 供后续使用
    def write_raw(path, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        with open(path, "wb") as fp:
            fp.write(struct.pack("<q", arr.ndim))
            for d in arr.shape:
                fp.write(struct.pack("<q", int(d)))
            fp.write(arr.tobytes())
    write_raw(os.path.join(WORK, "official_intermediates", "v_off.bin"), v_off_grid)
    write_raw(os.path.join(WORK, "official_intermediates", "x0_off.bin"), x0_off_np)
    print("[cmp] 已保存 v_off.bin / x0_off.bin")

    if c_fwd >= 0.99:
        print("[PASS] DiT 前向与官方对齐 (cos>=0.99)")
    else:
        print("[FAIL] DiT 前尚有 bug（需逐层对拍）")
    return 0 if c_fwd >= 0.99 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
逐层对拍（PyTorch FP32 版）：官方 DiT 逐块以 fp32 计算（权重 fp32 -> 注意力 fp32），
与我们的 numpy fp32 参考（run_e2e_ref 同款前向）每层后比对 cos。

目的：证明我们的 CPU 参考实现在「同精度(fp32)」下与官方 PyTorch 完全一致（>0.99）。
之前的 bisect_dit_layers.py 用官方 bf16 输出对拍得到 0.954，是因为官方 pipeline 硬编码
compute_dtype=bf16，bf16 噪声在 32 层注意力中累积。本脚本消除 bf16 噪声，给出公平对拍。

策略：模型保持 bf16 存储省显存，循环中对第 i 个 block 调 .float() 临时转 fp32 计算，
结束后 .half() 转回；注意力 compute_dtype=None 阻止下转；关闭 tf32 保证纯 fp32。
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
    vid_grid = read_raw(os.path.join(WORK, "vid_grid.bin"))   # (1,H,W,33)
    T, H, W = vid_grid.shape[0], vid_grid.shape[1], vid_grid.shape[2]
    noise_t = torch.from_numpy(vid_grid[..., :16].copy()).to(DEVICE).float()
    cond17 = torch.from_numpy(vid_grid[..., 16:].copy()).to(DEVICE).float()
    vid_input = torch.cat([noise_t, cond17], dim=-1)
    vid_flat, vid_shape = na.flatten([vid_input])
    txt_raw = torch.from_numpy(read_raw(os.path.join(WORK, "txt.bin")).astype(np.float32)).to(DEVICE).float()

    ctx = setup_generation_context(dit_device=DEVICE, vae_device=DEVICE,
                                   dit_offload_device=None, vae_offload_device=None,
                                   tensor_offload_device=None, debug=Debug(enabled=False))
    ctx['compute_dtype'] = torch.float32   # 阻止 attention 下转 bf16
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
    # 保持 bf16 存储；逐块转 fp32
    dit = runner.dit.to(DEVICE).half().eval()
    # 关闭所有 block 注意力的下转
    for blk in dit.blocks:
        blk.attn.attn.compute_dtype = None

    txt_flat, txt_shape = na.flatten([txt_raw])

    # 初始 embedding 以 fp32 计算
    dit.vid_in.float()
    dit.txt_in.float()
    dit.emb_in.float()
    patched_vid, patched_vid_shape = dit.vid_in(vid_flat, vid_shape)
    txt_proj = dit.txt_in(txt_flat)
    tstept = torch.tensor([1000.0], device=DEVICE)
    emb_t = dit.emb_in(tstept, device=DEVICE, dtype=torch.float32)
    dit.vid_in.half(); dit.txt_in.half(); dit.emb_in.half()

    # 我们的对应项
    x_patch = R.patchify(vid_grid, T, H, W)
    with io.open_sd() as f:
        Wpin = io.get_top_weight(f, "vid_in.proj", "weight"); Bpin = io.get_top_weight(f, "vid_in.proj", "bias")
        Wtxt = io.get_top_weight(f, "txt_in", "weight"); Btxt = io.get_top_weight(f, "txt_in", "bias")
    our_vid_in = x_patch @ Wpin.T + Bpin
    our_txt_in = txt_raw.cpu().numpy().astype(np.float32) @ Wtxt.T + Btxt
    our_emb = R.time_embedding(1000.0)
    print(f"[bis-fp32] cos(emb)      = {cos(our_emb, emb_t.cpu().numpy()):.6f}")
    print(f"[bis-fp32] cos(init vid) = {cos(our_vid_in, patched_vid.cpu().numpy()):.6f}")
    print(f"[bis-fp32] cos(txt_in)   = {cos(our_txt_in, txt_proj.cpu().numpy()):.6f}")

    txt_np = txt_raw.cpu().numpy().astype(np.float32)
    our_vid = our_vid_in.astype(np.float32)
    our_txt = our_txt_in.astype(np.float32)
    emb_np = our_emb.astype(np.float32)
    emb3 = emb_np.reshape(1, R.HEAD_D * R.HEADS, 2, 3)

    cache = Cache(disable=True)
    off_vid = patched_vid.float()
    off_txt = txt_proj.float()
    emb_t = emb_t.float()
    print("\n[bis-fp32] layer : cos(our_vid, off_vid fp32) after block")
    min_c = 1.0
    for i in range(R.NUM_LAYERS):
        blk = dit.blocks[i]
        blk.float()
        off_vid, off_txt, _, _ = dit.blocks[i](
            vid=off_vid, txt=off_txt, vid_shape=patched_vid_shape, txt_shape=txt_shape,
            emb=emb_t, cache=cache)
        blk.half()
        # 我们的 block
        window_method = "720pwin_by_size_bysize" if i % 2 == 0 else "720pswin_by_size_bysize"
        Wb = R.block_weights(i)
        Adv = Wb["vid"]["ada"]; Adt = Wb["txt"]["ada"]
        vid_an = R.rmsnorm(our_vid); txt_an = R.rmsnorm(our_txt)
        sA, scA, gA = emb3[0, :, 0, 0], emb3[0, :, 0, 1], emb3[0, :, 0, 2]
        shB = Adv["attn_shift"]; scB = Adv["attn_scale"]; gB = Adv["attn_gate"]
        thB = Adt["attn_shift"]; tcB = Adt["attn_scale"]; tgB = Adt["attn_gate"]
        vid_an = vid_an * (scA + scB) + (sA + shB)
        txt_an = txt_an * (scA + tcB) + (sA + thB)
        vid_at, txt_at = R.awa_forward(vid_an, txt_an, (T, H // 2, W // 2), window_method, Wb["vid"], Wb["txt"], 58)
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
        vid_m = vid_m * (gAm + gBm); txt_m = txt_m * (gAm + tgBm)
        our_vid = (vid_m + vid).astype(np.float32); our_txt = (txt_m + txt_t).astype(np.float32)

        c = cos(our_vid, off_vid.cpu().numpy())
        min_c = min(min_c, c)
        mark = "" if c >= 0.99 else "  <-- 偏离"
        if i % 2 == 0 or c < 0.99:
            print(f"  layer {i:2d} : {c:.6f}{mark}")
    print(f"[bis-fp32] 完成；最小 cos = {min_c:.6f}")
    print("[bis-fp32] 若 min cos > 0.99，则我们的 CPU fp32 参考与 PyTorch fp32 一致（此前 0.954 纯为官方 bf16 噪声）")


if __name__ == "__main__":
    sys.exit(main())

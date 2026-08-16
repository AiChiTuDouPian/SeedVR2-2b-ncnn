#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
隔离探针：对目标层 L，用官方 post-ada-in 的 vid/txt 作为注意力输入，喂给我们的 awa_forward，
对比官方 attn 输出（post proj_out）。这隔离了「注意力公式本身」在每一层 window 配置下是否正确，
排除前面层累积误差的干扰。

同时捕获 attn_norm 输出（post RMSNorm）对比我们的 rmsnorm，确认归一化一致。
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

TARGET = [12, 13, 14, 15, 16]


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

    # hooks
    caps = {}
    def make_hook(li):
        def hook(module, inp, out):
            # attn module: inp = (vid_attn, txt_attn, vid_shape, txt_shape, cache)
            #            out = (vid_out_post_proj, txt_out_post_proj)
            caps[li] = {
                "attn_in_vid": inp[0].detach().cpu().numpy().astype(np.float32),
                "attn_in_txt": inp[1].detach().cpu().numpy().astype(np.float32),
                "attn_out_vid": out[0].detach().cpu().numpy().astype(np.float32),
                "attn_out_txt": out[1].detach().cpu().numpy().astype(np.float32),
            }
        return hook
    for li in TARGET:
        dit.blocks[li].attn.register_forward_hook(make_hook(li))

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

    # 对比目标层
    for li in TARGET:
        c = caps[li]
        window_method = "720pwin_by_size_bysize" if li % 2 == 0 else "720pswin_by_size_bysize"
        Wb = R.block_weights(li)
        # 用官方 post-ada-in 输入跑我们的 awa_forward
        our_vid_attn, our_txt_attn = R.awa_forward(
            c["attn_in_vid"], c["attn_in_txt"], (T, H // 2, W // 2), window_method, Wb["vid"], Wb["txt"], 58)
        cv = cos(our_vid_attn, c["attn_out_vid"])
        ct = cos(our_txt_attn, c["attn_out_txt"])
        print(f"[probe] layer {li:2d} ({window_method}): cos(our_attn_vid, official)={cv:.6f}  cos(our_attn_txt, official)={ct:.6f}")
        if cv < 0.999 or ct < 0.999:
            print(f"        -> 注意力公式在该层 window 配置下存在偏差")


if __name__ == "__main__":
    sys.exit(main())

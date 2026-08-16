#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最终 sr_latent 对比：PyTorch fp32 的 vid_out.proj 输出 (L,64) vs 我们的 numpy 参考 ref (L,64)。
bisect_dit_fp32 只验证了逐层 vid 隐状态，没验证最终 vid_out_norm+ada+proj。
这里补上最终输出对比，重点看「高频」（空间相邻差）是否一致。
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
        return np.frombuffer(fp.read(int(np.prod(shape)) * 4), dtype=np.float32).reshape(shape)


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

    x_patch = R.patchify(vid_grid, T, H, W)
    with io.open_sd() as f:
        Wpin = io.get_top_weight(f, "vid_in.proj", "weight"); Bpin = io.get_top_weight(f, "vid_in.proj", "bias")
        Wtxt = io.get_top_weight(f, "txt_in", "weight"); Btxt = io.get_top_weight(f, "txt_in", "bias")
    our_vid = (x_patch @ Wpin.T + Bpin).astype(np.float32)
    our_txt = (txt_raw.cpu().numpy().astype(np.float32) @ Wtxt.T + Btxt).astype(np.float32)
    our_emb = R.time_embedding(1000.0)
    emb3 = our_emb.reshape(1, R.HEAD_D * R.HEADS, 2, 3)

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
        window_method = "720pwin_by_size_bysize" if i % 2 == 0 else "720pswin_by_size_bysize"
        Wb = R.block_weights(i)
        Adv = Wb["vid"]["ada"]; Adt = Wb["txt"]["ada"]
        vid_an = R.rmsnorm(our_vid); txt_an = R.rmsnorm(our_txt)
        sA, scA, gA = emb3[0, :, 0, 0], emb3[0, :, 0, 1], emb3[0, :, 0, 2]
        vid_an = vid_an * (scA + Adv["attn_scale"]) + (sA + Adv["attn_shift"])
        txt_an = txt_an * (scA + Adt["attn_scale"]) + (sA + Adt["attn_shift"])
        vid_at, txt_at = R.awa_forward(vid_an, txt_an, (T, H // 2, W // 2), window_method, Wb["vid"], Wb["txt"], 58)
        vid_at = vid_at * (gA + Adv["attn_gate"]); txt_at = txt_at * (gA + Adt["attn_gate"])
        vid = vid_at + our_vid; txt_t = txt_at + our_txt
        vid_mn = R.rmsnorm(vid); txt_mn = R.rmsnorm(txt_t)
        sAm, scAm, gAm = emb3[0, :, 1, 0], emb3[0, :, 1, 1], emb3[0, :, 1, 2]
        vid_mn = vid_mn * (scAm + Adv["mlp_scale"]) + (sAm + Adv["mlp_shift"])
        txt_mn = txt_mn * (scAm + Adt["mlp_scale"]) + (sAm + Adt["mlp_shift"])
        vid_m = R.swiglu(vid_mn, Wb["vid"]["mlp_in"], Wb["vid"]["mlp_g"], Wb["vid"]["mlp_out"])
        txt_m = R.swiglu(txt_mn, Wb["txt"]["mlp_in"], Wb["txt"]["mlp_g"], Wb["txt"]["mlp_out"])
        vid_m = vid_m * (gAm + Adv["mlp_gate"]); txt_m = txt_m * (gAm + Adt["mlp_gate"])
        our_vid = (vid_m + vid).astype(np.float32); our_txt = (txt_m + txt_t).astype(np.float32)
        if (i + 1) % 8 == 0:
            print(f"  layer {i+1}/{R.NUM_LAYERS}  cos(vid)={cos(our_vid, off_vid.cpu().numpy()):.6f}")

    # ---- 最终输出：PyTorch fp32 的 vid_out ----
    # off_vid 是 (L, DIM)。官方：vid_out_norm -> vid_out_ada -> vid_out(proj)
    # 注意：官方 nadit.py 传完整 emb(15360) 给 vid_out_ada 会因 rearrange(d=5120) 报错，
    # 正确应为 emb 前 7680 维（= 3*2560，与我们的 run_e2e_ref 一致）。
    dit.vid_out_norm.float(); dit.vid_out_ada.float(); dit.vid_out.float()
    vn = dit.vid_out_norm(off_vid)
    vn = dit.vid_out_ada(vn, emb=emb_t[:, :3 * R.DIM], layer="out", mode="in",
                         hid_len=cache("vid_len", lambda: patched_vid_shape.prod(-1)),
                         cache=cache, branch_tag="vid")
    vn = vn.float()
    sr_pt = dit.vid_out.proj(vn)   # (L, 64)
    sr_pt = sr_pt.detach().cpu().numpy().astype(np.float32)

    # ---- 我们的最终输出 ----
    with io.open_sd() as f:
        Wvon = io.get_top_weight(f, "vid_out_norm", "weight")
        Wvoa_s = io.load_key(f, "vid_out_ada.out_shift")
        Wvoa_sc = io.load_key(f, "vid_out_ada.out_scale")
        Wvout = io.get_top_weight(f, "vid_out.proj", "weight"); Bvout = io.get_top_weight(f, "vid_out.proj", "bias")
    our_vn = R.rmsnorm(our_vid, Wvon)
    emb_out3 = our_emb[:, :R.HEAD_D * R.HEADS * 3].reshape(1, R.HEAD_D * R.HEADS, 1, 3)
    sAo = emb_out3[0, :, 0, 0]; scAo = emb_out3[0, :, 0, 1]
    our_vn = our_vn * (scAo + Wvoa_sc) + (sAo + Wvoa_s)
    our_sr = our_vn @ Wvout.T + Bvout   # (L, 64)

    print(f"\n=== 最终 sr_latent (L,64) 对比 ===")
    print(f"  cos(PyTorch fp32, ours) = {cos(sr_pt, our_sr):.6f}")
    print(f"  std: PyTorch={sr_pt.std():.4f}  ours={our_sr.std():.4f}")
    # 高频：空间相邻差（token 空间）
    Lv = T * (H // 2) * (W // 2)
    W2 = W // 2
    def spatial_adjdiff(x):
        # x: (Lv, 64)，取 channel 0，reshape (T, H2, W2)，算相邻差
        g = x[:, 0].reshape(T, H // 2, W2)
        return float(np.abs(np.diff(g, axis=2)).mean())
    print(f"  相邻差(token W方向): PyTorch={spatial_adjdiff(sr_pt):.4f}  ours={spatial_adjdiff(our_sr):.4f}")


if __name__ == "__main__":
    sys.exit(main())

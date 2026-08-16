#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M1+ 复用工具：权重加载、block 权重解析、pnnx 导出 Linear、参考数据存储。

权重解析权威规则（已查证 seedvr2_ema_3b_fp16.safetensors）：
- blocks 0-9   : shared_weights=False -> 键形如 blocks.b.{module}.vid.{name} / .txt.{name}
- blocks 10-31 : shared_weights=True  -> 键形如 blocks.b.{module}.all.{name}
- 顶层模块     : 形如 vid_in.proj.weight / emb_in.proj_in.weight / vid_out_ada.out_scale ...
"""
import os
import numpy as np
import torch
import pnnx
from safetensors import safe_open

DEFAULT_CKPT = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler/models/SEEDVR2/seedvr2_ema_3b_fp16.safetensors"
MM_LAYERS = 10
NUM_LAYERS = 32


def open_sd(ckpt=None):
    ckpt = ckpt or os.environ.get("SEEDVR2_CKPT", DEFAULT_CKPT)
    return safe_open(ckpt, framework="np")


def _branch_for_block(block, branch):
    """block<10 用给定 branch(vid/txt)；block>=10 强制用 'all'。"""
    return "all" if block >= MM_LAYERS else branch


def get_block_weight(f, block, module, name, branch="vid"):
    """读取 block 权重。module 不含 branch，如 'attn.proj_qkv'；name='weight'/'bias'。
    bias 缺失时返回 None（SeedVR2 的 proj_qkv/mlp 无 bias）。"""
    b = _branch_for_block(block, branch)
    key = f"blocks.{block}.{module}.{b}.{name}"
    if key not in f.keys():
        if name == "bias":
            return None
        raise KeyError(f"缺失 key: {key}（可用: 检查该 block 是 dual 还是 shared）")
    return np.asarray(f.get_tensor(key), dtype=np.float32)


def get_top_weight(f, module, name):
    key = f"{module}.{name}"
    if key not in f.keys():
        if name == "bias":
            return None
        raise KeyError(f"缺失顶层 key: {key}")
    return np.asarray(f.get_tensor(key), dtype=np.float32)


def load_key(f, key):
    """按完整 key 加载（缺失返回 None）。用于 branch 位置不一致的模块（attn vs mlp）。"""
    if key not in f.keys():
        return None
    return np.asarray(f.get_tensor(key), dtype=np.float32)


def block_key(block, sub, branch, name, shared_pattern=None):
    """构造 block 权重 key。shared_pattern 控制 branch 插入位置：
       'attn' -> blocks.b.attn.{sub}.{branch}.{name}
       'mlp'  -> blocks.b.mlp.{branch}.{sub}.{name}
       'ada'  -> blocks.b.ada.{branch}.{sub}（name 为 gate/scale/shift 等）
    branch 在 block>=MM_LAYERS 时强制 'all'。"""
    b = "all" if block >= MM_LAYERS else branch
    if shared_pattern == "mlp" or shared_pattern == "ada":
        return f"blocks.{block}.{shared_pattern}.{b}.{sub}.{name}"
    return f"blocks.{block}.attn.{sub}.{b}.{name}"


def export_linear(out_prefix, weight_f32, bias_f32, fp16=True):
    """用 pnnx 导出一个 Linear(out,in) 到 ncnn(fp16)。weight_f32/bias_f32 为 float32 numpy。

    注意：Windows 路径里的 \\v / \\t 会被 pnnx 生成的 .py 误当转义符，必须把所有路径
    统一转成正斜杠再传给 pnnx（这是 M1 踩过的一个坑）。
    """
    out_prefix = out_prefix.replace("\\", "/")
    out_out, out_in = weight_f32.shape
    m = torch.nn.Linear(out_in, out_out, bias=bias_f32 is not None)
    with torch.no_grad():
        m.weight.copy_(torch.from_numpy(np.ascontiguousarray(weight_f32, dtype=np.float32)))
        if bias_f32 is not None:
            m.bias.copy_(torch.from_numpy(np.ascontiguousarray(bias_f32, dtype=np.float32)))
    m.eval()
    x = torch.randn(1, out_in, dtype=torch.float32)
    ptpath = out_prefix + ".pt"
    pnnx.export(
        m, ptpath, inputs=(x,),
        ncnnparam=out_prefix + ".param", ncnnbin=out_prefix + ".bin", fp16=fp16,
    )


def save_npz(path, **arrays):
    np.savez(path, **arrays)

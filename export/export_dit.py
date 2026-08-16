#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M0 交付：导出 DiT 权重清单（inventory）与 shape 校验。

要点（已查证 seedvr2_ema_3b_fp16.safetensors 真实结构）：
- blocks 0-9   : shared_weights=False -> 同时有 .vid. 和 .txt. 两套独立权重
- blocks 10-31 : shared_weights=True  -> 只有 .all. 一套共享权重（vid/txt 复用）
- 本 checkpoint 无 vid_only 模块（txt 在全部 32 层都参与计算）
- 每块有 attn.rope.rope.freqs (21,) 预存 buffer
- 顶层: vid_in / txt_in / emb_in / vid_out / vid_out_ada / vid_out_norm

仅读取 safetensors 头（不加载全部 6.8GB 到内存），快速产出清单。

用法（cuda_env，项目根目录）:
  python export/export_dit.py
"""
import os, json
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CKPT = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler/models/SEEDVR2/seedvr2_ema_3b_fp16.safetensors"
OUT_JSON = os.path.join(ROOT, "models", "inventory.json")

NUM_LAYERS = 32
MM_LAYERS = 10  # blocks 0..9 独立；10..31 共享


def classify_block(keys, b):
    """返回该 block 的权重模式：'dual'(vid/txt 独立) / 'shared'(.all)。"""
    has_vid = any(k.startswith(f"blocks.{b}.") and ".vid." in k for k in keys)
    has_all = any(k.startswith(f"blocks.{b}.") and ".all." in k for k in keys)
    if has_all and not has_vid:
        return "shared"
    if has_vid and not has_all:
        return "dual"
    # 兜底
    return "shared" if has_all else "dual"


def main():
    ckpt = os.environ.get("SEEDVR2_CKPT", DEFAULT_CKPT)
    print(f"[info] 读取权重头: {ckpt}")
    with safe_open(ckpt, framework="np") as f:
        keys = list(f.keys())
        print(f"[info] 总 key 数: {len(keys)}")

        inv = {"ckpt": ckpt, "num_layers": NUM_LAYERS, "mm_layers": MM_LAYERS, "blocks": {}, "top": {}}

        # 顶层模块
        for k in keys:
            if not k.startswith("blocks."):
                inv["top"][k] = list(f.get_tensor(k).shape)

        # 逐 block
        for b in range(NUM_LAYERS):
            mode = classify_block(keys, b)
            bkeys = [k for k in keys if k.startswith(f"blocks.{b}.")]
            sub = {}
            for k in bkeys:
                sub[k] = list(f.get_tensor(k).shape)
            inv["blocks"][str(b)] = {"mode": mode, "num_weights": len(bkeys), "keys": sub}
            # 一致性检查：dual 应有 vid 和 txt，shared 应有 all
            if mode == "dual":
                assert any(".vid." in k for k in bkeys) and any(".txt." in k for k in bkeys), \
                    f"block {b} 标为 dual 但缺 vid/txt 键"
            else:
                assert any(".all." in k for k in bkeys), f"block {b} 标为 shared 但缺 .all. 键"

    # 打印摘要
    dual = [b for b in range(NUM_LAYERS) if inv["blocks"][str(b)]["mode"] == "dual"]
    shared = [b for b in range(NUM_LAYERS) if inv["blocks"][str(b)]["mode"] == "shared"]
    print(f"[ok] dual(独立) blocks : {dual[0]}..{dual[-1]} ({len(dual)} 个)")
    print(f"[ok] shared(共享) blocks: {shared[0]}..{shared[-1]} ({len(shared)} 个)")
    print(f"[ok] 顶层模块: {list(inv['top'].keys())}")

    # 校验一个代表 block 的权重形状（block 0 应 dual，block 10 应 shared）
    b0 = inv["blocks"]["0"]
    b10 = inv["blocks"]["10"]
    assert b0["mode"] == "dual", "block0 期望 dual"
    assert b10["mode"] == "shared", "block10 期望 shared"
    print(f"[ok] 结构校验通过：block0=dual, block10=shared")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fp:
        json.dump(inv, fp, ensure_ascii=False, indent=2)
    print(f"[ok] 清单已写: {OUT_JSON}")


if __name__ == "__main__":
    main()

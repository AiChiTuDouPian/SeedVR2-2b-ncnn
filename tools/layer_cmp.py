#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""逐层（逐块）对比：把引擎 `SEEDVR_DUMP_BLOCKOUT=1` 产出的 `blkout_{b}_vid.f32`
（每块输出后的 vid 残差流，形状 (Lv, 2560) fp32 裸数据）按「层号」对齐后对拍，
输出每层 cos / max|d| / mean|d| 表格。

不同 chunk 的 dump 层号不同：块 b（chunk=C）对应「第 (b+1)*C 层之后」的残差流。
本脚本按层号取交集，因此 chunk=1 与 chunk=2 的 dump 可以混用。

用法:
  python tools/layer_cmp.py --ref 名字=目录:chunk [--cmp 名字=目录:chunk ...]
例:
  python tools/layer_cmp.py --ref fp32=diag/layer_cmp/360_fp32:2 \
                            --cmp bf16=diag/layer_cmp/360_bf16:2 \
                            --cmp fp16=diag/layer_cmp/360_fp16:1
"""
import os
import sys

import numpy as np


def load_by_layer(d, chunk):
    """{层号: ndarray}；层号 = (块号+1)*chunk。"""
    out = {}
    for fn in sorted(os.listdir(d)):
        if not (fn.startswith("blkout_") and fn.endswith("_vid.f32")):
            continue
        b = int(fn[len("blkout_"):-len("_vid.f32")])
        out[(b + 1) * chunk] = np.fromfile(os.path.join(d, fn), dtype="<f4")
    return out


def parse_spec(s):
    """'名字=目录:chunk' -> (名字, 目录, chunk)"""
    name, rest = s.split("=", 1)
    d, _, ck = rest.rpartition(":")
    return name, d, int(ck)


def metrics(x, y):
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    d = y - x
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    cos = float(x @ y / (nx * ny)) if nx > 0 and ny > 0 else float("nan")
    return cos, float(np.abs(d).max()), float(np.abs(d).mean())


def main():
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    ref_spec, cmp_specs = None, []
    i = 0
    while i < len(argv):
        if argv[i] == "--ref":
            ref_spec = parse_spec(argv[i + 1]); i += 2
        elif argv[i] == "--cmp":
            cmp_specs.append(parse_spec(argv[i + 1])); i += 2
        else:
            i += 1
    if ref_spec is None or not cmp_specs:
        print(__doc__)
        return 1

    rname, rdir, rchunk = ref_spec
    ref = load_by_layer(rdir, rchunk)
    if not ref:
        print(f"[layer_cmp] 参考目录无 dump: {rdir}")
        return 1
    print(f"参考 {rname}: {rdir}  (chunk={rchunk}, {len(ref)} 层, 每层 {ref[min(ref)].size} 元素)\n")

    for cname, cdir, cchunk in cmp_specs:
        cm = load_by_layer(cdir, cchunk)
        layers = sorted(set(ref) & set(cm))
        if not layers:
            print(f"=== {cname}: 与参考无重叠层（参考层 {sorted(ref)[:4]}…，对比层 {sorted(cm)[:4]}…）===\n")
            continue
        print(f"=== {cname} vs {rname}（chunk={cchunk}，共 {len(layers)} 层对齐）===")
        print(f"{'层':>4} {'cos':>13} {'max|d|':>12} {'mean|d|':>12}")
        cs = []
        for L in layers:
            cos, mx, mn = metrics(ref[L], cm[L])
            cs.append(cos)
            print(f"{L:>4} {cos:>13.8f} {mx:>12.4f} {mn:>12.6f}")
        print(f"  最小 cos = {min(cs):.8f}（第 {layers[int(np.argmin(cs))]} 层）\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

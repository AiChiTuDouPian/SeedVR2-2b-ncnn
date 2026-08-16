#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
M0 烟雾测试：用 pnnx 导出一个最小 Linear(3->2) 到 ncnn(fp16)，
供 C++ 端验证「fp16 .bin 加载 + InnerProduct 运行」。
同时打印用固定权重算出的期望输出，便于 C++ 端核对。

用法（在 cuda_env 下，项目根目录执行）：
  python export/make_test_model.py
"""
import os, sys, torch, pnnx, numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)

PARAM_FP32 = os.path.join(MODELS, "test_linear_fp32.param")
BIN_FP32 = os.path.join(MODELS, "test_linear_fp32.bin")
PARAM_FP16 = os.path.join(MODELS, "test_linear.param")
BIN_FP16 = os.path.join(MODELS, "test_linear.bin")


class TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2, bias=True)

    def forward(self, x):
        return self.fc(x)


def main():
    m = TinyLinear()
    m.eval()
    # 固定权重，便于手工核对
    with torch.no_grad():
        m.fc.weight.copy_(torch.tensor([[0.1, 0.2, 0.3],
                                         [0.4, 0.5, 0.6]]))
        m.fc.bias.copy_(torch.tensor([0.7, 0.8]))

    x = torch.rand(1, 3)  # 仅用于 trace，值不重要（权重已固定）

    # pnnx.export: trace -> ncnn，fp16=True 直接产出 fp16 ncnn（带 0x01306B47 tag）
    ptpath = os.path.join(MODELS, "_tiny_linear.pt")
    pnnx.export(
        m, ptpath,
        inputs=(x,),
        ncnnparam=PARAM_FP16,
        ncnnbin=BIN_FP16,
        fp16=True,
    )
    print(f"[ok] pnnx 导出 fp16: {PARAM_FP16} / {BIN_FP16}")

    # 打印期望输出（输入 [1,2,3]）：y = Wx + b
    inp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    W = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    b = np.array([0.7, 0.8], dtype=np.float32)
    out = W @ inp + b
    print(f"[ref] 输入 x = {inp.tolist()}")
    print(f"[ref] 期望输出 y = {out.tolist()}  (fp16 下应几乎一致)")

    # 打印 param 的 Input/输出名，供 C++ 端使用
    with open(PARAM_FP16) as f:
        lines = f.read().splitlines()
    print("[param] 关键行:")
    for ln in lines:
        if ln.startswith("Input") or ln.startswith("InnerProduct") or ln.endswith("out0") or "out0" in ln:
            print("   ", ln)


def fp32_to_fp16(param_in, bin_in, param_out, bin_out):
    """将 ncnn fp32 .bin 转 fp16。按 param 中 MemoryData 的声明顺序「边读边推进」。
    fp16 blob 格式: 4 字节 tag(0x01306B47, little-endian) + 裸 fp16 字节。"""
    import struct
    with open(param_in, "r") as f:
        lines = f.read().split("\n")
    # 解析 MemoryData 的 (w,h,c) 以得到每个权重 blob 的大小（顺序即 bin 顺序）
    sizes = []
    for ln in lines[2:]:
        if not ln.strip():
            continue
        parts = ln.split()
        if parts[0] == "MemoryData":
            params = {}
            for tok in parts[4:]:
                if "=" in tok:
                    k, v = tok.split("=")
                    params[k] = int(v)
            w = params.get("0", 1)
            h = params.get("1", 1)
            c = params.get("2", 1)
            sizes.append(w * h * c)
    raw = open(bin_in, "rb").read()
    off = 0
    out = bytearray()
    TAG = struct.pack("<I", 0x01306B47)
    for size in sizes:
        arr = np.frombuffer(raw, dtype=np.float32, count=size, offset=off)
        off += size * 4
        half = arr.astype(np.float16)
        out += TAG
        out += half.tobytes()
    if off != len(raw):
        print(f"[warn] bin 长度未耗尽: 已读 {off}, 总长 {len(raw)}（可能 param 解析不全）")
    with open(bin_out, "wb") as f:
        f.write(out)
    # param 原样复制（ncnn 靠 bin 的 tag 判断 fp16，不靠 param）
    with open(param_out, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()

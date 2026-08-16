#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""验证 C++ 复现 PyTorch CPU torch.randn 的算法（mt19937 + cephes AVX2 近似 Box-Muller）。
   与 torch 真值逐位对比。"""
import struct
import numpy as np
import torch

def f2b(v):
    """float -> uint32 位模式（python int）"""
    return struct.unpack("<I", struct.pack("<f", float(np.float32(v))))[0]

def b2f(b):
    """uint32 位模式 -> float（python float，值是精确的 float32 值）"""
    return struct.unpack("<f", struct.pack("<I", int(b) & 0xffffffff))[0]

def F(v):
    """强制 float32 值（保留精确 float32 语义）"""
    return np.float32(v)

def fmul(a, b):  return b2f(f2b(F(a) * F(b)))
def fadd(a, b):  return b2f(f2b(F(a) + F(b)))
def fsub(a, b):  return b2f(f2b(F(a) - F(b)))

# ==================== mt19937 ====================
def make_mt(seed):
    mt = [0] * 624
    mt[0] = seed & 0xffffffff
    for i in range(1, 624):
        mt[i] = (1812433253 * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i) & 0xffffffff
    idx = 624
    def gen():
        nonlocal idx
        if idx >= 624:
            for i in range(624):
                y = (mt[i] & 0x80000000) | (mt[(i + 1) % 624] & 0x7fffffff)
                mt[i] = mt[(i + 397) % 624] ^ (y >> 1)
                if y & 1:
                    mt[i] ^= 0x9908b0df
            idx = 0
        y = mt[idx]
        idx += 1
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9d2c5680
        y ^= (y << 15) & 0xefc60000
        y ^= (y >> 18)
        return y & 0xffffffff
    return gen

# ==================== cephes log (float32, 精确复现 log256_ps) ====================
SQRTHF = b2f(f2b(0.707106781186547524))
LOG_P = [b2f(f2b(v)) for v in [
    7.0376836292E-2, -1.1514610310E-1, 1.1676998740E-1, -1.2420140846E-1,
    1.4249322787E-1, -1.6668057665E-1, 2.0000714765E-1, -2.4999993993E-1,
    3.3333331174E-1]]
LOG_Q1 = b2f(f2b(-2.12194440e-4))
LOG_Q2 = b2f(f2b(0.693359375))
MIN_NORM = b2f(0x00800000)

def cephes_log(x):
    x = max(x, MIN_NORM)  # float32 max
    xb = f2b(x)
    imm0 = xb >> 23                     # 指数（含 bias 127）
    xb = (xb & 0x807fffff) | 0x3f000000  # 清零指数，设为 0.5~1
    x = b2f(xb)
    imm0 = imm0 - 0x7f
    e = b2f(f2b(F(imm0)))               # int -> float32
    e = fadd(e, 1.0)
    mask = x < SQRTHF
    tmp = x if mask else 0.0
    x = fsub(x, 1.0)
    e = fsub(e, 1.0) if mask else e
    x = fadd(x, tmp)
    z = fmul(x, x)
    y = LOG_P[0]
    for p in LOG_P[1:]:
        y = fadd(fmul(y, x), p)
    y = fmul(y, x)
    y = fmul(y, z)
    y = fadd(y, fmul(e, LOG_Q1))
    y = fsub(y, fmul(z, 0.5))
    x = fadd(x, y)
    x = fadd(x, fmul(e, LOG_Q2))
    return x

# ==================== cephes sincos (float32, 精确复现 sincos256_ps) ====================
DP1 = b2f(f2b(-0.78515625))
DP2 = b2f(f2b(-2.4187564849853515625e-4))
DP3 = b2f(f2b(-3.77489497744594108e-8))
SIN_P = [b2f(f2b(v)) for v in [-1.9515295891E-4, 8.3321608736E-3, -1.6666654611E-1]]
COS_P = [b2f(f2b(v)) for v in [2.443315711809948E-005, -1.388731625493765E-003, 4.166664568298827E-002]]
FOPI = b2f(f2b(1.27323954473516))

def cephes_sincos(x):
    sign_bit_sin = f2b(x) & 0x80000000
    x = b2f(f2b(x) & 0x7fffffff)          # abs
    y = fmul(x, FOPI)
    imm2 = int(F(y))                       # cvttps_epi32（向零截断）
    imm2 = (imm2 + 1) & (~1)
    y = b2f(f2b(F(imm2)))
    imm4 = imm2 & 0xffffffff
    imm0 = ((imm2 & 4) << 29) & 0xffffffff  # sine swap sign
    imm2a = imm2 & 2
    poly_mask = 0xffffffff if imm2a == 0 else 0
    swap_sign_bit_sin = imm0
    # magic pass
    x = fadd(fadd(fadd(x, fmul(y, DP1)), fmul(y, DP2)), fmul(y, DP3))
    imm4 = (imm4 - 2) & 0xffffffff
    imm4 = ((~imm4) & 4) << 29
    sign_bit_cos = imm4 & 0xffffffff
    sign_bit_sin = (sign_bit_sin ^ swap_sign_bit_sin) & 0xffffffff
    z = fmul(x, x)
    # cos 多项式
    y = COS_P[0]
    for p in COS_P[1:]:
        y = fadd(fmul(y, z), p)
    y = fmul(y, z)
    y = fmul(y, z)
    y = fsub(y, fmul(z, 0.5))
    y = fadd(y, 1.0)
    # sin 多项式
    y2 = SIN_P[0]
    for p in SIN_P[1:]:
        y2 = fadd(fmul(y2, z), p)
    y2 = fmul(y2, z)
    y2 = fmul(y2, x)
    y2 = fadd(y2, x)
    # select（and_ps / andnot_ps 位运算）
    ysin2 = (f2b(y2) & poly_mask) & 0xffffffff
    ysin1 = (f2b(y) & (~poly_mask)) & 0xffffffff
    y2b = (f2b(y2) - ysin2) & 0xffffffff
    yb = (f2b(y) - ysin1) & 0xffffffff
    xmm1 = (ysin1 + ysin2) & 0xffffffff
    xmm2 = (yb + y2b) & 0xffffffff
    s = b2f(xmm1 ^ sign_bit_sin)
    c = b2f(xmm2 ^ sign_bit_cos)
    return s, c

# ==================== uniform + Box-Muller ====================
def uniform_f32(v):
    # (v & 0xFFFFFF) * (1.0f / 16777216.0f)
    return b2f(f2b(F(v & 0xFFFFFF) * F(1.0 / 16777216.0)))

TWO_PI_F32 = b2f(f2b(2.0 * np.pi))  # float32(6.283185307179586) = 6.2831855f

def box_muller_16(data):
    for j in range(8):
        u1 = fsub(1.0, data[j])
        u2 = data[j + 8]
        radius = b2f(f2b(np.float32(np.sqrt(F(-2.0) * F(cephes_log(u1))))))
        # AVX2: theta = float32( float32(2*pi) * u2 )
        theta = fmul(TWO_PI_F32, u2)
        s, c = cephes_sincos(theta)
        data[j] = b2f(f2b(F(radius) * F(c)))
        data[j + 8] = b2f(f2b(F(radius) * F(s)))
    return data

def randn_cpu(seed, n):
    g = make_mt(seed)
    uni = [uniform_f32(g()) for _ in range(n)]
    out = [0.0] * n
    # 主循环：每 16 个做 Box-Muller（只处理完整块）
    for i in range(0, n - 15, 16):
        out[i:i + 16] = box_muller_16(uni[i:i + 16])
    # 末尾：重新生成最后 16 个 uniform，覆盖 out[n-16:n]
    if n % 16 != 0:
        data = [uniform_f32(g()) for _ in range(16)]
        out[n - 16:n] = box_muller_16(data)
    return out

if __name__ == "__main__":
    N = 64
    torch.manual_seed(42)
    ref = torch.randn(N, dtype=torch.float32).numpy()
    ref_bits = [f2b(v) for v in ref]
    sim = randn_cpu(42, N)
    sim_bits = [f2b(v) for v in sim]
    match = sum(1 for a, b in zip(ref_bits, sim_bits) if a == b)
    print(f"逐位匹配: {match}/{N}")
    for i, (a, b) in enumerate(zip(ref_bits, sim_bits)):
        if a != b:
            print(f"  idx {i}: torch={a:08x} sim={b:08x}  torch_val={ref[i]:.9g} sim_val={sim[i]:.9g}")
    # 也测非 16 倍数
    torch.manual_seed(7)
    ref2 = torch.randn(20, dtype=torch.float32).numpy()
    sim2 = randn_cpu(7, 20)
    match2 = sum(1 for a, b in zip([f2b(v) for v in ref2], [f2b(v) for v in sim2]) if a == b)
    print(f"非16倍数(20) 逐位匹配: {match2}/20")

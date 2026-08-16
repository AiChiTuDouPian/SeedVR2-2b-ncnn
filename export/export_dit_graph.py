#!/usr/bin/env python3
# export_dit_graph.py — 阶段2：把 216 个独立 param/bin 合并成单个 32 层 DiT 整图
# 输出：models/m5_graph/dit_graph.param + dit_graph.bin
#
# 整图结构（vid/txt 双流，AWA 处汇合）：
#   Input: in_vid0, in_txt0 + 每层每流 ada 参数（attn scale/shift/gate, mlp scale/shift/gate）
#   每层 i:
#     vid: RMSNorm -> ada(scale,shift) -> qkv --\
#     txt: RMSNorm -> ada(scale,shift) -> qkv --+-> AWA -> v_attn/t_attn
#     vid: gate -> 残差 -> RMSNorm -> ada -> mlp_in/g -> Swish(mul) -> mlp_out -> gate -> 残差
#     txt: 同
#   最终: RMSNorm(affine, Wvon) -> ada -> vid_out_proj(64) -> out0
#
# ada 参数是运行时由宿主预计算的（emb + branch 常数），作为图的 Input 注入；
# 权重 bin 直接拼接现有 216 个 .bin（fp16 flag 保留，与单层加载数值一致）。
# AWA 层 nq/nk 从 branch .bin（fp32 raw）读，写 type=0 格式（4 字节 flag 0 + fp32）。
# 用法: python export_dit_graph.py [model_dir]
import os, struct, sys

DIM = 2560
QKV = 7680
MLP = 6912
NUM_LAYERS = 32
MM_LAYERS = 10      # 前 10 层双流（vid+txt），后 22 层单流（all）
TXT = 58
ROPE_ROT = 126
EPS = "0.00001"     # RMSNorm eps（≤9 位小数，避开 vstr_to_float 溢出）
SPV_DIR_PARAM = '"models/m5/"'

MODEL = sys.argv[1] if len(sys.argv) > 1 else "models/m5"
OUT = MODEL + "_graph"
# 可选：只生成前 N 层（阶段2 验证 1 层用，权重小可 fp32 加载）
N_LAYERS = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_LAYERS
if N_LAYERS > NUM_LAYERS:
    N_LAYERS = NUM_LAYERS
# 阶段3 分块子图：生成 [L0, L1) 层子图（输入=残差 latent，跳过 vid/txt 投影；L1==NUM_LAYERS 时含最终 norm+out_proj）
L0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
L1 = int(sys.argv[4]) if len(sys.argv) > 4 else N_LAYERS
OUT_NAME = sys.argv[5] if len(sys.argv) > 5 else "dit_graph"
# fp16 模式（第 6 个参数 "f16"）：AWA 前后插 Cast 层（InnerProduct fp16 存储 + AWA fp32 计算）
FP16_MODE = (sys.argv[6] if len(sys.argv) > 6 else "") == "f16"
if L1 > N_LAYERS:
    L1 = N_LAYERS
os.makedirs(OUT, exist_ok=True)


def parse_ip(base):
    """读现有 InnerProduct param 的最后一行，返回 (num_output, bias_term, weight_data_size)。"""
    with open(f"{MODEL}/{base}.param") as f:
        lines = f.read().splitlines()
    d = {}
    for tok in lines[-1].split():
        if '=' in tok:
            k, v = tok.split('=', 1)
            d[k] = v
    return int(d['0']), int(d['1']), int(d['2'])


def ip_blob(base, in_blob, out_blob, name):
    """生成一个 InnerProduct 层行 + 记录权重。"""
    nout, bias, wsize = parse_ip(base)
    lines.append(f"InnerProduct  {name}  1 1 {in_blob} {out_blob} 0={nout} 1={bias} 2={wsize}")
    with open(f"{MODEL}/{base}.bin", 'rb') as f:
        bin_parts.append(f.read())


def raw_f32(path):
    """读 read_raw_f 格式的 fp32 bin（int64 ndim + int64 shape×ndim + float data），只返回数据部分。"""
    with open(path, 'rb') as f:
        data = f.read()
    ndim = struct.unpack('<q', data[:8])[0]
    header = 8 + 8 * ndim
    return data[header:]


def awa_weight_blob(name, path, blob):
    """AWA 层权重：type=0（4 字节 flag 0 + fp32）。"""
    lines.append(f"AWA  {name}  2 2 {blob}")
    # nq_v/nk_v/nq_t/nk_t 由调用方先 append 到 awa_weights
    for p in path:
        bin_parts.append(struct.pack('<I', 0) + raw_f32(p))


lines = ["7767517"]
bin_parts = []
input_names = []


def add_input(name):
    input_names.append(name)
    lines.append(f"Input  {name}  0 1 {name}")


def ada_names(i, flow):
    """返回 (a_sc, a_sh, a_g, m_sc, m_sh, m_g) 的 Input blob 名。"""
    return tuple(f"ada_{i}_{flow}_{k}" for k in ("a_sc", "a_sh", "a_g", "m_sc", "m_sh", "m_g"))


# ---------------- 图结构 ----------------
# Inputs
if L0 == 0:
    add_input("in_vid0")
    add_input("in_txt0")
for i in range(L0, L1):
    flows = ("v", "t") if i < MM_LAYERS else ("v", "t")   # 后 22 层两流引用同一 ada（all）
    for f in flows:
        for k in ada_names(i, f):
            add_input(k)
if L1 == N_LAYERS:
    add_input("ada_fin_sc")
    add_input("ada_fin_sh")

# 顶层投影（残差 blob 用唯一名 v_cur_0 / t_cur_0，ncnn 的 blob 只能有一个 producer）
if L0 == 0:
    ip_blob("vid_in_proj", "in_vid0", "v_cur_0", "proj_vid_in")
    ip_blob("txt_in", "in_txt0", "t_cur_0", "proj_txt_in")
else:
    # 子图：输入就是上一块的残差 latent（(Lv,DIM)/(TXT,DIM)），直接作为 v_cur_{L0}
    lines.append(f"Input  v_cur_{L0}  0 1 v_cur_{L0}")
    lines.append(f"Input  t_cur_{L0}  0 1 t_cur_{L0}")

# 层
for i in range(L0, L1):
    dual = i < MM_LAYERS
    bv = f"b{i}_vid" if dual else f"b{i}_all"
    bt = f"b{i}_txt" if dual else f"b{i}_all"
    win_type = i % 2
    fv, ft = "v", "t"
    v_in = f"v_cur_{i}"
    t_in = f"t_cur_{i}"
    v_a = f"v_cur_{i}_a"      # attn 残差后（mlp 的输入）
    t_a = f"t_cur_{i}_a"
    v_out = f"v_cur_{i + 1}"  # mlp 残差后（传给下一层）
    t_out = f"t_cur_{i + 1}"

    # ---- vid 流 ----
    # fp16 模式：残差链（v_cur_*）保持 fp32（避免 fp16 域残差累积），层内算子转 f16
    (a_sc, a_sh, a_g, m_sc, m_sh, m_g) = ada_names(i, fv)
    v_in_l, t_in_l = v_in, t_in
    lines.append(f"RMSNorm       rn_a_{i}_v   1 1 {v_in_l} v_rn_{i} 0={DIM} 1={EPS} 2=0")
    lines.append(f"BinaryOp      ad_sc_{i}_v  2 1 v_rn_{i} {a_sc} v_m1_{i} 0=2")
    lines.append(f"BinaryOp      ad_sh_{i}_v  2 1 v_m1_{i} {a_sh} v_m2_{i} 0=0")
    ip_blob(f"{bv}_qkv", f"v_m2_{i}", f"v_qkv_{i}", f"qkv_{i}_v")
    # ---- txt 流 ----
    (t_sc, t_sh, t_g, tm_sc, tm_sh, tm_g) = ada_names(i, ft)

    lines.append(f"RMSNorm       rn_a_{i}_t   1 1 {t_in_l} t_rn_{i} 0={DIM} 1={EPS} 2=0")
    lines.append(f"BinaryOp      ad_sc_{i}_t  2 1 t_rn_{i} {t_sc} t_m1_{i} 0=2")
    lines.append(f"BinaryOp      ad_sh_{i}_t  2 1 t_m1_{i} {t_sh} t_m2_{i} 0=0")
    ip_blob(f"{bt}_qkv", f"t_m2_{i}", f"t_qkv_{i}", f"qkv_{i}_t")
    # ---- AWA（vid/txt 汇合）----
    # 低精度模式：AWA 内部处理 bf16 I/O（AwaLayer 内转换），图中无 Cast 层
    qv_in, qt_in = f"v_qkv_{i}", f"t_qkv_{i}"
    lines.append(f"AWA  awa_{i}  2 2 {qv_in} {qt_in} v_attn_{i} t_attn_{i} 0={SPV_DIR_PARAM} 1={win_type} 2=0 3={TXT}")
    bin_parts.append(struct.pack('<I', 0) + raw_f32(f"{MODEL}/{bv}_nq.bin"))
    bin_parts.append(struct.pack('<I', 0) + raw_f32(f"{MODEL}/{bv}_nk.bin"))
    bin_parts.append(struct.pack('<I', 0) + raw_f32(f"{MODEL}/{bt}_nq.bin"))
    bin_parts.append(struct.pack('<I', 0) + raw_f32(f"{MODEL}/{bt}_nk.bin"))
    # ---- attn 输出投影（out）后接 gate（对齐 awa_forward: vattn -> lin(out) -> gate -> 残差）----
    va_in, ta_in = f"v_attn_{i}", f"t_attn_{i}"
    ip_blob(f"{bv}_out", va_in, f"v_outp_{i}", f"out_{i}_v")
    ip_blob(f"{bt}_out", ta_in, f"t_outp_{i}", f"out_{i}_t")
    # ---- vid attn 后处理 ----
    lines.append(f"BinaryOp      g_{i}_v       2 1 v_outp_{i} {a_g} v_g_{i} 0=2")
    lines.append(f"BinaryOp      g_{i}_t       2 1 t_outp_{i} {t_g} t_g_{i} 0=2")
    vg_in, tg_in = f"v_g_{i}", f"t_g_{i}"
    lines.append(f"BinaryOp      res_{i}_v     2 1 {vg_in} {v_in} {v_a} 0=0")
    lines.append(f"BinaryOp      res_{i}_t     2 1 {tg_in} {t_in} {t_a} 0=0")
    # ---- vid mlp ----
    v_a_l, t_a_l = v_a, t_a
    lines.append(f"RMSNorm       rn_m_{i}_v   1 1 {v_a_l} v_rn2_{i} 0={DIM} 1={EPS} 2=0")
    lines.append(f"BinaryOp      ad_msc_{i}_v 2 1 v_rn2_{i} {m_sc} v_m3_{i} 0=2")
    lines.append(f"BinaryOp      ad_msh_{i}_v 2 1 v_m3_{i} {m_sh} v_m4_{i} 0=0")
    ip_blob(f"{bv}_mlp_in",  f"v_m4_{i}", f"v_h_{i}",   f"min_{i}_v")
    ip_blob(f"{bv}_mlp_g",   f"v_m4_{i}", f"v_g2_{i}",  f"mig_{i}_v")
    lines.append(f"Swish         sw_{i}_v     1 1 v_g2_{i} v_gs_{i}")
    lines.append(f"BinaryOp      mul_{i}_v    2 1 v_h_{i} v_gs_{i} v_hs_{i} 0=2")
    ip_blob(f"{bv}_mlp_out", f"v_hs_{i}", f"v_o_{i}",   f"mout_{i}_v")
    lines.append(f"BinaryOp      mg_{i}_v     2 1 v_o_{i} {m_g} v_og_{i} 0=2")
    # fp16 模式：mlp 输出转 f32 后做残差 ADD（fp32 域），v_out 保持 fp32 残差链
    vog_in = f"v_og_{i}"
    lines.append(f"BinaryOp      mres_{i}_v   2 1 {vog_in} {v_a} {v_out} 0=0")
    # ---- txt mlp ----
    lines.append(f"RMSNorm       rn_m_{i}_t   1 1 {t_a_l} t_rn2_{i} 0={DIM} 1={EPS} 2=0")
    lines.append(f"BinaryOp      ad_msc_{i}_t 2 1 t_rn2_{i} {tm_sc} t_m3_{i} 0=2")
    lines.append(f"BinaryOp      ad_msh_{i}_t 2 1 t_m3_{i} {tm_sh} t_m4_{i} 0=0")
    ip_blob(f"{bt}_mlp_in",  f"t_m4_{i}", f"t_h_{i}",   f"min_{i}_t")
    ip_blob(f"{bt}_mlp_g",   f"t_m4_{i}", f"t_g2_{i}",  f"mig_{i}_t")
    lines.append(f"Swish         sw_{i}_t     1 1 t_g2_{i} t_gs_{i}")
    lines.append(f"BinaryOp      mul_{i}_t    2 1 t_h_{i} t_gs_{i} t_hs_{i} 0=2")
    ip_blob(f"{bt}_mlp_out", f"t_hs_{i}", f"t_o_{i}",   f"mout_{i}_t")
    lines.append(f"BinaryOp      mg_{i}_t     2 1 t_o_{i} {tm_g} t_og_{i} 0=2")
    tog_in = f"t_og_{i}"
    lines.append(f"BinaryOp      mres_{i}_t   2 1 {tog_in} {t_a} {t_out} 0=0")

# 最终：vid 流出图（txt 流丢弃）
if L1 == N_LAYERS:
    lines.append(f"RMSNorm       rn_fin  1 1 v_cur_{L1} v_fin 0={DIM} 1={EPS} 2=1")
    bin_parts.append(raw_f32(f"{MODEL}/vid_out_norm_w.bin"))   # gamma（type=1 无 flag）
    lines.append(f"BinaryOp      fin_sc  2 1 v_fin ada_fin_sc v_f1 0=2")
    lines.append(f"BinaryOp      fin_sh  2 1 v_f1 ada_fin_sh v_f2 0=0")
    ip_blob("vid_out_proj", "v_f2", "out0", "outproj")

# ---------------- blob_count ----------------
blobs = set()
for ln in lines[1:]:
    parts = ln.split()
    # Input: 类型 名 0 1 blob
    if parts[0] == "Input":
        blobs.add(parts[4])
    else:
        nb = int(parts[2]); nt = int(parts[3])
        for k in range(4, 4 + nb): blobs.add(parts[k])
        for k in range(4 + nb, 4 + nb + nt): blobs.add(parts[k])

layer_count = len(lines) - 1
# blob_count = 所有层 top 数之和（ncnn 按「每层 top 顺序编号」分配 blob index，不去重）
blob_count = sum(int(ln.split()[3]) for ln in lines[1:])
if len(blobs) != blob_count:
    print(f"WARN: 去重 blob {len(blobs)} != top 总数 {blob_count}（有重复 top 名？）")
out_param = [lines[0], f"{layer_count} {blob_count}"] + lines[1:]
with open(f"{OUT}/{OUT_NAME}.param", 'w', newline='\n') as f:
    f.write("\n".join(out_param) + "\n")
with open(f"{OUT}/{OUT_NAME}.bin", 'wb') as f:
    for b in bin_parts:
        f.write(b)

print(f"[export_dit_graph] OK: {OUT}/{OUT_NAME}.param ({layer_count} layers, {blob_count} blobs, L0={L0} L1={L1})")
print(f"                    bin 大小 = {sum(len(b) for b in bin_parts)/1e6:.1f} MB")

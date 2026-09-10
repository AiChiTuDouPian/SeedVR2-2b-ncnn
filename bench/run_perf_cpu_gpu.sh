#!/bin/bash
# run_perf_cpu_gpu.sh — 性能批测：fp16 / bf16 / fp32 × GPU / CPU @1080p
#
# 目的：量化各精度/后端的单图端到端耗时，并给出 CPU 路径的内存可行性边界。
# 注意：本脚本用引擎默认设置（色彩校正开启），只用于「计时」；
#       质量对拍请用 bench/run_pt_compare.sh（必须 --no-colorfix 对齐官方 --color_correction none）。
#
# 产物：diag/perf_1080_20260910/<cfg>/{out.png,run.log} + summary.log
#
# 已知结论（本机 15.8GB 系统内存）：
#   - GPU fp16 102s / GPU bf16 101s / GPU fp32(forward) 185s  —— 全部成功
#   - CPU fp32 在 DiT ~20-24 层 OOM 崩溃（rc=132）；CPU bf16/fp16 同命运（x86 低精度不省内存）
#     根因：src/dit_vk.cpp evict_nets() 的 CPU 模式 net 缓存 cap=1024（≈全权重常驻）
#           + cache_file() 的 fcache 永久驻留 .bin 原始字节
#
# 用法: bash bench/run_perf_cpu_gpu.sh
cd F:/Seedvr2/seedvr2-ncnn || exit 1
OUT=diag/perf_1080_20260910
EXE=./build-cmake/seedvr2_run.exe
SRC=$OUT/src.png

mkdir -p $OUT
run() {
    local cfg=$1; shift
    local d=$OUT/$cfg
    mkdir -p $d
    echo "[batch] === $cfg 开始 $(date +%T) ===" | tee -a $OUT/summary.log
    local t0=$(date +%s)
    $EXE $SRC $d/out.png --resolution 1080 "$@" > $d/run.log 2>&1
    local rc=$?
    local t1=$(date +%s)
    echo "[batch] $cfg 结束 rc=$rc wall=$((t1 - t0))s ($(date +%T))  out=$d/out.png" | tee -a $OUT/summary.log
    if [ $rc -ne 0 ]; then
        echo "[batch] $cfg FAILED (rc=$rc), 日志尾部:" | tee -a $OUT/summary.log
        tail -5 $d/run.log | tee -a $OUT/summary.log
    fi
    return 0
}

echo "[batch] ===== SeedVR2 1080p 性能批跑 $(date '+%F %T') =====" | tee $OUT/summary.log
echo "[batch] 输入: $SRC  (短边=1080 → 输出 ~1272x1080)" | tee -a $OUT/summary.log

# ---- GPU 三精度 ----
run gpu_fp16    --fp16                       # graph 块路径 (f16 32块 chunk=1)
run gpu_bf16    --bf16                       # graph 块路径 (bf16 32块 chunk=1)
run gpu_fp32    --no-graph                   # forward_latent fp32 (graph fp32 1080p 已知 OOM)

# ---- CPU 三精度 (--cpu 强制 forward_latent; x86 低精度存储可能回退 fp32) ----
run cpu_fp32    --cpu
run cpu_bf16    --cpu --bf16
run cpu_fp16    --cpu --fp16

echo "[batch] ===== 全部完成 $(date '+%F %T') =====" | tee -a $OUT/summary.log

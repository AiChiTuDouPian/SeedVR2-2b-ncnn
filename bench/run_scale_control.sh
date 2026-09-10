#!/bin/bash
# run_scale_control.sh — 「任务缩放比」对照实验（解释不同分辨率下 ncnn vs PyTorch 一致性差异）
#
# 动机：
#   bench/run_pt_compare.sh 用同一张源图（1131x960）跑 360/720/1080，结果呈现明显的
#   分辨率依赖（PSNR 23.1 / 34.8 / 46.6 dB）。原因是「有效缩放比」不同：
#       360p  → 1131x960 缩到 424x360    = 0.375×  （大幅**缩小**，超分模型的分布外任务）
#       720p  → 缩到 848x720             = 0.75×   （缩小）
#       1080p → 缩到 1272x1080           = 1.125×  （轻微放大，接近训练分布）
#   SeedVR2 是超分/复原模型，输入被缩小时网络进入分布外区间，雅可比更大 →
#   两个实现之间任何微小数值差异都会被显著放大。
#
# 本实验：把源图先缩到 212x180（0.1875×），再让两边都做 --resolution 360
#   → 有效缩放比变成 360/180 = 2.0×（**真放大**，回到模型训练分布）
#   若此时 ncnn vs PT 的 PSNR 大幅回升到 ~45dB，即证明上述解释成立，
#   也说明 360/720 档的低 PSNR 不是实现缺陷。
#
# 用法: bash bench/run_scale_control.sh
cd F:/Seedvr2/seedvr2-ncnn || exit 1
OUT=diag/pt_compare_20260910/scale_control
EXE=./build-cmake/seedvr2_run.exe
PY="C:/Users/15327/.workbuddy/binaries/python/envs/default/Scripts/python.exe"
SRC_SMALL=input/椎名真白_212x180.png

mkdir -p $OUT
echo "[control] ===== 缩放比对照实验 $(date '+%F %T') =====" | tee $OUT/summary.log
echo "[control] 输入: $SRC_SMALL (212x180)  --resolution 360  → 有效缩放 2.0x（真放大）" | tee -a $OUT/summary.log

# 1) 官方 PyTorch 参考（同样用小图）
if [ ! -f "$OUT/pt360_small.png" ]; then
    echo "[control] 生成官方参考 ..." | tee -a $OUT/summary.log
    /c/Users/15327/miniconda3/envs/cuda_env/python.exe tools/pt_reference.py \
        "$SRC_SMALL" "$OUT/pt360_small.png" 360 > $OUT/pt_ref.log 2>&1
    echo "[control] 官方参考 rc=$?" | tee -a $OUT/summary.log
fi

# 2) ncnn 三精度
for P in "fp32:--no-graph" "bf16:--bf16" "fp16:--fp16"; do
    cfg=${P%%:*}; flag=${P##*:}
    echo "[control] --- ncnn $cfg ---" | tee -a $OUT/summary.log
    $EXE "$SRC_SMALL" $OUT/ncnn_${cfg}.png --resolution 360 --no-colorfix $flag > $OUT/ncnn_${cfg}.log 2>&1
    echo "[control] ncnn $cfg rc=$?" | tee -a $OUT/summary.log
done

# 3) 对拍
echo "" | tee -a $OUT/summary.log
"$PY" tools/img_metrics.py $OUT/pt360_small.png \
    $OUT/ncnn_fp32.png $OUT/ncnn_bf16.png $OUT/ncnn_fp16.png --ssim 2>&1 | tee -a $OUT/summary.log

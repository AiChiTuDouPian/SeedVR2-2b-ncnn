#!/bin/bash
# run_pt_compare.sh — ncnn 引擎 vs PyTorch 官方 的图像矩阵（椎名真白，非官方示例图）
#
# 一次运行同时产出两类数据：
#   ① 性能：wall / VAE enc / DiT 32 层 / VAE dec（从 run.log 提取）
#   ② 质量：与官方 PyTorch 参考图逐像素对拍（tools/img_metrics.py）
#
# ===== 口径（必须与 tools/pt_reference.py 成对使用，否则对拍无效）=====
#   输入   : input/椎名真白.png (1131x960)
#   分辨率 : 360 / 720 / 1080（短边，与官方 --resolution 同义）
#            ⚠ 480 档位不可用：源图宽 1131*480/960=565.5，ncnn 用 std::round→566，
#              官方取整到 4 的倍数→564，尺寸不一致无法逐像素对拍。
#   精度   : fp32 / fp16 / bf16
#   色彩   : --no-colorfix  ← 对齐官方 --color_correction none
#            ⚠ 这是本矩阵最关键的一条。两边算法实现不同（ncnn = LAB transfer 强度 0.8，
#              官方 lab = 小波多尺度），只关一边会引入全局色调差，PSNR 会从 ~46dB 掉到 ~27dB。
#   seed   : 42（引擎默认，与官方 --seed 42 一致）
#   参考   : F:/Seedvr2/pt_ref/mashiro_pt{360,720,1080}.png（tools/pt_reference.py 生成）
#
# fp32 说明：graph 分块路径 1080p 会 OOM（16GB 显存），故 fp32 统一走 --no-graph
#            （forward_latent）。历史已验证 graph chunk=1/8 在无导出 net 时自动回退
#            forward_latent，产物与 --no-graph 逐位一致，故口径一致。
#
# 用法: bash bench/run_pt_compare.sh
cd F:/Seedvr2/seedvr2-ncnn || exit 1
OUT=diag/pt_compare_20260910
EXE=./build-cmake/seedvr2_run.exe
SRC=input/椎名真白.png
PY="C:/Users/15327/.workbuddy/binaries/python/envs/default/Scripts/python.exe"

mkdir -p $OUT
echo "[matrix] ===== ncnn vs PyTorch 图像矩阵 $(date '+%F %T') =====" | tee $OUT/summary.log
echo "[matrix] 输入: $SRC  口径: --no-colorfix --seed 42" | tee -a $OUT/summary.log

run() {   # run <cfg> <res> <精度flags...>
    local cfg=$1 res=$2; shift 2
    local d=$OUT/$cfg
    mkdir -p $d
    echo "[matrix] --- $cfg (res=$res) 开始 $(date +%T) ---" | tee -a $OUT/summary.log
    local t0=$(date +%s)
    $EXE "$SRC" $d/out.png --resolution $res --no-colorfix "$@" > $d/run.log 2>&1
    local rc=$?
    local t1=$(date +%s)
    local wall=$((t1 - t0))
    echo "[matrix] $cfg rc=$rc wall=${wall}s  $(date +%T)" | tee -a $OUT/summary.log
    if [ $rc -ne 0 ]; then
        echo "[matrix] $cfg FAILED (rc=$rc), 尾部日志:" | tee -a $OUT/summary.log
        tail -5 $d/run.log | tee -a $OUT/summary.log
    fi
    return 0
}

for R in 360 720 1080; do
    run gpu_fp32_$R $R --no-graph
    run gpu_bf16_$R $R --bf16
    run gpu_fp16_$R $R --fp16
done

echo "[matrix] ===== 全部完成 $(date '+%F %T') =====" | tee -a $OUT/summary.log

# ---- 质量对拍：每档 vs 官方 PT 参考 ----
echo "" | tee -a $OUT/summary.log
echo "[matrix] ===== 质量对拍（vs 官方 PyTorch 参考）=====" | tee -a $OUT/summary.log
for R in 360 720 1080; do
    REF="F:/Seedvr2/pt_ref/mashiro_pt${R}.png"
    echo "" | tee -a $OUT/summary.log
    echo "--- resolution $R ---" | tee -a $OUT/summary.log
    "$PY" tools/img_metrics.py "$REF" \
        $OUT/gpu_fp32_$R/out.png $OUT/gpu_bf16_$R/out.png $OUT/gpu_fp16_$R/out.png \
        --ssim 2>&1 | tee -a $OUT/summary.log
done

echo "" | tee -a $OUT/summary.log
echo "[matrix] ===== 精度内自比（同一分辨率，以 fp32 为基准）=====" | tee -a $OUT/summary.log
for R in 360 720 1080; do
    echo "" | tee -a $OUT/summary.log
    echo "--- resolution $R ---" | tee -a $OUT/summary.log
    "$PY" tools/img_metrics.py $OUT/gpu_fp32_$R/out.png \
        $OUT/gpu_bf16_$R/out.png $OUT/gpu_fp16_$R/out.png 2>&1 | tee -a $OUT/summary.log
done

@echo off
setlocal enabledelayedexpansion
REM 批量导出旧格式（无 AdaCompose）32 个单层块图，覆盖 dit_block_bf16_*
cd /d F:\Seedvr2\seedvr2-ncnn
set SEEDVR_ADACOMPOSE=0
for /L %%b in (0,1,31) do (
  set /a L1=%%b+1
  echo exporting block %%b (L0=%%b L1=!L1!)
  C:\Users\15327\miniconda3\envs\cuda_env\python.exe export\export_dit_graph.py models\m5 32 %%b !L1! dit_block_bf16_%%b
)
echo DONE

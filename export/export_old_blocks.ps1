$env:SEEDVR_ADACOMPOSE = '0'
Set-Location 'F:\Seedvr2\seedvr2-ncnn'
$py = 'C:\Users\15327\miniconda3\envs\cuda_env\python.exe'
for ($b = 0; $b -lt 32; $b++) {
    $l1 = $b + 1
    Write-Output "export block $b (L0=$b L1=$l1)"
    $out = & $py 'export\export_dit_graph.py' 'models\m5' 32 $b $l1 ("dit_block_bf16_" + $b) 2>&1
    $out | Select-Object -Last 1
}
Write-Output 'EXPORT_ALL_DONE'

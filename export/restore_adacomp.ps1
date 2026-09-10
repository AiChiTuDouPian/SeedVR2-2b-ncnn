$src = 'F:\Seedvr2\seedvr2-ncnn\models\m5_graph_adacomp'
$dst = 'F:\Seedvr2\seedvr2-ncnn\models\m5_graph'
Get-ChildItem $src | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $dst $_.Name) -Force
}
Write-Output "restored $((Get-ChildItem $src).Count) files"

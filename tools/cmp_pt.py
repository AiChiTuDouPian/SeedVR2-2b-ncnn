import numpy as np, cv2, os, sys
def load(p):
    img = cv2.imread(p)
    return None if img is None else img.astype(np.float32)
def cosim(a,b):
    da=a/255.0; db=b/255.0
    if da.shape!=db.shape:
        db=cv2.resize(db,(da.shape[1],da.shape[0]))
    return float(np.sum(da*db)/(np.linalg.norm(da)*np.linalg.norm(db)+1e-9))
def psnr(a,b):
    da=a/255.0; db=b/255.0
    if da.shape!=db.shape:
        db=cv2.resize(db,(da.shape[1],da.shape[0]))
    mse=np.mean((da-db)**2)
    return 10*np.log10(1.0/(mse+1e-10))
pt=load('F:/Seedvr2/bench_pt_480.png')
if pt is None:
    print('no pt ref'); sys.exit(0)
# 列出要对比的 ncnn 输出
cands = {
    'bf16(AdaCompose图)': 'F:/Seedvr2/bf16_480.png',
    'fp32store(AdaCompose图)': 'F:/Seedvr2/fp32store_480.png',
    'bf16(旧格式图)': 'F:/Seedvr2/oldfmt_480.png',
}
print('PyTorch 480 ref:', pt.shape)
print('='*70)
for name, path in cands.items():
    nc = load(path)
    if nc is None:
        print(f'{name}: 缺失'); continue
    c = cosim(nc, pt); p = psnr(nc, pt)
    print(f'{name:30s} cos={c:.6f}  PSNR={p:.2f}dB')

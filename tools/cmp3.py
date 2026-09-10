import numpy as np, cv2
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
b=load('F:/Seedvr2/bf16_480.png')
f=load('F:/Seedvr2/fp32store_480.png')
print('sizes:', 'pt', None if pt is None else pt.shape, 'bf16', None if b is None else b.shape, 'fp32store', None if f is None else f.shape)
print('cos bf16 vs pt      =', cosim(b,pt))
print('cos fp32store vs pt =', cosim(f,pt))
print('cos bf16 vs fp32store=', cosim(b,f))
print('psnr bf16 vs pt      =', psnr(b,pt))
print('psnr fp32store vs pt =', psnr(f,pt))
# 差值热图保存
d1=np.abs((b/255.0)-(cv2.resize(pt/255.0,(b.shape[1],b.shape[0]))))
cv2.imwrite('F:/Seedvr2/diff_bf16_pt.png',(d1*255).astype(np.uint8))

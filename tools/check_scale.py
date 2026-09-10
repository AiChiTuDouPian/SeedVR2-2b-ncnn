import numpy as np, struct
with open('models/m5/r_dbgr_vid_b0_full.bin','rb') as f:
    ndim=struct.unpack('<q',f.read(8))[0]
    sh=[struct.unpack('<q',f.read(8))[0] for _ in range(ndim)]
    ref=np.frombuffer(f.read(),np.float32).reshape(sh)
nc=np.fromfile('blkout_0_vid.f32',np.float32).reshape(1350,2560)
k=float(np.sum(ref*nc)/np.sum(ref*ref))
cos=float(np.sum(ref*nc)/(np.linalg.norm(ref)*np.linalg.norm(nc)))
print('ref range',ref.min(),ref.max(),' nc range',nc.min(),nc.max())
print('scale k=',k,' cos=',cos)
ratio=nc/(k*ref+1e-6)
print('nc/(k*ref) mean',ratio.mean(),'std',ratio.std())
# 每 token 的 norm
refn=np.linalg.norm(ref,axis=1); ncn=np.linalg.norm(nc,axis=1)
print('ref norm per-token mean',refn.mean(),' nc norm per-token mean',ncn.mean())
print('norm ratio nc/ref mean',(ncn/(refn+1e-9)).mean())

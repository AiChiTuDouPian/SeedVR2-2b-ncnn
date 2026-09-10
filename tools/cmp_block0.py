import numpy as np, struct

# numpy 参考 r_dbgr_vid_b0_full.bin (save_raw: int64 ndim + int64 shapes + float32)
with open('models/m5/r_dbgr_vid_b0_full.bin','rb') as f:
    ndim = struct.unpack('<q', f.read(8))[0]
    sh = struct.unpack('<%dq'%ndim, f.read(8*ndim))
    ref = np.frombuffer(f.read(), np.float32).reshape(sh)
print('ref', ref.shape, 'range', ref.min(), ref.max())

# ncnn block0_vid.bin (raw float32)
nc = np.fromfile('block0_vid.bin', np.float32)
print('nc', nc.shape, 'range', nc.min(), nc.max())

Lv, DIM = 1350, 2560
if nc.size == Lv*DIM:
    nc = nc.reshape(Lv, DIM)
    cos = float(np.sum(ref*nc)/(np.linalg.norm(ref)*np.linalg.norm(nc)))
    print('cos =', cos)
    d = np.abs(ref-nc)
    print('max|diff| =', d.max(), 'mean =', d.mean())
    # 检查是否只是 scale
    # 每维做线性回归 nc ~ k*ref
    k = float(np.sum(ref*nc)/np.sum(ref*ref))
    print('least-squares scale k =', k)
    print('nc/(k*ref) mean =', (nc/(k*ref+1e-6)).mean())

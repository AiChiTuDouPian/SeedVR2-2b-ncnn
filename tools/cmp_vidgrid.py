import numpy as np, struct
# engine vid_grid.bin: Q + 2 int + H*W*33 f32 channel-last
with open('vid_grid.bin','rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]
    H=struct.unpack('<i',f.read(4))[0]; W=struct.unpack('<i',f.read(4))[0]
    eg=np.frombuffer(f.read(),np.float32).reshape(H,W,33)
print("engine vid_grid", eg.shape, "range", eg.min(), eg.max())
# e2e_work/vid_grid.bin: q ndim + q shapes + f32 (T,H,W,33)
with open('e2e_work/vid_grid.bin','rb') as f:
    ndim=struct.unpack('<q',f.read(8))[0]
    sh=[struct.unpack('<q',f.read(8))[0] for _ in range(ndim)]
    pg=np.frombuffer(f.read(),np.float32).reshape(sh)
pg=pg[0]
print("pt vid_grid", pg.shape, "range", pg.min(), pg.max())
if eg.shape==pg.shape:
    cos=float(np.sum(eg*pg)/(np.linalg.norm(eg)*np.linalg.norm(pg)+1e-12))
    d=np.abs(eg-pg)
    print("cos =", cos, "maxdiff", d.max(), "meandiff", d.mean())
    # 每通道对比
    for c in range(33):
        dc=np.abs(eg[...,c]-pg[...,c]).mean()
        if dc>0.05:
            print(f"  ch{c}: meandiff={dc:.4f}  engine[{eg[...,c].min():.3f},{eg[...,c].max():.3f}] pt[{pg[...,c].min():.3f},{pg[...,c].max():.3f}]")

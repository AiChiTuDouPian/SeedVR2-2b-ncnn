import numpy as np, struct, os
# 读 e2e_work/vid_grid.bin (write_raw: q ndim + q shapes + f32 data)
def read_raw(p):
    with open(p,'rb') as f:
        ndim=struct.unpack('<q',f.read(8))[0]
        sh=[struct.unpack('<q',f.read(8))[0] for _ in range(ndim)]
        d=np.frombuffer(f.read(int(np.prod(sh))*4),np.float32).reshape(sh)
    return sh,d
for p in ['e2e_work/vid_grid.bin','e2e_work/txt.bin']:
    if os.path.exists(p):
        sh,d=read_raw(p)
        print(p,'shape',sh,'range',d.min(),d.max())
    else:
        print(p,'MISSING')

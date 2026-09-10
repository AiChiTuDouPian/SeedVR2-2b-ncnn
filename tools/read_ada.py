import numpy as np, os
f = 'block0_ada_0_v_a_sc.f32'
if not os.path.exists(f):
    print('MISSING', f)
else:
    d = np.fromfile(f, np.float32)
    print(f'ada_0_v_a_sc n={d.size} range[{d.min():.4f},{d.max():.4f}] mean={d.mean():.4f}')

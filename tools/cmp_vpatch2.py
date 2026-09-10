import numpy as np, struct, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import run_e2e_ref as ref
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# engine vid_grid
with open(os.path.join(ROOT,"vid_grid.bin"),'rb') as f:
    n=struct.unpack('<Q',f.read(8))[0]
    H=struct.unpack('<i',f.read(4))[0]; W=struct.unpack('<i',f.read(4))[0]
    vg=np.frombuffer(f.read(),np.float32).reshape(H,W,33)
vg=vg[None,...]
# numpy x_patch from engine vid_grid
xp = ref.patchify(vg, 1, H, W)
print("numpy x_patch(from engine vid_grid)", xp.shape, "range", xp.min(), xp.max())
# engine vid_patch
ep = np.fromfile(os.path.join(ROOT,"vidpatch_engine.f32"), np.float32).reshape(xp.shape)
print("engine vid_patch", ep.shape, "range", ep.min(), ep.max())
cos=float(np.sum(xp*ep)/(np.linalg.norm(xp)*np.linalg.norm(ep)+1e-12))
print("cos =", cos, " maxdiff", np.abs(xp-ep).max(), " meandiff", np.abs(xp-ep).mean())
# 找到差异位置
d=np.abs(xp-ep)
idx=np.unravel_index(np.argmax(d), d.shape)
print("max diff at (token,ch)", idx, "numpy", xp[idx], "engine", ep[idx])
# 检查 engine vid_patch 是否是 vid_grid 直接 reshape（未 patchify）
raw = vg.reshape(-1,33)
ep_flat = ep.reshape(-1,132)
# engine 布局检查
print("\nengine vid_patch[0,:8]:", ep[0,:8])
print("numpy  x_patch[0,:8]:", xp[0,:8])

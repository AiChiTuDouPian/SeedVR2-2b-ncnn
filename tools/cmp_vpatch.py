import numpy as np, struct, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import run_e2e_ref as ref
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# numpy x_patch
wd = os.path.join(ROOT, "e2e_work")
vg = ref.read_raw(os.path.join(wd, "vid_grid.bin"))
with open(os.path.join(wd, "params.txt")) as f:
    T,Hg,Wg,TXT,ts=f.read().split(); T,Hg,Wg,TXT=int(T),int(Hg),int(Wg),int(TXT)
xp = ref.patchify(vg, T, Hg, Wg)
print("numpy x_patch", xp.shape, "range", xp.min(), xp.max())
# engine vid_patch (raw f32)
ep = np.fromfile(os.path.join(ROOT,"vidpatch_engine.f32"), np.float32)
print("engine vid_patch", ep.shape, "range", ep.min(), ep.max())
if ep.size==xp.size:
    ep=ep.reshape(xp.shape)
    cos=float(np.sum(xp*ep)/(np.linalg.norm(xp)*np.linalg.norm(ep)+1e-12))
    print("cos =", cos)
    d=np.abs(xp-ep)
    print("maxdiff", d.max(), "meandiff", d.mean())
    # 每 132 通道块对比（token 级）
    diff_token = np.abs(xp-ep).reshape(xp.shape[0],-1).mean(1)
    print("per-token max mean diff idx", np.argmax(diff_token), "value", diff_token.max())

import numpy as np, struct, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import run_e2e_ref as ref
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# engine vid_patch (raw f32, 1350*132)
ep = np.fromfile(os.path.join(ROOT,"vidpatch_engine.f32"), np.float32).reshape(1350,132)
# proj 权重
with ref.io.open_sd() as f:
    Wpin=ref.io.get_top_weight(f,"vid_in.proj","weight"); Bpin=ref.io.get_top_weight(f,"vid_in.proj","bias")
vid_in = ep @ Wpin.T + Bpin
print("numpy vid_in:", vid_in.shape, "range", vid_in.min(), vid_in.max())
# engine v_cur_0 (b0mid)
# 读 b0mid_v_cur_0.f32 (raw, w=2560 h=1350)
ev = np.fromfile(os.path.join(ROOT,"b0mid_v_cur_0.f32"), np.float32)
print("engine v_cur_0:", ev.shape, "range", ev.min(), ev.max())
if ev.size==vid_in.size:
    ev=ev.reshape(vid_in.shape)
    cos=float(np.sum(vid_in*ev)/(np.linalg.norm(vid_in)*np.linalg.norm(ev)+1e-12))
    print("cos(vid_in numpy, engine) =", cos)
    print("maxdiff", np.abs(vid_in-ev).max(), "meandiff", np.abs(vid_in-ev).mean())
    print("engine v_cur_0[0,:8]:", ev[0,:8])
    print("numpy  vid_in[0,:8]:", vid_in[0,:8])

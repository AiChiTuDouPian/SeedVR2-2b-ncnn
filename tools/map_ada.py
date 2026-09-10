import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import run_e2e_ref as ref
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
emb = ref.time_embedding(1000.0).reshape(-1)
Wb = ref.block_weights(0); Adv = Wb["vid"]["ada"]
# numpy 5 个 ada (vid 流): k = [1,0,2,4,3,5] for [a_sc,a_sh,a_g,m_sc,m_sh,m_g]
names = ["a_sc","a_sh","a_g","m_sc","m_sh","m_g"]
ks = [1,0,2,4,3,5]
branches = [Adv["attn_scale"],Adv["attn_shift"],Adv["attn_gate"],Adv["mlp_scale"],Adv["mlp_shift"],Adv["mlp_gate"]]
npy_ada = {}
for n,k,b in zip(names,ks,branches):
    out = emb.reshape(-1,6)[:,k] + np.asarray(b,np.float32)
    npy_ada[n] = out.astype(np.float16).astype(np.float32)
    print(f"numpy {n:6s} k={k} range {out.min():7.3f} {out.max():7.3f}")
# engine dump
print("\n--- 匹配 ---")
for n in names:
    e = np.fromfile(os.path.join(ROOT,f"b0mid_ada_0_v_{n}.f32"), np.float32)
    best = max((float(np.sum(e*npy)/(np.linalg.norm(e)*np.linalg.norm(npy)+1e-12)), n2) for n2,npy in npy_ada.items())
    print(f"engine {n:6s} best_match={best[1]:6s} cos={best[0]:.4f}  engine_range[{e.min():7.3f},{e.max():7.3f}]")

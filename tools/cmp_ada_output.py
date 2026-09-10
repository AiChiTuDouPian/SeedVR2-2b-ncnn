import numpy as np, sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import run_e2e_ref as ref
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
emb = ref.time_embedding(1000.0).reshape(-1)   # (15360,)
Wb = ref.block_weights(0); Adv = Wb["vid"]["ada"]
# AdaCompose 输出 (fp16 量化): out[d] = emb[d*6+k] + branch[d]
def adacompose(k, branch):
    out = emb.reshape(-1,6)[:,k] + np.asarray(branch, np.float32)
    return out.astype(np.float16).astype(np.float32)
a_sc = adacompose(1, Adv["attn_scale"])
a_sh = adacompose(0, Adv["attn_shift"])
m_sc = adacompose(4, Adv["mlp_scale"])   # 注意 k 索引
m_sh = adacompose(3, Adv["mlp_shift"])
m_g  = adacompose(5, Adv["mlp_gate"])
print("numpy ada a_sc range", a_sc.min(), a_sc.max())
print("numpy ada a_sh range", a_sh.min(), a_sh.max())
print("numpy ada m_sc range", m_sc.min(), m_sc.max())
print("numpy ada m_sh range", m_sh.min(), m_sh.max())
print("numpy ada m_g  range", m_g.min(), m_g.max())
# 对比 engine dump
for name, npy in [("a_sc",a_sc),("a_sh",a_sh),("m_sc",m_sc),("m_sh",m_sh),("m_g",m_g)]:
    e = np.fromfile(os.path.join(ROOT,f"b0mid_ada_0_v_{name}.f32"), np.float32)
    cos=float(np.sum(npy*e)/(np.linalg.norm(npy)*np.linalg.norm(e)+1e-12))
    print(f"engine ada_0_v_{name} range {e.min():.3f} {e.max():.3f}  cos vs numpy={cos:.4f}  maxdiff={np.abs(npy-e).max():.3f}")

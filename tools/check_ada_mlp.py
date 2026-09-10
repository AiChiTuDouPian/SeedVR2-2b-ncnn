import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import run_e2e_ref as ref
emb = ref.time_embedding(1000.0); emb3=emb.reshape(1,2560,2,3)
Wb = ref.block_weights(0); Adv=Wb["vid"]["ada"]
# mlp scale: scAm + scBm
scAm = emb3[0,:,1,1]   # (D,2,3) 第二组 mlp, 中间=scale
scBm = Adv["mlp_scale"]
m_sc = (scAm + scBm).astype(np.float16).astype(np.float32)
print("numpy m_sc range", m_sc.min(), m_sc.max(), "mean", m_sc.mean())
# attn scale
scA = emb3[0,:,0,1]; scB=Adv["attn_scale"]
a_sc = (scA+scB).astype(np.float16).astype(np.float32)
print("numpy a_sc range", a_sc.min(), a_sc.max(), "mean", a_sc.mean())
# mlp shift
sAm=emb3[0,:,1,0]; shBm=Adv["mlp_shift"]
m_sh=(sAm+shBm).astype(np.float16).astype(np.float32)
print("numpy m_sh range", m_sh.min(), m_sh.max())

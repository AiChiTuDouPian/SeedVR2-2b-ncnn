import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import run_e2e_ref as ref
emb = ref.time_embedding(1000.0).reshape(-1)
Wb = ref.block_weights(0); Adv = Wb["vid"]["ada"]
names = ["a_sc","a_sh","a_g","m_sc","m_sh","m_g"]
ks = [1,0,2,4,3,5]
branches = [Adv["attn_scale"],Adv["attn_shift"],Adv["attn_gate"],Adv["mlp_scale"],Adv["mlp_shift"],Adv["mlp_gate"]]
for n,k,b in zip(names,ks,branches):
    out = emb.reshape(-1,6)[:,k] + np.asarray(b,np.float32)
    out = out.astype(np.float16).astype(np.float32)
    print(f"numpy {n:6s} first3 = {out[0]:.4f} {out[1]:.4f} {out[2]:.4f}")
# engine adavi 最后一次 top_blobs 值 (vi=0,2,3,4,5)
print("\nengine top_blobs[0](vi=0): 3.17 3.31 2.17")
print("engine top_blobs[2](vi=2): 0.35 0.35 -0.36")
print("engine top_blobs[3](vi=3): 0.88 1.06 0.81")
print("engine top_blobs[4](vi=4): 0.02 0.01 -0.01")
print("engine top_blobs[5](vi=5): -0.82 0.64 -0.66")

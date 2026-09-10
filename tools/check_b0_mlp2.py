import numpy as np, sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import run_e2e_ref as ref
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
emb = ref.time_embedding(1000.0); emb3=emb.reshape(1,2560,2,3)
Wb = ref.block_weights(0); Adv=Wb["vid"]["ada"]
# 用 engine 的 v_cur_0_a (attn 残差后) 作为 mlp 输入
vid = np.fromfile(os.path.join(ROOT,"b0mid_v_cur_0_a.f32"), np.float32).reshape(1350,2560)
print("mlp 输入 (engine v_cur_0_a) range", vid.min(), vid.max())
# numpy mlp
vid_mn = ref.rmsnorm(vid)
sAm,scAm,gAm = emb3[0,:,1,0],emb3[0,:,1,1],emb3[0,:,1,2]
shBm=Adv["mlp_shift"];scBm=Adv["mlp_scale"];gBm=Adv["mlp_gate"]
vid_mn=vid_mn*(scAm+scBm)+(sAm+shBm)
print("vid_mn (mlp ada) range", vid_mn.min(), vid_mn.max())
vid_h = ref.swiglu(vid_mn, Wb["vid"]["mlp_in"], Wb["vid"]["mlp_g"], Wb["vid"]["mlp_out"])
print("numpy swiglu 输出 range", vid_h.min(), vid_h.max())
# 对比 engine v_hs_0
eng_hs = np.fromfile(os.path.join(ROOT,"b0mid_v_hs_0.f32"), np.float32)
print("engine v_hs_0 range", eng_hs.min(), eng_hs.max(), "size", eng_hs.size)
if eng_hs.size==vid_h.size:
    cos=float(np.sum(vid_h*eng_hs)/(np.linalg.norm(vid_h)*np.linalg.norm(eng_hs)+1e-12))
    print("cos(swiglu numpy, engine)=", cos)
    print("maxdiff", np.abs(vid_h-eng_hs).max())
# mlp 输出
vid_m = vid_h*(gAm+gBm); vid_out = vid_m + vid
print("numpy mlp 后 (层0输出) range", vid_out.min(), vid_out.max())
eng_cur1 = np.fromfile(os.path.join(ROOT,"b0mid_v_cur_1.f32"), np.float32)
print("engine v_cur_1 range", eng_cur1.min(), eng_cur1.max())
cos2=float(np.sum(vid_out*eng_cur1)/(np.linalg.norm(vid_out)*np.linalg.norm(eng_cur1)+1e-12))
print("cos(层0输出 numpy, engine)=", cos2)

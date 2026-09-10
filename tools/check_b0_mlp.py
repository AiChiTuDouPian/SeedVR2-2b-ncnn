import numpy as np, sys, os, struct
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import run_e2e_ref as ref
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 读 engine vid_patch
ep = np.fromfile(os.path.join(ROOT,"vidpatch_engine.f32"), np.float32).reshape(1350,132)
# 读 txt
txt = ref.read_raw(os.path.join(ROOT,"e2e_work/txt.bin"))
# 层0 重算 mlp
with ref.io.open_sd() as f:
    Wpin=ref.io.get_top_weight(f,"vid_in.proj","weight"); Bpin=ref.io.get_top_weight(f,"vid_in.proj","bias")
    Wvon=ref.io.get_top_weight(f,"vid_out_norm","weight")
vid = ep @ Wpin.T + Bpin
print("vid_in range", vid.min(), vid.max())
emb = ref.time_embedding(1000.0); emb3=emb.reshape(1,2560,2,3)
Wb = ref.block_weights(0)
Adv = Wb["vid"]["ada"]
# attn 部分
vid_an = ref.rmsnorm(vid)
sA,scA,gA = emb3[0,:,0,0],emb3[0,:,0,1],emb3[0,:,0,2]
shB=Adv["attn_shift"];scB=Adv["attn_scale"];gB=Adv["attn_gate"]
vid_an=vid_an*(scA+scB)+(sA+shB)
vid_at,_ = ref.awa_forward(vid_an, None, (1,30,45), "720pwin_by_size_bysize", Wb["vid"], {"qkv":np.zeros((58,7680)),"out":np.zeros((58,2560)),"out_b":np.zeros(2560),"nq":np.zeros(128),"nk":np.zeros(128)}, 58)
vid_at = vid_at*(gA+gB)
vid = vid_at + vid
print("vid1 (attn残差) range", vid.min(), vid.max())
# mlp
vid_mn = ref.rmsnorm(vid)
sAm,scAm,gAm = emb3[0,:,1,0],emb3[0,:,1,1],emb3[0,:,1,2]
shBm=Adv["mlp_shift"];scBm=Adv["mlp_scale"];gBm=Adv["mlp_gate"]
vid_mn=vid_mn*(scAm+scBm)+(sAm+shBm)
print("vid_mn (mlp ada) range", vid_mn.min(), vid_mn.max())
vid_h = ref.swiglu(vid_mn, Wb["vid"]["mlp_in"], Wb["vid"]["mlp_g"], Wb["vid"]["mlp_out"])
print("swiglu 输出 range", vid_h.min(), vid_h.max())
vid_m = vid_h*(gAm+gBm)
vid = vid_m + vid
print("层0 输出 range", vid.min(), vid.max())
# 对比 engine
eng_cur1 = np.fromfile(os.path.join(ROOT,"b0mid_v_cur_1.f32"), np.float32)
print("engine v_cur_1 range", eng_cur1.min(), eng_cur1.max())
eng_hs = np.fromfile(os.path.join(ROOT,"b0mid_v_hs_0.f32"), np.float32)
print("engine v_hs_0 range", eng_hs.min(), eng_hs.max(), "size", eng_hs.size, "numpy swiglu size", vid_h.size)
if eng_hs.size==vid_h.size:
    cos=float(np.sum(vid_h*eng_hs)/(np.linalg.norm(vid_h)*np.linalg.norm(eng_hs)+1e-12))
    print("cos(swiglu numpy, engine v_hs_0)=", cos)

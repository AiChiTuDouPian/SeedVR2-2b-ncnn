#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比 ncnn block0 的 b0mid dump 与 numpy 参考中间量，逐层定位 ncnn 第一处数值偏离。
b0mid_*.f32 是 raw float32（无 header，block0 中间 blob dump）。
numpy 中间量由 recompute_block0.py 逻辑重算并保存到本地。
"""
import os, sys, math, struct
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.join(os.path.dirname(_HERE), "export")
sys.path.insert(0, _HERE)
sys.path.insert(0, _EXP)
import ncnn_io as io
import awa_window, mmrope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M5 = os.path.join(ROOT, "models", "m5")

HEADS=20; HEAD_D=128; DIM=2560; QKV=HEADS*HEAD_D*3
EPS=1e-5; MM_LAYERS=10; WINDOW=(4,3,3); TXT_LEN=58; TIMESTEP=1000.0
H8, W8 = 60, 90
T, H, Wd = 1, H8//2, W8//2
Lv = T*H*Wd

def f16(w):
    if w is None: return None
    return w.astype(np.float16).astype(np.float32)

def rmsnorm(x, w=None):
    ms=np.mean(x*x,axis=-1,keepdims=True); y=x/np.sqrt(ms+EPS)
    return y*w if w is not None else y

def sinusoidal_embedding(t,dim):
    half=dim//2
    freqs=np.exp(-np.log(10000.0)*np.arange(half,dtype=np.float64)/half)
    args=t.astype(np.float64)[:,None]*freqs[None,:]
    return np.concatenate([np.sin(args),np.cos(args)],axis=-1).astype(np.float32)

def time_embedding(t):
    with io.open_sd() as f:
        Wpi=f16(io.get_top_weight(f,"emb_in.proj_in","weight"));Bpi=f16(io.get_top_weight(f,"emb_in.proj_in","bias"))
        Wph=f16(io.get_top_weight(f,"emb_in.proj_hid","weight"));Bph=f16(io.get_top_weight(f,"emb_in.proj_hid","bias"))
        Wpo=f16(io.get_top_weight(f,"emb_in.proj_out","weight"));Bpo=f16(io.get_top_weight(f,"emb_in.proj_out","bias"))
    e=sinusoidal_embedding(np.array([t],dtype=np.float32),256)
    e=e@Wpi.T+Bpi;e=e/(1+np.exp(-e))
    e=e@Wph.T+Bph;e=e/(1+np.exp(-e))
    e=e@Wpo.T+Bpo
    return e

def ada_params(f,block,branch):
    out={}
    for layer in ["attn","mlp"]:
        for nm in ["shift","scale","gate"]:
            out[f"{layer}_{nm}"]=f16(io.load_key(f,f"blocks.{block}.ada.{branch}.{layer}_{nm}"))
    return out

def load_nc_raw(name, dim=None):
    """读 b0mid_<name>.f32（raw float32）。dim 给定时 reshape (n, dim)，否则按文件大小推断。"""
    p = os.path.join(ROOT, f"b0mid_{name}.f32")
    if not os.path.exists(p): return None
    d = np.fromfile(p, np.float32)
    if dim is None:
        if d.size % DIM == 0 and d.size != Lv*132: return d.reshape(-1, DIM)
        if d.size % 132 == 0: return d.reshape(-1, 132)
        return d
    return d.reshape(-1, dim)

def cosim(a,b):
    a=a.ravel(); b=b.ravel()
    return float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def cmp(tag, nc, ref):
    if nc is None: print(f"  [skip] {tag}: 无 ncnn dump"); return
    c = cosim(nc, ref)
    d = np.abs(nc-ref)
    print(f"  [{tag}] cos={c:.6f} maxdiff={d.max():.6f}  ncnn_range[{nc.min():.4f},{nc.max():.4f}] ref_range[{ref.min():.4f},{ref.max():.4f}]")

# ---------- 输入 ----------
with open(os.path.join(ROOT,"vid_grid.bin"),"rb") as fp:
    struct.unpack("<Q",fp.read(8))[0]
    gH = struct.unpack("<i",fp.read(4))[0]; gW = struct.unpack("<i",fp.read(4))[0]
    vid_grid = np.frombuffer(fp.read(),np.float32).reshape(gH,gW,33)
x_patch = np.zeros((Lv, 132), np.float32)
for hh in range(H):
    for ww in range(Wd):
        for ph in range(2):
            for pw in range(2):
                for c in range(33):
                    gi = ((hh*2+ph)*gW + (ww*2+pw))*33 + c
                    pi = (hh*Wd + ww)*132 + ((ph*2+pw)*33 + c)
                    x_patch.flat[pi] = vid_grid.flat[gi]
x_txt = np.fromfile(os.path.join(M5,"txt.bin"), dtype=np.float32)
n_txt = 5120*TXT_LEN
if len(x_txt) >= n_txt+16: x_txt = x_txt[16:16+n_txt]
x_txt = x_txt[:n_txt].reshape(TXT_LEN,5120)

# ---------- block0 权重 ----------
block=0
with io.open_sd() as f:
    Wpin=f16(io.get_top_weight(f,"vid_in.proj","weight"));Bpin=f16(io.get_top_weight(f,"vid_in.proj","bias"))
    Wtxt=f16(io.get_top_weight(f,"txt_in","weight"));Btxt=f16(io.get_top_weight(f,"txt_in","bias"))
    def L(sub,branch,name,sp=None):
        return f16(io.load_key(f, io.block_key(block,sub,branch,name,shared_pattern=sp)))
    W={}
    for br,slot in [("vid","vid"),("txt","txt")]:
        W[slot]={"qkv":L("proj_qkv",br,"weight"),"out":L("proj_out",br,"weight"),"out_b":L("proj_out",br,"bias"),
                 "nq":L("norm_q",br,"weight"),"nk":L("norm_k",br,"weight"),
                 "mlp_in":L("proj_in",br,"weight",sp="mlp"),"mlp_g":L("proj_in_gate",br,"weight",sp="mlp"),
                 "mlp_out":L("proj_out",br,"weight",sp="mlp"),"mlp_out_b":L("proj_out",br,"bias",sp="mlp"),
                 "ada":ada_params(f,block,br)}

vid=x_patch@Wpin.T+Bpin
txt=x_txt@Wtxt.T+Btxt
emb=time_embedding(TIMESTEP)
emb3=emb.reshape(1,DIM,2,3)

print("===== 输入层 =====")
nc_in0 = load_nc_raw("in_vid0")
if nc_in0 is not None and nc_in0.size == x_patch.size:
    cmp("in_vid0 (patch输入)", nc_in0.reshape(-1,132), x_patch)
cmp("v_cur_0 (proj输出=vid)", load_nc_raw("v_cur_0"), vid)

# ---------- attn norm + ada ----------
vid_an=rmsnorm(vid)
sA,scA,gA=emb3[0,:,0,0],emb3[0,:,0,1],emb3[0,:,0,2]
shB=W["vid"]["ada"]["attn_shift"];scB=W["vid"]["ada"]["attn_scale"];gB=W["vid"]["ada"]["attn_gate"]
vid_an_m=vid_an*(scA+scB)+(sA+shB)
print("===== attn 前 =====")
cmp("v_rn_0 (rmsnorm)", load_nc_raw("v_rn_0"), vid_an)
cmp("v_m2_0 (ada调制)", load_nc_raw("v_m2_0"), vid_an_m)

# ---------- ada 向量 ----------
def load_nc_vec(name):
    p = os.path.join(ROOT, f"b0mid_{name}.f32")
    if not os.path.exists(p): return None
    return np.fromfile(p, np.float32)
for nm, kidx in [("a_sc",1),("a_sh",0),("a_g",2),("m_sc",4),("m_sh",3),("m_g",5)]:
    wname = "attn_scale" if nm=="a_sc" else "attn_shift" if nm=="a_sh" else "attn_gate" if nm=="a_g" else "mlp_scale" if nm=="m_sc" else "mlp_shift" if nm=="m_sh" else "mlp_gate"
    ref_ada = (emb.reshape(DIM,6)[:, kidx] + W["vid"]["ada"][wname])
    nc_ada = load_nc_vec(f"ada_0_v_{nm}")
    if nc_ada is not None:
        print(f"  [ada_0_v_{nm}] cos={cosim(nc_ada, ref_ada):.6f}  ncnn[0:3]={nc_ada[:3]} ref[0:3]={ref_ada[:3]}")

# ---------- qkv ----------
vqkv=(vid_an_m@W["vid"]["qkv"].T).reshape(Lv,3,HEADS,HEAD_D)
print("===== qkv =====")
cmp("v_qkv_0 (qkv raw)", load_nc_raw("v_qkv_0", 7680), vqkv.reshape(Lv,-1))

# ---------- attn 输出（未 proj） ----------
# numpy attn 输出 vout_rev (Lv,DIM) —— 重新计算
windows=awa_window.make_720Pwindows_bysize((T,H,Wd),WINDOW)
nwin=len(windows); win_shapes=[]; f_list=[]
for (st,sh,sw) in windows:
    wt=st.stop-st.start;wh=sh.stop-sh.start;ww=sw.stop-sw.start
    win_shapes.append((wt,wh,ww));f_list.append(wt*wh*ww)
vid_freq,txt_freq=mmrope.build_window_freqs(win_shapes,TXT_LEN)
vq,vk,vv=vqkv[:,0],vqkv[:,1],vqkv[:,2]
vq=rmsnorm(vq,W["vid"]["nq"]);vk=rmsnorm(vk,W["vid"]["nk"])
vq_win=vq.reshape(T,H,Wd,HEADS,HEAD_D);vk_win=vk.reshape(T,H,Wd,HEADS,HEAD_D);vv_win=vv.reshape(T,H,Wd,HEADS,HEAD_D)
vqw_list,vkw_list,vvw_list=[],[],[]
for (st,sh,sw) in windows:
    vqw_list.append(vq_win[st,sh,sw].reshape(-1,HEADS,HEAD_D))
    vkw_list.append(vk_win[st,sh,sw].reshape(-1,HEADS,HEAD_D))
    vvw_list.append(vv_win[st,sh,sw].reshape(-1,HEADS,HEAD_D))
vqw=np.concatenate(vqw_list,0);vkw=np.concatenate(vkw_list,0);vvw=np.concatenate(vvw_list,0)
vq_w=np.transpose(vqw,(1,0,2));vk_w=np.transpose(vkw,(1,0,2));vv_w=np.transpose(vvw,(1,0,2))
vq_w=mmrope.apply_rotary_emb(vid_freq,vq_w);vk_w=mmrope.apply_rotary_emb(vid_freq,vk_w)
# txt
tqkv=(rmsnorm(txt)*(scA+W["txt"]["ada"]["attn_scale"])+(sA+W["txt"]["ada"]["attn_shift"])).reshape(TXT_LEN,-1)@W["txt"]["qkv"].T
tqkv=tqkv.reshape(TXT_LEN,3,HEADS,HEAD_D)
tq,tk,tv=tqkv[:,0],tqkv[:,1],tqkv[:,2]
tq=rmsnorm(tq,W["txt"]["nq"]);tk=rmsnorm(tk,W["txt"]["nk"])
tq_rep=np.tile(tq,(nwin,1,1));tk_rep=np.tile(tk,(nwin,1,1));tv_rep=np.tile(tv,(nwin,1,1))
tq_w=np.transpose(tq_rep,(1,0,2));tk_w=np.transpose(tk_rep,(1,0,2));tv_w=np.transpose(tv_rep,(1,0,2))
tq_w=mmrope.apply_rotary_emb(txt_freq,tq_w);tk_w=mmrope.apply_rotary_emb(txt_freq,tk_w)
out_v_seg=[];vg=0
for wi in range(nwin):
    f_i=f_list[wi]
    q=np.concatenate([vq_w[:,vg:vg+f_i,:],tq_w[:,wi*TXT_LEN:(wi+1)*TXT_LEN,:]],1)
    k=np.concatenate([vk_w[:,vg:vg+f_i,:],tk_w[:,wi*TXT_LEN:(wi+1)*TXT_LEN,:]],1)
    v=np.concatenate([vv_w[:,vg:vg+f_i,:],tv_w[:,wi*TXT_LEN:(wi+1)*TXT_LEN,:]],1)
    vg+=f_i
    S=q.shape[1];o=np.empty((HEADS,S,HEAD_D),np.float32);scale=1.0/math.sqrt(HEAD_D)
    for hh in range(HEADS):
        qq=q[hh];kk=k[hh];vvh=v[hh]
        sc=(qq@kk.T)*scale;sc=sc-sc.max(-1,keepdims=True)
        p=np.exp(sc);p=p/p.sum(-1,keepdims=True)
        o[hh]=p@vvh
    out_v_seg.append(o[:,:f_i,:])
vout_target=np.zeros((T,H,Wd,HEADS,HEAD_D),np.float32)
for wi,(st,sh,sw) in enumerate(windows):
    wt=st.stop-st.start;wh=sh.stop-sh.start;ww=sw.stop-sw.start
    blk=np.transpose(out_v_seg[wi],(1,0,2)).reshape(wt,wh,ww,HEADS,HEAD_D)
    vout_target[st,sh,sw]=blk
vout_rev=vout_target.reshape(Lv,DIM)
print("===== attn 输出 =====")
cmp("v_attn_0 (attn out)", load_nc_raw("v_attn_0"), vout_rev)
vout_rev.astype(np.float32, copy=False).tofile(os.path.join(ROOT,"b0mid_v_attn_0_ref.f32"))

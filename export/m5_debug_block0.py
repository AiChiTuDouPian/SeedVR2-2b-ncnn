#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""仅重算 block 0 的逐子层中间量，供 verify_dit 分诊 AWA 分歧点。
直接读 safetensors（无 pnnx），很快。"""
import os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ncnn_io as io
import awa_window, mmrope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M5 = os.path.join(ROOT, "models", "m5")

HEADS=20; HEAD_D=128; DIM=2560; QKV=HEADS*HEAD_D*3
EPS=1e-5; MM_LAYERS=10; WINDOW=(4,3,3); TXT_LEN=8; TIMESTEP=500.0
GRID=(2,40,40); T,H,Wd=GRID; Lv=T*H*Wd

def save_raw(path, arr):
    arr=np.ascontiguousarray(arr,dtype=np.float32)
    with open(path,"wb") as fp:
        fp.write(np.int64(arr.ndim).tobytes())
        for d in arr.shape: fp.write(np.int64(d).tobytes())
        fp.write(arr.tobytes())

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

def f16(w):
    """模拟 ncnn fp16 权重的量化误差：safetensors float32 -> fp16 -> float32。"""
    if w is None: return None
    return w.astype(np.float16).astype(np.float32)

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
                 "mlp_out":L("proj_out",br,"weight",sp="mlp"),"ada":ada_params(f,block,br)}

rng=np.random.RandomState(20240713)
x_patch=rng.randn(Lv,33*1*2*2).astype(np.float32)
x_txt=rng.randn(TXT_LEN,5120).astype(np.float32)
vid=x_patch@Wpin.T+Bpin
txt=x_txt@Wtxt.T+Btxt
emb=time_embedding(TIMESTEP)
emb3=emb.reshape(1,DIM,2,3)

window_method="720pwin_by_size_bysize"
windows=awa_window.make_720Pwindows_bysize(GRID,WINDOW)
nwin=len(windows); win_shapes=[]; f_list=[]
for (st,sh,sw) in windows:
    wt=st.stop-st.start;wh=sh.stop-sh.start;ww=sw.stop-sw.start
    win_shapes.append((wt,wh,ww));f_list.append(wt*wh*ww)
M=sum(f_list)
vid_freq,txt_freq=mmrope.build_window_freqs(win_shapes,TXT_LEN)

# attn_norm
vid_an=rmsnorm(vid);txt_an=rmsnorm(txt)
sA,scA,gA=emb3[0,:,0,0],emb3[0,:,0,1],emb3[0,:,0,2]
shB=W["vid"]["ada"]["attn_shift"];scB=W["vid"]["ada"]["attn_scale"];gB=W["vid"]["ada"]["attn_gate"]
thB=W["txt"]["ada"]["attn_shift"];tcB=W["txt"]["ada"]["attn_scale"];tgB=W["txt"]["ada"]["attn_gate"]
vid_an=vid_an*(scA+scB)+(sA+shB)
txt_an=txt_an*(scA+tcB)+(sA+thB)
save_raw(os.path.join(M5,"dbg_vid_an.bin"),vid_an)
save_raw(os.path.join(M5,"dbg_txt_an.bin"),txt_an)

# proj_qkv + qk_norm
Wv=W["vid"];Wt=W["txt"]
vqkv=(vid_an@Wv["qkv"].T).reshape(Lv,3,HEADS,HEAD_D)
tqkv=(txt_an@Wt["qkv"].T).reshape(TXT_LEN,3,HEADS,HEAD_D)
vq,vk,vv=vqkv[:,0],vqkv[:,1],vqkv[:,2]
tq,tk,tv=tqkv[:,0],tqkv[:,1],tqkv[:,2]
vq=rmsnorm(vq,Wv["nq"]);vk=rmsnorm(vk,Wv["nk"])
tq=rmsnorm(tq,Wt["nq"]);tk=rmsnorm(tk,Wt["nk"])
save_raw(os.path.join(M5,"dbg_vq.bin"),vq.reshape(Lv,DIM))
save_raw(os.path.join(M5,"dbg_vk.bin"),vk.reshape(Lv,DIM))
save_raw(os.path.join(M5,"dbg_vv.bin"),vv.reshape(Lv,DIM))

# partition
vq_win=vq.reshape(T,H,Wd,HEADS,HEAD_D);vk_win=vk.reshape(T,H,Wd,HEADS,HEAD_D);vv_win=vv.reshape(T,H,Wd,HEADS,HEAD_D)
vqw_list,vkw_list,vvw_list=[],[],[]
for (st,sh,sw) in windows:
    vqw_list.append(vq_win[st,sh,sw].reshape(-1,HEADS,HEAD_D))
    vkw_list.append(vk_win[st,sh,sw].reshape(-1,HEADS,HEAD_D))
    vvw_list.append(vv_win[st,sh,sw].reshape(-1,HEADS,HEAD_D))
vqw=np.concatenate(vqw_list,0);vkw=np.concatenate(vkw_list,0);vvw=np.concatenate(vvw_list,0)
save_raw(os.path.join(M5,"dbg_vqw.bin"),vqw.reshape(M,DIM))
save_raw(os.path.join(M5,"dbg_vkw.bin"),vkw.reshape(M,DIM))
save_raw(os.path.join(M5,"dbg_vvw.bin"),vvw.reshape(M,DIM))

# RoPE
vq_w=np.transpose(vqw,(1,0,2));vk_w=np.transpose(vkw,(1,0,2));vv_w=np.transpose(vvw,(1,0,2))
vq_w=mmrope.apply_rotary_emb(vid_freq,vq_w);vk_w=mmrope.apply_rotary_emb(vid_freq,vk_w)
save_raw(os.path.join(M5,"dbg_vqw_rope.bin"),np.transpose(vq_w,(1,0,2)).reshape(M,DIM))
save_raw(os.path.join(M5,"dbg_vkw_rope.bin"),np.transpose(vk_w,(1,0,2)).reshape(M,DIM))

# SDPA + reverse
tq_rep=np.tile(tq,(nwin,1,1));tk_rep=np.tile(tk,(nwin,1,1));tv_rep=np.tile(tv,(nwin,1,1))
tq_w=np.transpose(tq_rep,(1,0,2));tk_w=np.transpose(tk_rep,(1,0,2));tv_w=np.transpose(tv_rep,(1,0,2))
tq_w=mmrope.apply_rotary_emb(txt_freq,tq_w);tk_w=mmrope.apply_rotary_emb(txt_freq,tk_w)
out_v_seg,out_t_seg=[],[];vg=0
for wi in range(nwin):
    f_i=f_list[wi]
    q=np.concatenate([vq_w[:,vg:vg+f_i,:],tq_w[:,wi*TXT_LEN:(wi+1)*TXT_LEN,:]],1)
    k=np.concatenate([vk_w[:,vg:vg+f_i,:],tk_w[:,wi*TXT_LEN:(wi+1)*TXT_LEN,:]],1)
    v=np.concatenate([vv_w[:,vg:vg+f_i,:],tv_w[:,wi*TXT_LEN:(wi+1)*TXT_LEN,:]],1)
    vg+=f_i
    S=q.shape[1];o=np.empty((HEADS,S,HEAD_D),np.float32);scale=1.0/math.sqrt(HEAD_D)
    for hh in range(HEADS):
        qq=q[hh];kk=k[hh];vv=v[hh]
        sc=(qq@kk.T)*scale;sc=sc-sc.max(-1,keepdims=True)
        p=np.exp(sc);p=p/p.sum(-1,keepdims=True)
        o[hh]=p@vv
    out_v_seg.append(o[:,:f_i,:]);out_t_seg.append(o[:,f_i:,:])
vout_target=np.zeros((T,H,Wd,HEADS,HEAD_D),np.float32)
for wi,(st,sh,sw) in enumerate(windows):
    wt=st.stop-st.start;wh=sh.stop-sh.start;ww=sw.stop-sw.start
    blk=np.transpose(out_v_seg[wi],(1,0,2)).reshape(wt,wh,ww,HEADS,HEAD_D)
    vout_target[st,sh,sw]=blk
vout_rev=vout_target.reshape(Lv,DIM)
tout=np.stack(out_t_seg,0).mean(0);tout=np.transpose(tout,(1,0,2)).reshape(TXT_LEN,DIM)
save_raw(os.path.join(M5,"dbg_vout_rev.bin"),vout_rev)
save_raw(os.path.join(M5,"dbg_tout.bin"),tout)

# proj_out
vid_awa=vout_rev@Wv["out"].T+Wv["out_b"]
txt_awa=tout@Wt["out"].T+Wt["out_b"]
save_raw(os.path.join(M5,"dbg_vid_awa.bin"),vid_awa)   # AWA 输出（ada out 之前）
save_raw(os.path.join(M5,"dbg_txt_awa.bin"),txt_awa)

# ada attn out + residual
vid_at=vid_awa*(gA+gB);txt_at=txt_awa*(gA+tgB)
vid1=vid_at+vid;txt1=txt_at+txt
save_raw(os.path.join(M5,"dbg_vid_b0_attn.bin"),vid1)  # attn 子层后

print(f"[ok] block0 debug 导出: M={M} nwin={nwin} Lv={Lv}")

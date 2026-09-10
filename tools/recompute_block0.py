#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用真实 e2e 输入重算 block0 完整输出，对比 ncnn block0_vid.bin。
输入源：ncnn dump 的 vid_grid.bin(60x90x33 channel-last) + models/m5/txt.bin(58,5120)。
窗口：与 ncnn engine 一致 make_win(1, token_h=30, token_w=45, {4,3,3}, TXT=58, shifted=false)。
timestep=1000（engine 用 TIMESTEP=1000）。
输出：与 block0_vid.bin(1350x2560) 对比 cos/maxdiff，并 dump 各子层中间量供进一步定位。
"""
import os, sys, math
import numpy as np
import struct
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
# 真实 token 网格（480p）：H8=60, W8=90 -> token_h=30, token_w=45
H8, W8 = 60, 90
T, H, Wd = 1, H8//2, W8//2   # (1, 30, 45)
Lv = T*H*Wd

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
    if w is None: return None
    return w.astype(np.float16).astype(np.float32)

# ---------- 载入真实输入 ----------
# vid_grid.bin: header(Q,H,W) + (H*W*33) channel-last
with open(os.path.join(ROOT,"vid_grid.bin"),"rb") as fp:
    n = struct.unpack("<Q",fp.read(8))[0]
    gH = struct.unpack("<i",fp.read(4))[0]
    gW = struct.unpack("<i",fp.read(4))[0]
    vid_grid = np.frombuffer(fp.read(),np.float32).reshape(gH,gW,33)
print(f"vid_grid: H={gH} W={gW} ch=33, range {vid_grid.min():.4f} {vid_grid.max():.4f}")

# patchify -> (Lv, 33*2*2=132)，顺序 (ph,pw,c)，与 DitVk::patchify_grid 一致
x_patch = np.zeros((Lv, 132), np.float32)
# 与 DitVk::patchify_grid 一致：flat 索引 gi，输出 flat 索引 pi
for hh in range(H):
    for ww in range(Wd):
        for ph in range(2):
            for pw in range(2):
                for c in range(33):
                    gi = ((hh*2+ph)*gW + (ww*2+pw))*33 + c
                    pi = (hh*Wd + ww)*132 + ((ph*2+pw)*33 + c)
                    x_patch.flat[pi] = vid_grid.flat[gi]
print(f"x_patch: {x_patch.shape} range {x_patch.min():.4f} {x_patch.max():.4f}")

# txt: models/m5/txt.bin (58,5120) raw
x_txt = np.fromfile(os.path.join(M5,"txt.bin"), dtype=np.float32)
print(f"txt.bin floats={len(x_txt)}")
# engine 读 txt.bin 用 read_raw（header+shape+data）；但这里 txt.bin 可能无 header。
# 尝试两种：若无 header 则直接 reshape
n_txt = 5120 * TXT_LEN
if len(x_txt) >= n_txt + 16:
    x_txt = x_txt[16:16+n_txt]  # 有 header 则跳过
x_txt = x_txt[:n_txt].reshape(TXT_LEN, 5120)
print(f"x_txt: {x_txt.shape} range {x_txt.min():.4f} {x_txt.max():.4f}")

# ---------- 载入 block0 权重（双分支） ----------
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
print(f"vid_in: {vid.shape} range {vid.min():.4f} {vid.max():.4f}")
print(f"txt_in: {txt.shape} range {txt.min():.4f} {txt.max():.4f}")
print(f"emb range: {emb.min():.4f} {emb.max():.4f}")

# ---------- 窗口（nonshifted） ----------
windows=awa_window.make_720Pwindows_bysize((T,H,Wd),WINDOW)
nwin=len(windows); win_shapes=[]; f_list=[]
for (st,sh,sw) in windows:
    wt=st.stop-st.start;wh=sh.stop-sh.start;ww=sw.stop-sw.start
    win_shapes.append((wt,wh,ww));f_list.append(wt*wh*ww)
M=sum(f_list)
vid_freq,txt_freq=mmrope.build_window_freqs(win_shapes,TXT_LEN)
print(f"windows: nwin={nwin} M={M} shapes={win_shapes[:4]}")

# ---------- attn_norm ----------
vid_an=rmsnorm(vid);txt_an=rmsnorm(txt)
sA,scA,gA=emb3[0,:,0,0],emb3[0,:,0,1],emb3[0,:,0,2]
shB=W["vid"]["ada"]["attn_shift"];scB=W["vid"]["ada"]["attn_scale"];gB=W["vid"]["ada"]["attn_gate"]
thB=W["txt"]["ada"]["attn_shift"];tcB=W["txt"]["ada"]["attn_scale"];tgB=W["txt"]["ada"]["attn_gate"]
vid_an=vid_an*(scA+scB)+(sA+shB)
txt_an=txt_an*(scA+tcB)+(sA+thB)
save_raw(os.path.join(M5,"r_dbgr_vid_an.bin"),vid_an)

# ---------- proj_qkv + qk_norm ----------
Wv=W["vid"];Wt=W["txt"]
vqkv=(vid_an@Wv["qkv"].T).reshape(Lv,3,HEADS,HEAD_D)
tqkv=(txt_an@Wt["qkv"].T).reshape(TXT_LEN,3,HEADS,HEAD_D)
vq,vk,vv=vqkv[:,0],vqkv[:,1],vqkv[:,2]
tq,tk,tv=tqkv[:,0],tqkv[:,1],tqkv[:,2]
vq=rmsnorm(vq,Wv["nq"]);vk=rmsnorm(vk,Wv["nk"])
tq=rmsnorm(tq,Wt["nq"]);tk=rmsnorm(tk,Wt["nk"])

# ---------- partition ----------
vq_win=vq.reshape(T,H,Wd,HEADS,HEAD_D);vk_win=vk.reshape(T,H,Wd,HEADS,HEAD_D);vv_win=vv.reshape(T,H,Wd,HEADS,HEAD_D)
vqw_list,vkw_list,vvw_list=[],[],[]
for (st,sh,sw) in windows:
    vqw_list.append(vq_win[st,sh,sw].reshape(-1,HEADS,HEAD_D))
    vkw_list.append(vk_win[st,sh,sw].reshape(-1,HEADS,HEAD_D))
    vvw_list.append(vv_win[st,sh,sw].reshape(-1,HEADS,HEAD_D))
vqw=np.concatenate(vqw_list,0);vkw=np.concatenate(vkw_list,0);vvw=np.concatenate(vvw_list,0)

# ---------- RoPE ----------
vq_w=np.transpose(vqw,(1,0,2));vk_w=np.transpose(vkw,(1,0,2));vv_w=np.transpose(vvw,(1,0,2))
vq_w=mmrope.apply_rotary_emb(vid_freq,vq_w);vk_w=mmrope.apply_rotary_emb(vid_freq,vk_w)

# ---------- SDPA + reverse ----------
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
        qq=q[hh];kk=k[hh];vvh=v[hh]
        sc=(qq@kk.T)*scale;sc=sc-sc.max(-1,keepdims=True)
        p=np.exp(sc);p=p/p.sum(-1,keepdims=True)
        o[hh]=p@vvh
    out_v_seg.append(o[:,:f_i,:]);out_t_seg.append(o[:,f_i:,:])
vout_target=np.zeros((T,H,Wd,HEADS,HEAD_D),np.float32)
for wi,(st,sh,sw) in enumerate(windows):
    wt=st.stop-st.start;wh=sh.stop-sh.start;ww=sw.stop-sw.start
    blk=np.transpose(out_v_seg[wi],(1,0,2)).reshape(wt,wh,ww,HEADS,HEAD_D)
    vout_target[st,sh,sw]=blk
vout_rev=vout_target.reshape(Lv,DIM)
tout=np.stack(out_t_seg,0).mean(0);tout=np.transpose(tout,(1,0,2)).reshape(TXT_LEN,DIM)

# ---------- proj_out + ada attn out + residual ----------
vid_awa=vout_rev@Wv["out"].T+Wv["out_b"]
txt_awa=tout@Wt["out"].T+Wt["out_b"]
vid_at=vid_awa*(gA+gB);txt_at=txt_awa*(gA+tgB)
vid1=vid_at+vid
txt1=txt_at+txt
save_raw(os.path.join(M5,"r_dbgr_vid_b0_attn.bin"),vid1)

# ---------- mlp（SwiGLU + ada mlp + residual） ----------
def add_bias(x, b):
    return x + b if b is not None else x
vid_an2=rmsnorm(vid1)
shM=W["vid"]["ada"]["mlp_shift"];scM=W["vid"]["ada"]["mlp_scale"];gM=W["vid"]["ada"]["mlp_gate"]
thM=W["txt"]["ada"]["mlp_shift"];tcM=W["txt"]["ada"]["mlp_scale"];tgM=W["txt"]["ada"]["mlp_gate"]
m_sA,m_scA,m_gA=emb3[0,:,1,0],emb3[0,:,1,1],emb3[0,:,1,2]   # (D,2,3) 第二组 = mlp
vid_mn=vid_an2*(m_scA+scM)+(m_sA+shM)
txt_mn=rmsnorm(txt1)*(m_scA+tcM)+(m_sA+thM)
# SwiGLU: gate * up
vid_gate=vid_mn@Wv["mlp_g"].T
vid_up=vid_mn@Wv["mlp_in"].T
vid_up=vid_up/(1+np.exp(-vid_up))   # silu
vid_h=vid_gate*vid_up
vid_mlp=add_bias(vid_h@Wv["mlp_out"].T, Wv["mlp_out_b"])
vid2=vid_mlp*(m_gA+gM)+vid1
save_raw(os.path.join(M5,"r_dbgr_vid_b0_full.bin"),vid2)

txt_gate=txt_mn@Wt["mlp_g"].T
txt_up=txt_mn@Wt["mlp_in"].T
txt_up=txt_up/(1+np.exp(-txt_up))
txt_h=txt_gate*txt_up
txt_mlp=add_bias(txt_h@Wt["mlp_out"].T, Wt["mlp_out_b"])
txt2=txt_mlp*(m_gA+tgM)+txt1

# ---------- 对比 ncnn block0_vid.bin ----------
nc = np.fromfile(os.path.join(ROOT,"block0_vid.bin"), dtype=np.float32).reshape(Lv,DIM)
def cosim(a,b):
    return float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)))
print("\n=========== block0 完整输出对比 ===========")
print(f"  numpy vid2: shape {vid2.shape} range {vid2.min():.4f} {vid2.max():.4f}")
print(f"  ncnn block0: shape {nc.shape} range {nc.min():.4f} {nc.max():.4f}")
print(f"  cos = {cosim(vid2, nc):.8f}")
d=np.abs(vid2-nc)
print(f"  max|diff| = {d.max():.6f}  mean = {d.mean():.6f}  RMS = {np.sqrt((d**2).mean()):.6f}")
print("  attn 后 cos =", cosim(vid1.reshape(-1), 0*vid1.reshape(-1)) if False else "")

print("\n[ok] block0 真实输入重算完成")

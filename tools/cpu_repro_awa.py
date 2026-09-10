#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CPU 复现 awa.comp（逐 query + online softmax），对比 run_e2e_ref GT。
定位 shader 的 head 16 norm 偏大是逻辑 bug 还是 GPU 执行 bug。"""
import os, sys, math
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "export"))
import run_e2e_ref as R
import ncnn_io as io
import mmrope, awa_window

HD=128; HEADS=20; DIM=2560; ROPE_ROT=126
wd = os.path.join(ROOT, "e2e_work")
vid_grid = R.read_raw(os.path.join(wd, "vid_grid.bin")); txt = R.read_raw(os.path.join(wd, "txt.bin"))
with open(os.path.join(wd, "params.txt")) as fp:
    T,Hg,Wg,TXT_LEN,ts = fp.read().split()
    T,Hg,Wg,TXT_LEN=int(T),int(Hg),int(Wg),int(TXT_LEN); ts=float(ts)
H2,W2=Hg//2,Wg//2
x_patch=R.patchify(vid_grid,T,Hg,Wg); x_txt=txt.astype(np.float32)
with io.open_sd() as f:
    Wpin=io.get_top_weight(f,"vid_in.proj","weight"); Bpin=io.get_top_weight(f,"vid_in.proj","bias")
    Wtxt=io.get_top_weight(f,"txt_in","weight"); Btxt=io.get_top_weight(f,"txt_in","bias")
vid=x_patch@Wpin.T+Bpin; txt_t=x_txt@Wtxt.T+Btxt
emb=R.time_embedding(ts); emb3=emb.reshape(1,DIM,2,3)
Wb=R.block_weights(0); Wv,Wt=Wb["vid"],Wb["txt"]; Adv,Adt=Wv["ada"],Wt["ada"]
vid_an=R.rmsnorm(vid); txt_an=R.rmsnorm(txt_t)
sA,scA,gA=emb3[0,:,0,0],emb3[0,:,0,1],emb3[0,:,0,2]
vid_an=vid_an*(scA+Adv["attn_scale"])+(sA+Adv["attn_shift"])
txt_an=txt_an*(scA+Adt["attn_scale"])+(sA+Adt["attn_shift"])
windows=awa_window.make_720Pwindows_bysize((T,H2,W2),(4,3,3))
nwin=len(windows); f_list=[]; win_shapes=[]
for (st,sh,sw) in windows:
    wt=st.stop-st.start;wh=sh.stop-sh.start;ww=sw.stop-sw.start
    win_shapes.append((wt,wh,ww)); f_list.append(wt*wh*ww)
vid_freq,txt_freq=mmrope.build_window_freqs(win_shapes,TXT_LEN)
Lv=T*H2*W2
vqkv=(vid_an@Wv["qkv"].T).reshape(Lv,3,HEADS,HD)
tqkv=(txt_an@Wt["qkv"].T).reshape(TXT_LEN,3,HEADS,HD)
vq,vk,vv=vqkv[:,0],vqkv[:,1],vqkv[:,2]
tq,tk,tv=tqkv[:,0],tqkv[:,1],tqkv[:,2]
vq=R.rmsnorm(vq,Wv["nq"]);vk=R.rmsnorm(vk,Wv["nk"])
tq=R.rmsnorm(tq,Wt["nq"]);tk=R.rmsnorm(tk,Wt["nk"])
vq_win=vq.reshape(T,H2,W2,HEADS,HD);vk_win=vk.reshape(T,H2,W2,HEADS,HD);vv_win=vv.reshape(T,H2,W2,HEADS,HD)
# vidx: token 网格索引 = h*W2+w
vidx=[];cumf=[0.0]
for (st,sh,sw) in windows:
    wt=st.stop-st.start;wh=sh.stop-sh.start;ww=sw.stop-sw.start
    f_i=wt*wh*ww; cumf.append(cumf[-1]+f_i)
    for lh in range(wh):
        for lw in range(ww):
            vidx.append((sh.start+lh)*W2+(sw.start+lw))
vidx=np.array(vidx,int)

def rope(x,f):
    # x: (tokens, heads, d) 或 (tokens, heads, d), f: (tokens, rot)
    rh=np.empty_like(x)
    rh[...,0::2]=-x[...,1::2]; rh[...,1::2]=x[...,0::2]   # 相邻配对
    rot=min(f.shape[1],x.shape[-1])
    c=np.cos(f[:,:rot])[...,None,:]; s=np.sin(f[:,:rot])[...,None,:]
    out=x.copy(); out[...,:rot]=x[...,:rot]*c+rh[...,:rot]*s
    return out

out_shader=np.zeros((Lv,HEADS,HD),np.float32)
for wi,(st,sh,sw) in enumerate(windows):
    f_i=f_list[wi]; base=sum(f_list[:wi])
    qv=vq_win[st,sh,sw].reshape(-1,HEADS,HD); kv=vk_win[st,sh,sw].reshape(-1,HEADS,HD); vv_p=vv_win[st,sh,sw].reshape(-1,HEADS,HD)
    qv_rope=rope(qv, vid_freq[base:base+f_i]); kv_rope=rope(kv, vid_freq[base:base+f_i])
    tqr=tq.reshape(TXT_LEN,HEADS,HD)
    tq_rope=rope(np.tile(tqr,(1,1,1)), txt_freq[wi*TXT_LEN:(wi+1)*TXT_LEN])
    tk_rope=rope(np.tile(tk,(1,1,1)), txt_freq[wi*TXT_LEN:(wi+1)*TXT_LEN])
    Q=np.transpose(np.concatenate([qv_rope,tq_rope],0),(1,0,2))
    K=np.transpose(np.concatenate([kv_rope,tk_rope],0),(1,0,2))
    V=np.transpose(np.concatenate([vv_p,np.tile(tv,(1,1,1))],0),(1,0,2))
    S=Q.shape[1]; scale=1.0/math.sqrt(HD)
    dot=(Q@np.transpose(K,(0,2,1)))*scale  # (HEADS,S,S)
    for hh in range(HEADS):
        dd=dot[hh]
        out=np.zeros((f_i,HD),np.float32)
        for qi in range(f_i):
            mx=-1e30; ssum=0.0; ov=np.zeros(HD,np.float32)
            for j in range(S):
                d=float(dd[qi,j]); mn=max(mx,d); rescale=math.exp(mx-mn); pr=math.exp(d-mn)
                ssum=ssum*rescale+pr; ov=ov*rescale+pr*V[hh,j]; mx=mn
            out[qi]=ov/ssum
        for qi in range(f_i):
            out_shader[vidx[base+qi],hh]=out[qi]

out_shader_flat=out_shader.reshape(Lv,DIM)
gt=np.fromfile(os.path.join(ROOT,"b0mid_v_attn_gt.f32"),np.float32).reshape(Lv,DIM)
def cos(a,b):return float(np.sum(a.ravel()*b.ravel())/(np.linalg.norm(a.ravel())*np.linalg.norm(b.ravel())+1e-12))
print(f"CPU复现awa.comp vs GT: cos={cos(out_shader_flat,gt):.6f}")
for h in [2,16,3,13]:
    a=out_shader_flat[:,h*HD:(h+1)*HD]; b=gt[:,h*HD:(h+1)*HD]
    nsha=np.linalg.norm(a,axis=1).mean(); ngt=np.linalg.norm(b,axis=1).mean()
    print(f"  head {h} cos={cos(a,b):.5f} norm_shader={nsha:.3f} norm_gt={ngt:.3f}")

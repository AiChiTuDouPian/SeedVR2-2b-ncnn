#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "export"))
import ncnn_io as io

def sinusoidal_embedding(t, dim):
    half = dim//2
    freqs = np.exp(-np.log(10000.0)*np.arange(half,dtype=np.float64)/half)
    args = np.array([t],dtype=np.float64)[:,None]*freqs[None,:]
    return np.concatenate([np.sin(args),np.cos(args)],axis=-1).astype(np.float32)

def time_embedding(t):
    with io.open_sd() as f:
        Wpi=io.get_top_weight(f,"emb_in.proj_in","weight");Bpi=io.get_top_weight(f,"emb_in.proj_in","bias")
        Wph=io.get_top_weight(f,"emb_in.proj_hid","weight");Bph=io.get_top_weight(f,"emb_in.proj_hid","bias")
        Wpo=io.get_top_weight(f,"emb_in.proj_out","weight");Bpo=io.get_top_weight(f,"emb_in.proj_out","bias")
    e=sinusoidal_embedding(t,256)
    e=e@Wpi.T+Bpi; e=e/(1+np.exp(-e))
    e=e@Wph.T+Bph; e=e/(1+np.exp(-e))
    e=e@Wpo.T+Bpo
    return e.reshape(-1).astype(np.float32)

emb = time_embedding(1000.0)
print(f"numpy emb: n={emb.size} range[{emb.min():.4f},{emb.max():.4f}]")

nc = np.fromfile("emb_ncnn.f32", np.float32)
print(f"ncnn emb: n={nc.size} range[{nc.min():.4f},{nc.max():.4f}]")
if nc.size == emb.size:
    cos = float(np.sum(nc*emb)/(np.linalg.norm(nc)*np.linalg.norm(emb)))
    print(f"cos(emb) = {cos:.6f}")
    print(f"max|diff| = {np.abs(nc-emb).max():.6f}")
    print("numpy[0:6] =", emb[:6])
    print("ncnn [0:6] =", nc[:6])

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
导出 SeedVR2 video VAE（单帧 T=1 等价 2D 网络）到 ncnn param + bin。

单帧等价性（已验证，误差 1e-5~1e-6）：
- InflatedCausalConv3d (k_t=3)  -> 2D Conv，核 = W.sum(dim=2)（extend_head 复制首帧的等价）
- InflatedCausalConv3d (k_t=1)  -> 2D Conv，核 = W[:, :, 0]
- Downsample3D -> Padding(右/下 1) + Conv(stride=2, k=3)
- Upsample3D(temporal_up) -> 1x1 Conv(z=0 子集重排) + PixelShuffle(2) + Conv(k=3)
- GroupNorm(32, eps=1e-6) -> ncnn GroupNorm
- mid_block 单头 Attention -> C++ 端实现（qkv/out 投影 + softmax），故拆成 4 个纯 conv 子图

输出（models/m6_vae/）：
  vae_enc1  conv_in + down_blocks           （图像 -> mid 特征 512ch）
  vae_enc2  mid resnets + norm_out+act+out  （mid 特征 -> 32ch mean+logvar）
  vae_dec1  conv_in                         （16ch latent -> mid 特征 512ch）
  vae_dec2  mid resnets + up_blocks + out   （mid 特征 -> 3ch 图像）
  vae_attn_enc / vae_attn_dec 的权重单独导出到 *_attn.npz（group_norm + qkv + out）
"""
import os
import sys
import numpy as np
import torch

COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
sys.path.insert(0, COMFY)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "m6_vae")
MAGIC = 7767517


def conv3d_to_2d_ncnn(weight_3d, bias):
    """(out,in,kt,kh,kw) -> temporal 求和 -> (out,in,kh,kw) + bias。
    ncnn Convolution 权重布局是 (out,in,kh,kw)，与 PyTorch Conv2d 完全一致，无需转置。"""
    w = weight_3d.detach()
    w2 = w[:, :, 0, :, :] if w.shape[2] == 1 else w.sum(dim=2)
    return w2.contiguous().numpy().astype(np.float32), bias.detach().numpy().astype(np.float32)


def upscale_conv_ncnn(weight, bias, channels, temporal_up):
    """Upsample3D upscale_conv (c*N, c, 1,1,1) -> pixel_shuffle 布局 (4c, c) + bias。"""
    c = channels
    w = weight.detach().squeeze(-1).squeeze(-1).squeeze(-1)  # (c*N, c)
    n_out = w.shape[0]
    if temporal_up:
        idx = [i for i in range(n_out) if (i // c) % 2 == 0]  # z=0 子集 4c
    else:
        idx = list(range(n_out))  # 4c，顺序 (x,y,c)
    w_z = w[idx]  # (4c, c) 行序 (x,y) 嵌套
    b_z = bias.detach()[idx] if bias is not None else None
    # 重排 (组=4, c_idx, c_in) -> (c_idx, 组, c_in) -> (4c, c)
    w_final = w_z.view(4, c, c).permute(1, 0, 2).contiguous().view(4 * c, c)
    b_final = b_z.view(4, c).permute(1, 0).contiguous().view(4 * c) if b_z is not None else None
    return w_final.numpy().astype(np.float32), (b_final.numpy().astype(np.float32) if b_final is not None else None)


class NcnnWriter:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.blob_id = 0

    def blob(self):
        self.blob_id += 1
        return f"b{self.blob_id}"

    def add_weight(self, arr, need_flag=False):
        # need_flag: type=0 的权重（如 Convolution weight）前需 4 字节 flag(0=fp32)；
        #            type=1 的权重（bias/gamma/beta）前无 flag。
        if arr is not None:
            self.weights.append((need_flag, np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)))

    def conv(self, name, bottom, weight, bias, stride=1, pad=0, has_bias=True):
        # weight 布局 (out, in, kh, kw)，与 ncnn/PyTorch 一致
        out, inch, kh, kw = weight.shape
        top = self.blob()
        p = f"0={out} 1={kw} 11={kh} 3={stride} 13={stride} 4={pad} 15={pad} 14={pad} 16={pad} 5={1 if has_bias else 0} 6={weight.size}"
        self.layers.append(f"Convolution  {name}  1 1 {bottom} {top} {p}")
        self.add_weight(weight, need_flag=True)
        if has_bias:
            self.add_weight(bias)
        return top

    def conv1x1(self, name, bottom, weight, bias):
        out, inch = weight.shape
        return self.conv(name, bottom, weight.reshape(out, inch, 1, 1), bias, stride=1, pad=0)

    def groupnorm(self, name, bottom, gamma, beta, groups=32, eps=1e-6):
        top = self.blob()
        self.layers.append(f"GroupNorm  {name}  1 1 {bottom} {top} 0={groups} 1={gamma.shape[0]} 2={eps} 3=1")
        self.add_weight(gamma)
        self.add_weight(beta)
        return top

    def swish(self, name, bottom):
        top = self.blob()
        self.layers.append(f"Swish  {name}  1 1 {bottom} {top}")
        return top

    def padding(self, name, bottom, top_b, bottom_b, left_b, right_b):
        top = self.blob()
        self.layers.append(f"Padding  {name}  1 1 {bottom} {top} 0={top_b} 1={bottom_b} 2={left_b} 3={right_b} 4=0 5=0.0")
        return top

    def pixelshuffle(self, name, bottom, factor=2):
        top = self.blob()
        self.layers.append(f"PixelShuffle  {name}  1 1 {bottom} {top} 0={factor} 1=0")
        return top

    def add(self, name, a, b):
        top = self.blob()
        self.layers.append(f"BinaryOp  {name}  2 1 {a} {b} {top} 0=0")
        return top

    def permute(self, name, bottom, order_type):
        top = self.blob()
        self.layers.append(f"Permute  {name}  1 1 {bottom} {top} 0={order_type}")
        return top

    def reshape(self, name, bottom, w=-233, h=-233, c=-233):
        top = self.blob()
        parts = [f"0={w}"]
        if h != -233:
            parts.append(f"1={h}")
        if c != -233:
            parts.append(f"2={c}")
        self.layers.append(f"Reshape  {name}  1 1 {bottom} {top} " + " ".join(parts))
        return top

    def mha(self, name, bottom, embed_dim, num_heads, weight_data_size, kdim, vdim):
        # 注意：不写 6=scale，让 ncnn 用默认 scale=1/sqrt(embed_dim/num_heads)。
        # 单头时 head_dim=embed_dim，默认值 = 1/sqrt(embed_dim) = PyTorch SDPA 的 scale，精确匹配。
        # （ncnn ParamDict::vstr_to_float 对 >9 位小数的 float 有 pow10 溢出 bug，显式写 scale 会解析错。）
        top = self.blob()
        self.layers.append(
            f"MultiHeadAttention  {name}  1 1 {bottom} {top} "
            f"0={embed_dim} 1={num_heads} 2={weight_data_size} 3={kdim} 4={vdim} 5=0"
        )
        return top

    def split(self, name, bottom, n=2):
        tops = [self.blob() for _ in range(n)]
        self.layers.append(f"Split  {name}  1 {n} {bottom} " + " ".join(tops))
        return tops

    def save(self, prefix):
        os.makedirs(OUT_DIR, exist_ok=True)
        param_path = os.path.join(OUT_DIR, prefix + ".param")
        bin_path = os.path.join(OUT_DIR, prefix + ".bin")
        layers = list(self.layers)
        # 最后一层输出 blob 重命名为 out0：用 num_bottom/num_top 定位 top blob
        parts = layers[-1].split()
        nb = int(parts[2]); nt = int(parts[3])
        last_top = 4 + nb + nt - 1
        parts[last_top] = 'out0'
        layers[-1] = ' '.join(parts)
        with open(param_path, "w") as f:
            f.write(f"{MAGIC}\n{len(layers)} {self.blob_id + 1}\n")
            for line in layers:
                f.write(line + "\n")
        # ncnn bin 格式：type=0 权重前 4 字节 flag(0x00000000=fp32)，type=1 权重无 flag
        with open(bin_path, "wb") as f:
            for need_flag, w in self.weights:
                if need_flag:
                    f.write(b"\x00\x00\x00\x00")
                f.write(w.tobytes())
        print(f"[export] {prefix}: {len(layers)} layers, {self.blob_id + 1} blobs, {sum(w.size for _, w in self.weights)} floats")


def resnet(wr, x, rn, prefix):
    x_main, x_short = wr.split(prefix + "_split", x)
    h = wr.groupnorm(prefix + "_norm1", x_main, rn.norm1.weight.detach().numpy().astype(np.float32),
                     rn.norm1.bias.detach().numpy().astype(np.float32))
    h = wr.swish(prefix + "_act1", h)
    w, b = conv3d_to_2d_ncnn(rn.conv1.weight, rn.conv1.bias)
    h = wr.conv(prefix + "_conv1", h, w, b, stride=1, pad=1)
    h = wr.groupnorm(prefix + "_norm2", h, rn.norm2.weight.detach().numpy().astype(np.float32),
                     rn.norm2.bias.detach().numpy().astype(np.float32))
    h = wr.swish(prefix + "_act2", h)
    w, b = conv3d_to_2d_ncnn(rn.conv2.weight, rn.conv2.bias)
    h = wr.conv(prefix + "_conv2", h, w, b, stride=1, pad=1)
    if rn.conv_shortcut is not None:
        w, b = conv3d_to_2d_ncnn(rn.conv_shortcut.weight, rn.conv_shortcut.bias)
        s = wr.conv(prefix + "_shortcut", x_short, w, b, stride=1, pad=0)
    else:
        s = x_short
    return wr.add(prefix + "_add", h, s)


def downsample(wr, x, ds, prefix):
    x = wr.padding(prefix + "_pad", x, 0, 1, 0, 1)
    w, b = conv3d_to_2d_ncnn(ds.conv.weight, ds.conv.bias)
    return wr.conv(prefix + "_conv", x, w, b, stride=2, pad=0)


def upsample(wr, x, us, prefix):
    c = us.channels
    w, b = upscale_conv_ncnn(us.upscale_conv.weight, us.upscale_conv.bias, c, us.temporal_up)
    x = wr.conv1x1(prefix + "_up1x1", x, w, b)
    x = wr.pixelshuffle(prefix + "_ps", x, 2)
    w, b = conv3d_to_2d_ncnn(us.conv.weight, us.conv.bias)
    return wr.conv(prefix + "_conv", x, w, b, stride=1, pad=1)


def export_attention_npz(attn, path):
    d = {
        "gn_weight": attn.group_norm.weight.detach().numpy().astype(np.float32),
        "gn_bias": attn.group_norm.bias.detach().numpy().astype(np.float32),
        "q_weight": attn.to_q.weight.detach().numpy().astype(np.float32),
        "q_bias": attn.to_q.bias.detach().numpy().astype(np.float32),
        "k_weight": attn.to_k.weight.detach().numpy().astype(np.float32),
        "k_bias": attn.to_k.bias.detach().numpy().astype(np.float32),
        "v_weight": attn.to_v.weight.detach().numpy().astype(np.float32),
        "v_bias": attn.to_v.bias.detach().numpy().astype(np.float32),
        "o_weight": attn.to_out[0].weight.detach().numpy().astype(np.float32),
        "o_bias": attn.to_out[0].bias.detach().numpy().astype(np.float32),
    }
    np.savez(path, **d)
    print(f"[export] attention 权重 -> {path}")


def export_attention_param(attn, prefix, H, W):
    """导出 mid_block 单头 attention 子图（ncnn 标准层，全 Vulkan 可跑）。
    H/W 为 mid 分辨率（对应 ncnn Mat 的 h=H, w=W）。子图：
    Input(W,H,512) -> Split(残差) -> GroupNorm -> Permute(0=3) -> Reshape(3D->2D)
      -> MultiHeadAttention -> Reshape(2D->3D) -> Permute(0=4) -> BinaryOp(ADD 残差)
    """
    wr = NcnnWriter()
    wr.layers.append("Input  in0  0 1 in0")
    wr.blob_id = 0
    x_res, x_gn = wr.split(prefix + "_split", "in0")
    x = wr.groupnorm(prefix + "_gn", x_gn,
                     attn.group_norm.weight.detach().numpy().astype(np.float32),
                     attn.group_norm.bias.detach().numpy().astype(np.float32))
    x = wr.permute(prefix + "_p1", x, 3)           # (W,H,512) -> (512,W,H)
    x = wr.reshape(prefix + "_r1", x, w=512, h=-1)  # 3D -> 2D (512, W*H)
    x = wr.mha(prefix + "_mha", x, 512, 1, 512 * 512, 512, 512)
    x = wr.reshape(prefix + "_r2", x, w=512, h=W, c=H)  # 2D -> 3D (512, W, H)
    x = wr.permute(prefix + "_p2", x, 4)           # (512,W,H) -> (W,H,512)
    x = wr.add(prefix + "_add", x, x_res)
    # MHA 权重：q/k/v/o weight(flag) + bias(no flag)。gn 权重已由 groupnorm 写入。
    wr.add_weight(attn.to_q.weight.detach().numpy().astype(np.float32), need_flag=True)
    wr.add_weight(attn.to_q.bias.detach().numpy().astype(np.float32))
    wr.add_weight(attn.to_k.weight.detach().numpy().astype(np.float32), need_flag=True)
    wr.add_weight(attn.to_k.bias.detach().numpy().astype(np.float32))
    wr.add_weight(attn.to_v.weight.detach().numpy().astype(np.float32), need_flag=True)
    wr.add_weight(attn.to_v.bias.detach().numpy().astype(np.float32))
    wr.add_weight(attn.to_out[0].weight.detach().numpy().astype(np.float32), need_flag=True)
    wr.add_weight(attn.to_out[0].bias.detach().numpy().astype(np.float32))
    wr.save(prefix)


def main():
    from omegaconf import OmegaConf
    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    from safetensors.torch import load_file

    yaml_path = os.path.join(COMFY, "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    sdf = cfg.pop("spatial_downsample_factor")
    tdf = cfg.pop("temporal_downsample_factor")
    vae = VideoAutoencoderKLWrapper(spatial_downsample_factor=sdf, temporal_downsample_factor=tdf, freeze_encoder=False, **cfg)
    sd = load_file(os.path.join(COMFY, "models/SEEDVR2/ema_vae_fp16.safetensors"), device="cpu")
    vae.load_state_dict(sd, strict=False)
    vae.eval()

    os.makedirs(OUT_DIR, exist_ok=True)

    # ===== encoder part1: conv_in + down_blocks =====
    enc = vae.encoder
    wr = NcnnWriter()
    wr.layers.append("Input  in0  0 1 in0")
    wr.blob_id = 0
    w, b = conv3d_to_2d_ncnn(enc.conv_in.weight, enc.conv_in.bias)
    x = wr.conv("enc_conv_in", "in0", w, b, stride=1, pad=1)
    for bi, block in enumerate(enc.down_blocks):
        for ri, rn in enumerate(block.resnets):
            x = resnet(wr, x, rn, f"enc_db{bi}_rn{ri}")
        if block.downsamplers is not None:
            x = downsample(wr, x, block.downsamplers[0], f"enc_db{bi}_ds")
    # mid_block: resnet0 -> attention -> resnet1，resnet0 归入 part1
    x = resnet(wr, x, enc.mid_block.resnets[0], "enc_mid_rn0")
    wr.save("vae_enc1")

    # ===== encoder part2: mid resnet1 + norm_out + act + conv_out =====
    wr = NcnnWriter()
    wr.layers.append("Input  in0  0 1 in0")
    wr.blob_id = 0
    x = resnet(wr, "in0", enc.mid_block.resnets[1], "enc_mid_rn1")
    g = enc.conv_norm_out.weight.detach().numpy().astype(np.float32)
    be = enc.conv_norm_out.bias.detach().numpy().astype(np.float32)
    x = wr.groupnorm("enc_norm_out", x, g, be)
    x = wr.swish("enc_act_out", x)
    w, b = conv3d_to_2d_ncnn(enc.conv_out.weight, enc.conv_out.bias)
    x = wr.conv("enc_conv_out", x, w, b, stride=1, pad=1)
    wr.save("vae_enc2")

    # ===== decoder part1: conv_in + mid resnet0 =====
    dec = vae.decoder
    wr = NcnnWriter()
    wr.layers.append("Input  in0  0 1 in0")
    wr.blob_id = 0
    w, b = conv3d_to_2d_ncnn(dec.conv_in.weight, dec.conv_in.bias)
    x = wr.conv("dec_conv_in", "in0", w, b, stride=1, pad=1)
    x = resnet(wr, x, dec.mid_block.resnets[0], "dec_mid_rn0")
    wr.save("vae_dec1")

    # ===== decoder part2: mid resnet1 + up_blocks + norm_out + act + conv_out =====
    wr = NcnnWriter()
    wr.layers.append("Input  in0  0 1 in0")
    wr.blob_id = 0
    x = resnet(wr, "in0", dec.mid_block.resnets[1], "dec_mid_rn1")
    for bi, block in enumerate(dec.up_blocks):
        for ri, rn in enumerate(block.resnets):
            x = resnet(wr, x, rn, f"dec_ub{bi}_rn{ri}")
        if block.upsamplers is not None:
            x = upsample(wr, x, block.upsamplers[0], f"dec_ub{bi}_us")
    g = dec.conv_norm_out.weight.detach().numpy().astype(np.float32)
    be = dec.conv_norm_out.bias.detach().numpy().astype(np.float32)
    x = wr.groupnorm("dec_norm_out", x, g, be)
    x = wr.swish("dec_act_out", x)
    w, b = conv3d_to_2d_ncnn(dec.conv_out.weight, dec.conv_out.bias)
    x = wr.conv("dec_conv_out", x, w, b, stride=1, pad=1)
    wr.save("vae_dec2")

    # ===== attention 权重（encoder/decoder mid 各一个）=====
    export_attention_npz(enc.mid_block.attentions[0], os.path.join(OUT_DIR, "vae_attn_enc.npz"))
    export_attention_npz(dec.mid_block.attentions[0], os.path.join(OUT_DIR, "vae_attn_dec.npz"))

    # ===== attention 子图（ncnn 标准层，全 Vulkan）。mid 分辨率 = 输入/8，1080p 下 H=136 W=202 =====
    export_attention_param(enc.mid_block.attentions[0], "vae_attn_enc", 136, 202)
    export_attention_param(dec.mid_block.attentions[0], "vae_attn_dec", 136, 202)

    print("[export] VAE 分段导出完成 ->", OUT_DIR)


if __name__ == "__main__":
    main()

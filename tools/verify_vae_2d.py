#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 VAE 单帧 2D 等价性（端到端）：用 temporal 求和 + 2D conv + 2D attention 手动实现，
对比完整 3D VAE 的 encode/decode 输出。cos 应 ~1.0。
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F

COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
sys.path.insert(0, COMFY)

EPS = 1e-6


def c2d(conv3d, x):
    """causal conv3d -> temporal 求和 2D conv。x: (B,C,H,W)。"""
    w = conv3d.weight
    w2 = w[:, :, 0] if w.shape[2] == 1 else w.sum(dim=2)
    return F.conv2d(x, w2, conv3d.bias, stride=conv3d.stride[1:], padding=conv3d.padding[1:])


def resnet2d(rn, x):
    h = F.group_norm(x, 32, rn.norm1.weight, rn.norm1.bias, EPS)
    h = F.silu(h)
    h = c2d(rn.conv1, h)
    h = F.group_norm(h, 32, rn.norm2.weight, rn.norm2.bias, EPS)
    h = F.silu(h)
    h = c2d(rn.conv2, h)
    if rn.conv_shortcut is not None:
        x = c2d(rn.conv_shortcut, x)
    return x + h


def attention2d(attn, x):
    residual = x
    h = F.group_norm(x, 32, attn.group_norm.weight, attn.group_norm.bias, EPS)
    B, C, H, W = h.shape
    h = h.reshape(B, C, H * W).transpose(1, 2)  # (B, HW, C)
    q = attn.to_q(h)
    k = attn.to_k(h)
    v = attn.to_v(h)
    scale = 1.0 / np.sqrt(C)
    h = F.scaled_dot_product_attention(q, k, v, scale=scale)  # (B, HW, C)
    h = h.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
    # to_out 是 Linear，作用在 C 维：permute 到最后一维
    h = h.permute(0, 2, 3, 1)  # (B, H, W, C)
    h = attn.to_out[0](h)
    h = h.permute(0, 3, 1, 2)  # (B, C, H, W)
    return h + residual


def encoder2d(enc, x):
    x = c2d(enc.conv_in, x)
    for block in enc.down_blocks:
        for rn in block.resnets:
            x = resnet2d(rn, x)
        if block.downsamplers is not None:
            ds = block.downsamplers[0]
            x = F.pad(x, (0, 1, 0, 1))
            x = c2d(ds.conv, x)
    # mid_block: resnet0 -> attention -> resnet1
    x = resnet2d(enc.mid_block.resnets[0], x)
    x = attention2d(enc.mid_block.attentions[0], x)
    x = resnet2d(enc.mid_block.resnets[1], x)
    x = F.group_norm(x, 32, enc.conv_norm_out.weight, enc.conv_norm_out.bias, EPS)
    x = F.silu(x)
    x = c2d(enc.conv_out, x)
    return x

def upsample2d(us, x):
    c = us.channels
    w = us.upscale_conv.weight.squeeze(-1).squeeze(-1).squeeze(-1)  # (c*N, c)
    if us.temporal_up:
        idx = [i for i in range(w.shape[0]) if (i // c) % 2 == 0]
        w = w[idx]
    b = us.upscale_conv.bias
    b = b[idx] if us.temporal_up else b
    # 重排 (x,y) 组序 -> pixel_shuffle (c_idx,x,y)
    w = w.view(4, c, c).permute(1, 0, 2).reshape(4 * c, c)
    b = b.view(4, c).permute(1, 0).reshape(4 * c)
    x = F.conv2d(x, w.unsqueeze(-1).unsqueeze(-1), b)  # 1x1
    x = F.pixel_shuffle(x, 2)
    x = c2d(us.conv, x)
    return x


def decoder2d(dec, z):
    x = c2d(dec.conv_in, z)
    # mid_block: resnet0 -> attention -> resnet1
    x = resnet2d(dec.mid_block.resnets[0], x)
    x = attention2d(dec.mid_block.attentions[0], x)
    x = resnet2d(dec.mid_block.resnets[1], x)
    for block in dec.up_blocks:
        for rn in block.resnets:
            x = resnet2d(rn, x)
        if block.upsamplers is not None:
            x = upsample2d(block.upsamplers[0], x)
    x = F.group_norm(x, 32, dec.conv_norm_out.weight, dec.conv_norm_out.bias, EPS)
    x = F.silu(x)
    x = c2d(dec.conv_out, x)
    return x


def main():
    from omegaconf import OmegaConf
    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    from safetensors.torch import load_file
    from src.models.video_vae_v3.modules.types import MemoryState

    yaml_path = COMFY + "/src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml"
    cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    sdf = cfg.pop("spatial_downsample_factor")
    tdf = cfg.pop("temporal_downsample_factor")
    vae = VideoAutoencoderKLWrapper(spatial_downsample_factor=sdf, temporal_downsample_factor=tdf, freeze_encoder=False, **cfg)
    sd = load_file(COMFY + "/models/SEEDVR2/ema_vae_fp16.safetensors", device="cpu")
    vae.load_state_dict(sd, strict=False)
    vae.eval()

    torch.manual_seed(0)
    H, W = 64, 64  # 小尺寸快速验证
    x = torch.randn(1, 3, 1, H, W)

    with torch.no_grad():
        # 完整 3D VAE encode（单帧 T=1）
        z3d = vae.encoder(x, memory_state=MemoryState.DISABLED)  # (1, 32, 1, H/8, W/8)
        # 2D 等价 encoder
        x2 = x[:, :, 0]
        z2d = encoder2d(vae.encoder, x2)  # (1, 32, H/8, W/8)
        diff = (z3d.squeeze(2) - z2d).abs().max().item()
        cos = F.cosine_similarity(z3d.squeeze(2).flatten(), z2d.flatten(), dim=0).item()
        print(f"ENCODER: 3D vs 2D 等价  max|diff|={diff:.3e}  cos={cos:.6f}  shape={tuple(z2d.shape)}")

        # 完整 3D VAE decode
        z_lat = torch.randn(1, 16, 1, H // 8, W // 8)
        y3d = vae.decoder(z_lat, memory_state=MemoryState.DISABLED)  # (1,3,1,H,W)
        y2d = decoder2d(vae.decoder, z_lat[:, :, 0])  # (1,3,H,W)
        diff = (y3d.squeeze(2) - y2d).abs().max().item()
        cos = F.cosine_similarity(y3d.squeeze(2).flatten(), y2d.flatten(), dim=0).item()
        print(f"DECODER: 3D vs 2D 等价  max|diff|={diff:.3e}  cos={cos:.6f}  shape={tuple(y2d.shape)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SeedVR2 端到端推理驱动（Vulkan DiT 加速）。

链路：
  LR 图 → 预处理(SideResize 短边=resolution 上采样 / clamp / DivisiblePad16 / Normalize)
       → VAE encode(×0.9152) → cond_latent(16ch)
       → noise(randn, seed=42) + cond(16ch) + mask(1.0) → vid_grid(33ch)
       → 生成真实 token 网格窗口(TXT_LEN=58) → 调用 seedvr2_dit_vk_run.exe (Vulkan DiT)
       → sr_latent(16ch, 原始模型输出 v) → upscaled = noise - v
       → VAE decode(÷0.9152) → SR 图(裁剪 padding) → 保存

用法（cuda_env）:
  python tools/run_e2e.py <lr_image> <output_image> [--resolution 1080] [--workdir TMP] [--exe ...] [--modeldir ...]
"""
import sys, os, struct, argparse, subprocess
import numpy as np
import torch
from torchvision.transforms import functional as TVF
from PIL import Image
from torchvision.transforms import InterpolationMode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = "F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
EXE_DEFAULT = os.path.join(ROOT, "seedvr2_dit_vk_run.exe")
MODELDIR_DEFAULT = os.path.join(ROOT, "models", "m5")
POS_EMB = os.path.join(COMFY, "pos_emb.pt")
SCALING_FACTOR = 0.9152
TXT_LEN = 58
TIMESTEP = 1000.0

# 复用 ComfyUI 的窗口/mmrope 生成逻辑（与 m5_export_windows.py 同款）
sys.path.insert(0, os.path.join(ROOT, "export"))
import awa_window
import mmrope


def resize_shortest_edge(img, size, interpolation=InterpolationMode.BICUBIC):
    """忠实复现 SideResize(downsample_only=False)：短边 = size，保持比例，不上采样时不限制。"""
    h, w = img.shape[-2:]
    if min(h, w) == size:
        return img
    scale = size / float(min(h, w))
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    return TVF.resize(img, (new_h, new_w), interpolation, antialias=True)


def divisible_pad(img, factor=(16, 16)):
    """复现 DivisiblePad((16,16))：右/下补 0 到 16 的倍数。返回 (padded, pre_pad_h, pre_pad_w)。"""
    h, w = img.shape[-2:]
    pad_h = (factor[0] - (h % factor[0])) % factor[0]
    pad_w = (factor[1] - (w % factor[1])) % factor[1]
    if pad_h == 0 and pad_w == 0:
        return img, h, w
    # 新版 torchvision 把 pad 的 mode= 改名为 padding_mode=，这里改用 torch.nn.functional.pad（API 稳定）。
    # 注意 4 元组语义是 (left, right, top, bottom)，DivisiblePad 是右/下补 0 → (0, pad_w, 0, pad_h)。
    img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return img, h, w


def write_raw(path, arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with open(path, "wb") as fp:
        fp.write(struct.pack("<q", arr.ndim))
        for d in arr.shape:
            fp.write(struct.pack("<q", int(d)))
        fp.write(arr.tobytes())


def read_raw(path):
    """读 write_raw 写的裸格式（跳过 8B ndim + 4*8B shape 头），返回 (shape, data)。"""
    with open(path, "rb") as fp:
        ndim = struct.unpack("<q", fp.read(8))[0]
        shape = [struct.unpack("<q", fp.read(8))[0] for _ in range(ndim)]
        total = int(np.prod(shape))
        data = np.frombuffer(fp.read(total * 4), dtype=np.float32).reshape(shape)
    return data


def write_win(path, token_grid, windows, txt_len):
    """与 dit_vk.cpp load_win / m5_export_windows.py 同格式。"""
    t, h, w = token_grid
    nwin = len(windows)
    slices = []
    win_shapes = []
    for (st, sh, sw) in windows:
        wt = st.stop - st.start
        wh = sh.stop - sh.start
        ww = sw.stop - sw.start
        win_shapes.append((wt, wh, ww))
        slices.append([st.start, st.stop, sh.start, sh.stop, sw.start, sw.stop])
    vid_freq, txt_freq = mmrope.build_window_freqs(win_shapes, txt_len)
    with open(path, "wb") as fp:
        hdr = [t, h, w, 4, 3, 3, 0, 0, 0, nwin]
        for v in hdr:
            fp.write(struct.pack("<q", int(v)))
        sl = np.array(slices, dtype=np.int64)
        fp.write(sl.tobytes())
        fp.write(np.ascontiguousarray(vid_freq, dtype=np.float32).tobytes())
        fp.write(np.ascontiguousarray(txt_freq, dtype=np.float32).tobytes())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lr_image")
    ap.add_argument("output_image")
    ap.add_argument("--resolution", type=int, default=1080)
    ap.add_argument("--workdir", default=os.path.join(ROOT, "e2e_work"))
    ap.add_argument("--exe", default=EXE_DEFAULT)
    ap.add_argument("--modeldir", default=MODELDIR_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--color_fix", default="lab", choices=["lab", "none"],
                    help="lab=官方同款 LAB 直方图色彩校正（默认，修复单步扩散的偏色/颗粒感）；none=纯扩散原始输出")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)

    # ---- 0. 载入 VAE + 文本条件 ----
    sys.path.insert(0, COMFY)
    from omegaconf import OmegaConf
    from src.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
    yaml_path = os.path.join(COMFY, "src/models/video_vae_v3/s8_c16_t4_inflation_sd3.yaml")
    cfg = OmegaConf.to_container(OmegaConf.load(yaml_path), resolve=True)
    sdf = cfg.pop("spatial_downsample_factor")
    tdf = cfg.pop("temporal_downsample_factor")
    vae = VideoAutoencoderKLWrapper(
        spatial_downsample_factor=sdf, temporal_downsample_factor=tdf,
        freeze_encoder=False, **cfg)
    from safetensors.torch import load_file
    sd = load_file(os.path.join(COMFY, "models/SEEDVR2/ema_vae_fp16.safetensors"), device="cpu")
    vae.load_state_dict(sd, strict=False)
    vae = vae.eval().half().to(args.device)            # 强制 fp16（权重+bias 同 dtype），与官方 fp16 VAE 一致

    txt_pt = torch.load(POS_EMB, weights_only=True)   # (58, 5120) bfloat16
    # bfloat16 不能直接转 numpy，先 cast 到 float32
    if txt_pt.dtype == torch.bfloat16:
        txt_pt = txt_pt.float()
    txt = np.ascontiguousarray(txt_pt.cpu().numpy().astype(np.float32))

    # ---- 1. 预处理 LR → [-1,1] 的 [1,3,H,W] ----
    img = Image.open(args.lr_image).convert("RGB")
    x = TVF.to_tensor(img)                            # (3,H,W) [0,1]
    x = resize_shortest_edge(x, args.resolution)      # 短边=resolution（上采样）
    x = torch.clamp(x, 0.0, 1.0)
    x, pre_h, pre_w = divisible_pad(x, (16, 16))      # 右/下补 0
    x = TVF.normalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1,1]
    x = x.unsqueeze(0).to(args.device)                # (1,3,H,W)
    print(f"[e2e] 输入 {tuple(img.size)} -> 预处理后 {tuple(x.shape[-2:])} (pad 前 {pre_h}x{pre_w})")

    # ---- 2. VAE encode → cond_latent(1,H_lat,W_lat,16) ----
    # 对齐官方 RNG：官方 Phase1 用 seed_vae = seed+1000000 做 VAE 随机采样。
    # 关键：官方 vae_encode 在 vae_dtype(=fp16)==输入 dtype(fp16) 时不走 autocast，
    # 是纯 fp16 计算（含 layernorm/softmax 都在 fp16）。ncnn 之前用 autocast(fp16)
    # 会把 layernorm/softmax 留在 fp32（混合精度），导致 cond 与官方有 ~0.08 的偏差。
    # 这里改为与官方一致：输入 cast 到 fp16，直接 vae.encode（不加 autocast）。
    torch.manual_seed(args.seed + 1000000)
    z = vae.encode(x.half()).latent                     # (1,16,H_lat,W_lat) 纯 fp16，与官方一致
    z = (z.float() * SCALING_FACTOR).permute(0, 2, 3, 1)  # (1,H_lat,W_lat,16)
    H_lat, W_lat = z.shape[1], z.shape[2]
    cond = z.detach().cpu().numpy().astype(np.float32)
    del z                                             # 释放 VAE encoder 显存；保留 x(预处理LR)供下方 LAB 色彩校正
    torch.cuda.empty_cache()
    print(f"[e2e] latent {cond.shape} (token 网格 {(1, H_lat // 2, W_lat // 2)})")

    # ---- 3. 构造 33ch vid_grid（noise16 + cond16 + mask1） ----
    # 对齐官方 RNG：官方 Phase2 用 set_seed(seed) 后 torch.randn_like(latent)。
    # 关键：官方 latent 经 optimized_channels_to_last 后物理内存是 channel-first (1,16,H,W)，
    # 但逻辑形状是 channel-last (1,H,W,16)。randn_like 按物理内存顺序填充，因此逻辑上
    # 的噪声等于「先在 (1,16,H,W) 上生成 randn，再 permute 到 (1,H,W,16)」。
    # 这里必须复现这一布局，否则噪声逻辑值会被置换，导致 x0 变成噪声/颗粒。
    torch.manual_seed(args.seed)
    noise_cf = torch.randn((1, 16, H_lat, W_lat), device=args.device, dtype=torch.float32)
    noise = noise_cf.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32)
    mask = np.ones((1, H_lat, W_lat, 1), dtype=np.float32)
    cond17 = np.concatenate([cond, mask], axis=-1)     # (1,H_lat,W_lat,17)
    vid_grid = np.concatenate([noise, cond17], axis=-1)  # (1,H_lat,W_lat,33)
    write_raw(os.path.join(args.workdir, "vid_grid.bin"), vid_grid)
    write_raw(os.path.join(args.workdir, "txt.bin"), txt)

    # ---- 4. 真实 token 网格窗口（TXT_LEN=58） ----
    token_grid = (1, H_lat // 2, W_lat // 2)
    win_ns = awa_window.make_720Pwindows_bysize(token_grid, (4, 3, 3))
    win_sh = awa_window.make_shifted_720Pwindows_bysize(token_grid, (4, 3, 3))
    write_win(os.path.join(args.workdir, "win_ns.bin"), token_grid, win_ns, TXT_LEN)
    write_win(os.path.join(args.workdir, "win_sh.bin"), token_grid, win_sh, TXT_LEN)
    with open(os.path.join(args.workdir, "params.txt"), "w") as fp:
        fp.write(f"1 {H_lat} {W_lat} {TXT_LEN} {TIMESTEP}\n")

    # ---- 5. 调用 Vulkan DiT exe ----
    cmd = [args.exe, args.workdir, args.modeldir]
    print(f"[e2e] 运行: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("[e2e][FAIL] exe 返回非零:\n", r.stderr)
        sys.exit(1)

    # ---- 6. sr_latent(16ch) = 原始模型输出 v；upscaled = noise - v ----
    sr = read_raw(os.path.join(args.workdir, "sr_latent.bin"))   # (1,H_lat,W_lat,16)，跳过 40B 头
    upscaled = noise - sr                               # v_lerp 单步: x0 = x_t - (t/T)*v, t/T=1
    write_raw(os.path.join(args.workdir, "upscaled.bin"), upscaled)
    z_dec = torch.from_numpy(upscaled).permute(0, 3, 1, 2).to(args.device) / SCALING_FACTOR  # (1,16,H_lat,W_lat)

    # ---- 7. VAE decode → SR 图 ----
    # 对齐官方：官方 decode_all_batches 先把 latent cast 到 compute_dtype(=fp16)，
    # 再 vae_decode（vae_dtype==dtype 不进 autocast，纯 fp16）。这里同样纯 fp16。
    z_dec = (torch.from_numpy(upscaled).permute(0, 3, 1, 2).to(args.device) / SCALING_FACTOR).half()
    y = vae.decode(z_dec).sample                         # (1,3,H_px,W_px) [-1,1] 纯 fp16
    y = y.float()
    # 官方 Phase4 默认 lab 色彩校正：把超分结果的颜色分布对齐到预处理后的 LR（style=x）。
    if args.color_fix == "lab":
        sys.path.insert(0, COMFY)
        from src.utils.color_fix import lab_color_transfer
        from src.utils.debug import Debug
        # x 是预处理后的 LR，[1,3,padded_H,padded_W]，[-1,1]；y 是扩散解码结果，同空间。
        y = lab_color_transfer(y, x, Debug(enabled=False), luminance_weight=0.8)
    y = y.clamp(-1, 1).mul(0.5).add_(0.5).clamp(0, 1)   # [0,1]
    y = y[:, :, :pre_h, :pre_w]                         # 裁掉 padding
    out = (y.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(out).save(args.output_image)
    print(f"[e2e] 输出已保存: {args.output_image}  shape={out.shape[:2]}  color_fix={args.color_fix}")


if __name__ == "__main__":
    main()

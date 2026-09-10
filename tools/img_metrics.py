#!/usr/bin/env python3
# img_metrics.py — 两张同尺寸 PNG 的像素级对拍指标（纯 numpy/PIL，无其他依赖）
#
# 指标口径（与仓库历史报告一致）：
#   cos      = <a,b> / (||a|| ||b||)          —— 全局余弦相似度（0..1）
#   PSNR     = 10*log10(1 / MSE)              —— 在 [0,1] 浮点域，等价于 8bit 域的 10*log10(255^2/MSE)
#   mean|d|  = 平均绝对误差（[0,1] 域，即 8bit 域的 mean/255）
#   max|d|   = 最大绝对误差（[0,1] 域）
#   SSIM     = 灰度 SSIM（8x8 滑窗，与历史 img_final_cmp.py 同实现）
#
# 用法:
#   python img_metrics.py a.png b.png [c.png ...]      # 第一个为参考，其余依次对比
#   python img_metrics.py --ssim a.png b.png
import sys, os
import numpy as np
from PIL import Image


def load_rgb(path):
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0


def ssim_gray(a, b):
    """灰度 SSIM，8x8 滑窗（无高斯核），与历史报告实现一致。"""
    ga = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gb = b @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    win, C1, C2 = 8, 0.01 ** 2, 0.03 ** 2
    H, W = ga.shape
    vals = []
    for y in range(0, H - win + 1, win):
        for x in range(0, W - win + 1, win):
            pa = ga[y:y + win, x:x + win].ravel()
            pb = gb[y:y + win, x:x + win].ravel()
            ma, mb = pa.mean(), pb.mean()
            va, vb = pa.var(), pb.var()
            cov = ((pa - ma) * (pb - mb)).mean()
            vals.append(((2 * ma * mb + C1) * (2 * cov + C2)) /
                        ((ma * ma + mb * mb + C1) * (va + vb + C2)))
    return float(np.mean(vals)) if vals else float("nan")


def compare(ref_path, path, do_ssim=False):
    a = load_rgb(ref_path)
    b = load_rgb(path)
    if a.shape != b.shape:
        print("SIZE MISMATCH %s %s vs %s %s" % (ref_path, a.shape, path, b.shape))
        return None
    d = a - b
    sse = float(np.sum(d * d))
    n = d.size
    mse = sse / n
    cos = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    row = {
        "ref": os.path.basename(ref_path),
        "img": os.path.basename(path),
        "wh": "%dx%d" % (a.shape[1], a.shape[0]),
        "cos": cos,
        "psnr": psnr,
        "mean_abs": float(np.mean(np.abs(d))),
        "max_abs": float(np.max(np.abs(d))),
        "mse": mse,
    }
    if do_ssim:
        row["ssim"] = ssim_gray(a, b)
    return row


def main():
    args = [a for a in sys.argv[1:]]
    do_ssim = "--ssim" in args
    args = [a for a in args if not a.startswith("--")]
    if len(args) < 2:
        print(__doc__)
        return 1
    ref = args[0]
    print("参考: %s" % ref)
    hdr = "%-34s %-11s %9s %9s %9s %9s" % ("对比图", "尺寸", "cos", "PSNR/dB", "mean|d|", "max|d|")
    if do_ssim:
        hdr += " %8s" % "SSIM"
    print(hdr)
    print("-" * len(hdr))
    for p in args[1:]:
        r = compare(ref, p, do_ssim)
        if r is None:
            continue
        line = "%-34s %-11s %9.6f %9.2f %9.5f %9.4f" % (
            r["img"][:34], r["wh"], r["cos"], r["psnr"], r["mean_abs"], r["max_abs"])
        if do_ssim:
            line += " %8.4f" % r.get("ssim", float("nan"))
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

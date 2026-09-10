#!/usr/bin/env python3
# pt_reference.py — 生成 SeedVR2 官方 PyTorch 全流程参考图（与 ncnn 引擎同口径）
#
# 口径约定（必须与 ncnn 侧一致，否则对拍无效）：
#   --resolution        N        目标短边像素（与 ncnn `--resolution N` 同义）
#   --color_correction  none     关闭官方 LAB 色彩校正
#                                 ↔ ncnn 侧必须加 `--no-colorfix`
#                                 （两边算法实现不同，只关一边会引入全局色调差，
#                                   PSNR 会从 ~46dB 掉到 ~27dB，务必成对使用）
#   --seed              42       与 ncnn 默认 seed 一致（ncnn: cfg.seed 默认 42）
#   --dit_model  seedvr2_ema_3b_fp16.safetensors
#                                官方 fp16 全流程图作为 ground truth
#                                （fp8 官方权重与 ncnn 权重同源性问题会带来 ~9% 底噪）
#
# 用法:
#   <cuda_env>/python pt_reference.py <input.png> <out.png> <resolution> [额外 CLI 参数...]
# 例:
#   python pt_reference.py F:/Seedvr2/seedvr2-ncnn/input/椎名真白.png F:/Seedvr2/pt_ref/mashiro_pt1080.png 1080
#
# 环境: 需要 CUDA 版 torch（本机 C:/Users/15327/miniconda3/envs/cuda_env）
import sys, os

COMFY = r"F:/Seedvr2/ComfyUI-SeedVR2_VideoUpscaler"
DIT_MODEL = "seedvr2_ema_3b_fp16.safetensors"


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1
    inp, out, res = sys.argv[1], sys.argv[2], sys.argv[3]
    extra = sys.argv[4:]

    # ⚠ 必须先转绝对路径：下面 os.chdir(COMFY) 会让相对路径失效
    #   （inference_cli.py 依赖相对路径 ./models，故必须 chdir，不能改它）
    inp = os.path.abspath(inp)
    out = os.path.abspath(out)
    if not os.path.isfile(inp):
        print("[pt_ref] 输入不存在: %s" % inp)
        return 1
    os.makedirs(os.path.dirname(out), exist_ok=True)

    os.chdir(COMFY)  # inference_cli.py 依赖相对路径 ./models
    sys.path.insert(0, COMFY)

    import importlib
    inf = importlib.import_module("inference_cli")

    argv = ["inference_cli.py", inp,
            "--output", out,
            "--resolution", str(res),
            "--cuda_device", "0",
            "--output_format", "png",
            "--dit_model", DIT_MODEL,
            "--color_correction", "none",
            "--seed", "42"] + extra
    print("[pt_ref] 命令: %s" % " ".join(argv), flush=True)
    sys.argv = argv
    inf.main()
    print("[pt_ref] 完成 -> %s" % out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

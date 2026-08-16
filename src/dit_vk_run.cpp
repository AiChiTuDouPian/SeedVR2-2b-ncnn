// dit_vk_run.cpp — SeedVR2-ncnn 命令行入口（端到端 DiT 加速）
// 用法:
//   seedvr2_dit_vk_run <workdir> <modeldir> [--fp16]
//   workdir: 含 vid_grid.bin / txt.bin / params.txt / win_ns.bin / win_sh.bin
//   modeldir: 含 *.param/*.bin + awa.spv
//   输出: workdir/sr_latent.bin
// 预处理(图像 resize/normalize) 与 VAE 编解码 由 Python 桥接(run_e2e.py) 完成。
#include "seedvr2_pipeline.h"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "用法: %s <workdir> <modeldir> [--fp16]\n", argv[0]);
        return 1;
    }
    std::string workdir = argv[1];
    std::string modeldir = argv[2];
    bool fp16 = false;
    for (int i = 3; i < argc; i++) {
        if (std::string(argv[i]) == "--fp16") fp16 = true;
    }
    return seedvr2::run_dit(workdir, modeldir, fp16);
}

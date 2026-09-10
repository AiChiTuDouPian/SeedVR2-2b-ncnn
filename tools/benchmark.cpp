// tools/benchmark.cpp
// SeedVR2-ncnn 推理性能 benchmark。
// 逐阶段计时（预处理 / VAE encode / DiT / VAE decode / 后处理），
// 并打印模型加载方式（单 Net 全部加载 vs 分块图 vs 旧分块路径）与计算位置（CPU/GPU）。
//
// 用法:
//   seedvr2_bench --model <dir> [--image <png/jpg>] [--resolution 720]
//                  [--precision 0|1|2] [--warmup 1] [--runs 3] [--seed 100]
//
// 若不提供 --image，则生成一张随机彩图作为基准输入。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

#include "image_io.h"
#include "seedvr2_engine.h"

static void print_usage(const char* prog) {
    fprintf(stderr,
        "用法: %s --model <dir> [options]\n"
        "  --model <dir>     模型根目录（含 dit/modeldir, vae/vaedir, 文本条件）[必填]\n"
        "  --graphdir <dir>  整图/分块图目录（dit_graph.* 或 dit_block_*）；默认 models/m5_graph\n"
        "  --image <path>    输入图（png/jpg）；省略则生成随机彩图\n"
        "  --resolution <n>  短边目标分辨率（默认 720）\n"
        "  --precision <0|1|2>  0=fp32 1=fp16 2=bf16（默认 2）\n"
        "  --fp16-arith      中间算术用 fp16 累加（默认 fp32 累加；fp16 累加大约 2 倍 GEMM 吞吐，但大激活可能溢出 Inf/NaN）\n"
        "  --warmup <n>      预热次数（默认 1）\n"
        "  --runs <n>        正式计时次数（默认 3）\n"
        "  --seed <n>        base seed（默认 100）\n"
        "  --no-colorfix     关闭 LAB 色彩校正\n",
        prog);
}

struct Accum {
    std::vector<double> samples; // 毫秒
    double sum = 0.0;
    void push(double ms) { samples.push_back(ms); sum += ms; }
    double avg() const { return samples.empty() ? 0.0 : sum / samples.size(); }
    double min() const { return samples.empty() ? 0.0 : *std::min_element(samples.begin(), samples.end()); }
    double max() const { return samples.empty() ? 0.0 : *std::max_element(samples.begin(), samples.end()); }
};

int main(int argc, char** argv) {
    std::string modeldir, image_path, graphdir;
    int resolution = 720;
    int precision = 2;
    int warmup = 1;
    int runs = 3;
    int base_seed = 100;
    bool fp16_arith = false;
    bool color_fix = true;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "缺少参数: %s\n", name); exit(1); }
            return argv[++i];
        };
        if (a == "--model") modeldir = next("--model");
        else if (a == "--graphdir") graphdir = next("--graphdir");
        else if (a == "--image") image_path = next("--image");
        else if (a == "--resolution") resolution = atoi(next("--resolution"));
        else if (a == "--precision") precision = atoi(next("--precision"));
        else if (a == "--fp16-arith") fp16_arith = true;
        else if (a == "--warmup") warmup = atoi(next("--warmup"));
        else if (a == "--runs") runs = atoi(next("--runs"));
        else if (a == "--seed") base_seed = atoi(next("--seed"));
        else if (a == "--no-colorfix") color_fix = false;
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { fprintf(stderr, "未知参数: %s\n", a.c_str()); print_usage(argv[0]); return 1; }
    }
    if (modeldir.empty()) { fprintf(stderr, "错误: 必须指定 --model\n"); print_usage(argv[0]); return 1; }
    if (precision < 0 || precision > 2) { fprintf(stderr, "错误: --precision 必须是 0/1/2\n"); return 1; }

    // ---- 准备输入图 ----
    std::vector<float> rgb; int W = 0, H = 0;
    if (!image_path.empty()) {
        if (!img::load(image_path, rgb, W, H)) {
            fprintf(stderr, "错误: 无法载入输入图 %s\n", image_path.c_str());
            return 1;
        }
        fprintf(stderr, "[bench] 载入输入图 %s  (%dx%d)\n", image_path.c_str(), W, H);
    } else {
        // 生成 720x1280 随机彩图作为基准
        W = 720; H = 1280;
        rgb.assign((size_t)W * H * 3, 0.0f);
        srand((unsigned)base_seed);
        for (size_t i = 0; i < rgb.size(); i++) rgb[i] = (float)rand() / RAND_MAX;
        fprintf(stderr, "[bench] 未提供 --image，使用随机彩图 %dx%d\n", W, H);
    }

    // ---- 初始化引擎 ----
    SeedVR2Engine::Config cfg;
    cfg.modeldir = modeldir;
    if (!graphdir.empty()) cfg.graphdir = graphdir;
    cfg.resolution = resolution;
    cfg.precision = precision;
    cfg.fp16_arith = fp16_arith;
    cfg.color_fix = color_fix;
    cfg.seed = base_seed;

    SeedVR2Engine engine;
    if (!engine.init(cfg)) {
        fprintf(stderr, "[bench] 引擎初始化失败\n");
        return 1;
    }

    // ---- 探测一次：触发模型加载（单 Net / 分块图 / 旧分块），后续读回加载方式 ----
    {
        std::vector<float> out; int oW = 0, oH = 0;
        if (!engine.process(rgb, W, H, 0, out, oW, oH)) {
            fprintf(stderr, "[bench] 探测推理失败\n");
            return 1;
        }
    }

    // ---- 打印模型加载方式与计算位置 ----
    SeedVR2Engine::LoadModeInfo lmi = engine.load_mode_info();
    std::string load_mode;
    int nb = 0, chunk = 0; bool resident = false, single = false, graph = false;
    if (lmi.single_net) {
        load_mode = "单 Net 全部加载 (dit_graph.*，32层常驻单一计算图)";
        nb = 1;
    } else if (lmi.graph_ready) {
        load_mode = "分块图 (dit_block_*，每块 " + std::to_string(lmi.block_chunk) +
                    " 层，共 " + std::to_string(lmi.num_blocks) + " 块)";
        nb = lmi.num_blocks; chunk = lmi.block_chunk;
    } else {
        load_mode = "旧分块路径 (逐层 ncnn Net，每 K 层 reset_vulkan_device)";
    }
    resident = lmi.graph_resident;
    single = lmi.single_net;
    graph = lmi.graph_ready;

    const char* prec_str = (precision == 0) ? "fp32" : (precision == 1) ? "fp16" : "bf16";

    printf("\n======================================== 模型配置 ========================================\n");
    printf("  模型目录      : %s\n", modeldir.c_str());
    printf("  加载方式      : %s\n", load_mode.c_str());
    printf("  图块常驻      : %s\n", resident ? "是 (帧间零加载)" : "否 (逐块释放)");
    printf("  计算后端      : Vulkan GPU (ncnn) + 少量 CPU 预处理/后处理\n");
    printf("  权重精度      : %s\n", prec_str);
    printf("  分辨率(短边)  : %d\n", resolution);
    printf("  LAB 色彩校正  : %s\n", color_fix ? "开" : "关");
    printf("  DiT 层数      : 32 (前10层 vid/txt 双分支, 后22层共享)\n");
    printf("  输入尺寸      : %dx%d  ->  输出尺寸由引擎决定\n", W, H);
    printf("=====================================================================================\n\n");

    // ---- 预热 ----
    for (int i = 0; i < warmup; i++) {
        std::vector<float> out; int oW, oH;
        engine.process(rgb, W, H, i, out, oW, oH);
    }
    fprintf(stderr, "[bench] 预热 %d 次完成\n", warmup);

    // ---- 正式计时：收集每个阶段 ----
    // 阶段索引稳定: 0预处理 1VAEenc 2DiT 3upscaled 4VAEdec 5LAB 6反归一化
    const size_t NSTAGE = 7;
    std::vector<Accum> acc(NSTAGE);
    std::vector<double> total_acc;
    std::vector<std::string> stage_names(NSTAGE);
    std::vector<std::string> stage_dev(NSTAGE);

    for (int r = 0; r < runs; r++) {
        std::vector<float> out; int oW, oH;
        if (!engine.process(rgb, W, H, warmup + r, out, oW, oH)) {
            fprintf(stderr, "[bench] 第 %d 次推理失败\n", r);
            return 1;
        }
        const auto& p = engine.profiler();
        double tot = 0.0;
        for (size_t s = 0; s < p.stages.size() && s < NSTAGE; s++) {
            acc[s].push(p.stages[s].ms);
            stage_names[s] = p.stages[s].name;
            stage_dev[s] = p.stages[s].device;
            tot += p.stages[s].ms;
        }
        total_acc.push_back(tot);
    }

    // ---- 打印逐阶段统计 ----
    double total_avg = std::accumulate(total_acc.begin(), total_acc.end(), 0.0) / total_acc.size();

    printf("=================== 各阶段耗时 (warmup=%d, runs=%d, precision=%s) ===================\n",
           warmup, runs, prec_str);
    printf("%-44s | %-14s | %8s %8s %8s %7s\n", "阶段", "计算位置", "avg(ms)", "min", "max", "占比");
    printf("-------------------------------------------------------------------------------------------------\n");
    for (size_t s = 0; s < NSTAGE; s++) {
        if (acc[s].samples.empty()) continue;
        double pct = total_avg > 0 ? acc[s].avg() / total_avg * 100.0 : 0.0;
        printf("%-44s | %-14s | %8.2f %8.2f %8.2f %6.1f%%\n",
               stage_names[s].c_str(), stage_dev[s].c_str(),
               acc[s].avg(), acc[s].min(), acc[s].max(), pct);
    }
    printf("-------------------------------------------------------------------------------------------------\n");
    printf("%-44s | %-14s | %8.2f %8.2f %8.2f\n", "总计 (端到端)", "CPU+GPU",
           total_avg,
           *std::min_element(total_acc.begin(), total_acc.end()),
           *std::max_element(total_acc.begin(), total_acc.end()));
    printf("=================================================================================================\n\n");

    // ---- 加载方式 vs 计算位置 简述 ----
    printf("[bench] 加载方式: %s\n", load_mode.c_str());
    if (graph || single) {
        printf("[bench] 计算位置: DiT 主体在 GPU(Vulkan) 常驻计算图执行；"
               "预处理/采样/后处理在 CPU。\n");
    } else {
        printf("[bench] 计算位置: GEMM 在 GPU，RMSNorm/Ada/窗口划分在 CPU 或 compute shader；"
               "逐层间有 CPU<->GPU 往返。\n");
    }
    printf("[bench] 单帧端到端平均: %.2f ms (≈ %.2f FPS)\n", total_avg, 1000.0 / total_avg);

    return 0;
}

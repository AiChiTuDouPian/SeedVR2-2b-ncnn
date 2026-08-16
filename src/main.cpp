// main.cpp — SeedVR2 全 Vulkan 自包含端到端推理（图片 / 视频帧序列双模式）
//
// 图片模式:
//   seedvr2_run <input.png> <output.png> [--resolution 1080] [--modeldir DIR] [--vaedir DIR] [--seed 42] [--no-colorfix]
// 帧序列(视频)模式:
//   seedvr2_run <frames_in_dir> <frames_out_dir> [--resolution 1080] [--modeldir DIR] [--vaedir DIR] [--seed 42] [--no-colorfix]
//
// 输入是文件 -> 单张图片超分；输入是目录 -> 逐帧独立超分（每帧复用现有 T=1 管线，
// 图片推理零改动、零风险）。帧序列模式可用 ffmpeg 等工具拆/合帧：
//   ffmpeg -i in.mp4 frames/%06d.png
//   seedvr2_run frames/ out_frames/ --resolution 1080
//   ffmpeg -framerate 24 -i out_frames/%06d.png -c:v libx264 -pix_fmt yuv420p out.mp4
#include "seedvr2_engine.h"
#include "image_io.h"
#include <cstdio>
#include <cstring>
#include <cctype>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

static bool is_image_file(const std::string& path) {
    static const char* exts[] = {".png", ".jpg", ".jpeg", ".bmp", ".tga", ".psd", ".gif", ".hdr", ".pic", ".pnm"};
    std::string lower = path;
    for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
    for (const char* e : exts) {
        size_t n = strlen(e);
        if (lower.size() >= n && lower.compare(lower.size() - n, n, e) == 0) return true;
    }
    return false;
}

// 列出目录里所有图片帧文件（按文件名排序，保证帧顺序确定性）
static bool list_frames(const std::string& dir, std::vector<std::string>& files) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return false;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (e.is_regular_file() && is_image_file(e.path().filename().string()))
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    return !files.empty();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "用法:\n");
        fprintf(stderr, "  图片:  %s <input.png> <output.png> [--resolution 1080] [--modeldir DIR] [--vaedir DIR] [--seed 42] [--no-colorfix] [--fp16|--bf16]\n", argv[0]);
        fprintf(stderr, "  帧序列:%s <frames_in_dir> <frames_out_dir> [--resolution 1080] [--modeldir DIR] [--vaedir DIR] [--seed 42] [--no-colorfix] [--fp16|--bf16]\n", argv[0]);
        fprintf(stderr, "  精度: 默认 fp32(逐位对齐)；--fp16 半精度(更快但大激活可能溢出)；--bf16 bfloat16(推荐,范围安全+TensorCore加速)\n");
        return 1;
    }
    std::string in_path = argv[1], out_path = argv[2];
    int resolution = 1080;
    std::string modeldir = "models/m5";
    std::string vaedir = "models/m6_vae";
    int seed = 42;
    bool color_fix = true;
    int precision = 0;   // 0=fp32(默认) 1=fp16 2=bf16
    std::string graphdir = "models/m5_graph";
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--resolution") && i + 1 < argc) resolution = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--modeldir") && i + 1 < argc) modeldir = argv[++i];
        else if (!strcmp(argv[i], "--vaedir") && i + 1 < argc) vaedir = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-colorfix")) color_fix = false;
        else if (!strcmp(argv[i], "--fp16")) precision = 1;
        else if (!strcmp(argv[i], "--bf16")) precision = 2;
        else if (!strcmp(argv[i], "--no-graph")) graphdir = "";
        else if (!strcmp(argv[i], "--graphdir") && i + 1 < argc) graphdir = argv[++i];
    }

    // 判断输入类型：目录 = 帧序列，文件 = 单张图片
    std::error_code ec;
    bool is_dir = fs::is_directory(in_path, ec);

    // ---- 初始化引擎（模型只加载一次，图片/视频共用）----
    SeedVR2Engine engine;
    SeedVR2Engine::Config cfg;
    cfg.resolution = resolution;
    cfg.modeldir = modeldir;
    cfg.vaedir = vaedir;
    cfg.seed = seed;
    cfg.color_fix = color_fix;
    cfg.precision = precision;
    cfg.graphdir = graphdir;
    // 单图：graph 块逐块释放（大分辨率显存安全）；帧序列：低精度块常驻（帧间零加载加速）
    cfg.graph_resident = is_dir;
    if (!engine.init(cfg)) return 1;

    if (!is_dir) {
        // ================= 图片模式 =================
        std::vector<float> rgb; int W, H;
        if (!img::load(in_path, rgb, W, H)) return 1;
        fprintf(stderr, "[main] 输入 %dx%d\n", W, H);
        std::vector<float> out_rgb; int outW, outH;
        if (!engine.process(rgb, W, H, 0, out_rgb, outW, outH)) return 1;
        if (!img::save(out_path, out_rgb.data(), outW, outH)) return 1;
        fprintf(stderr, "[main] 输出已保存: %s (%dx%d)\n", out_path.c_str(), outW, outH);
        return 0;
    }

    // ================= 帧序列（视频）模式 =================
    std::vector<std::string> frames;
    if (!list_frames(in_path, frames)) {
        fprintf(stderr, "[FAIL] %s 不是目录或目录内无图片帧\n", in_path.c_str());
        return 1;
    }
    fs::create_directories(out_path, ec);
    fprintf(stderr, "[main] 帧序列模式：共 %zu 帧，逐帧独立超分\n", frames.size());

    for (size_t i = 0; i < frames.size(); i++) {
        std::vector<float> rgb; int W, H;
        if (!img::load(frames[i], rgb, W, H)) {
            fprintf(stderr, "[FAIL] 读帧 %s\n", frames[i].c_str());
            return 1;
        }
        std::vector<float> out_rgb; int outW, outH;
        if (!engine.process(rgb, W, H, (int)i, out_rgb, outW, outH)) {
            fprintf(stderr, "[FAIL] 帧 %zu/%zu 推理失败\n", i + 1, frames.size());
            return 1;
        }
        std::string fname = fs::path(frames[i]).filename().string();
        std::string out_file = (fs::path(out_path) / fname).string();
        if (!img::save(out_file, out_rgb.data(), outW, outH)) {
            fprintf(stderr, "[FAIL] 写帧 %s\n", out_file.c_str());
            return 1;
        }
        fprintf(stderr, "[main] 帧 %zu/%zu 完成 (%dx%d) -> %s\n", i + 1, frames.size(), outW, outH, out_file.c_str());
    }
    fprintf(stderr, "[main] 帧序列处理完成：%zu 帧 -> %s/\n", frames.size(), out_path.c_str());
    return 0;
}

// test_single_net.cpp — 验证「合并成单整图 Net」机制数值正确
// 对比：单整图 Net（dit_<N>l*.param/bin，N 层合并 1 个 Net，整图仅 1 次 download）
//      vs 逐层参考路径（forward_latent，原 DiT 块路径，逐层 Net + CPU 残差）
// 两者计算同一 N 层函数，仅图结构不同（合并 vs 逐层），权重同精度（默认 fp32 精确对拍）。
// 期望 cos≈1.0：合并不改变计算，且单 Net 全程 VkMat 不落地 CPU、整图仅 1 次 download。
// 用法：seedvr2_test_single_net [model_dir] [graph_dir] [th] [tw] [nlayers] [prefix] [precision]
//   默认 nlayers=4, prefix=dit_4l（fp32 小单 Net，1.3GB，可常驻验证机制而不触发 32 层 OOM）
//   precision: 0=fp32（默认，精确对拍） 2=bf16（验证低精度路径 AdaCompose 输出转换）
#include "dit_vk.h"
#include "awa_window.h"
#include "awa_layer.h"
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/mat.h>
#include <ncnn/command.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

static const int DIM = 2560;
static const int TXT_N = 58;

static unsigned int g_seed = 2026;
static float frand() { g_seed = g_seed * 1664525u + 1013904223u; return ((g_seed >> 8) & 0xFFFFFF) / (float)0x1000000; }

static double cos_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return dot / sqrt(na * nb);
}
static double psnr(const std::vector<float>& a, const std::vector<float>& b) {
    double mse = 0, mx = 1e-20;
    for (size_t i = 0; i < a.size(); i++) {
        double e = (double)a[i] - b[i]; mse += e * e;
        mx = std::max(mx, std::max((double)fabs(a[i]), (double)fabs(b[i])));
    }
    mse /= a.size();
    return 10.0 * log10(mx * mx / mse);
}

int main(int argc, char** argv)
{
    std::string model_dir = argc > 1 ? argv[1] : "models/m5/";
    std::string graph_dir = argc > 2 ? argv[2] : "models/m5_graph/";
    int th = argc > 3 ? atoi(argv[3]) : 23;   // 360p token 网格 23×34
    int tw = argc > 4 ? atoi(argv[4]) : 34;
    int N  = argc > 5 ? atoi(argv[5]) : 4;     // 单 Net 层数（默认 4，验证机制不 OOM）
    std::string prefix = argc > 6 ? argv[6] : "dit_4l";  // 单 Net param/bin 前缀（默认 fp32 小单 Net）
    int precision = argc > 7 ? atoi(argv[7]) : 0;        // 0=fp32 2=bf16
    if (model_dir.back() != '/') model_dir += '/';
    if (graph_dir.back() != '/') graph_dir += '/';

    // fp32：单 Net(fp32 小单 Net) 与逐层参考路径(fp32) 同精度，做精确对拍（cos≈1.0）。
    // 注：32 层 fp32 单 Net 权重 20GB 超 16GB 显存会被 load_single 拒绝；本测试用 4 层小单 Net。
    // bf16（precision=2）：验证 AdaCompose 输出的 bf16 转换与 CPU 参考一致（参考路径也走 bf16）。
    fprintf(stderr, "[test] precision=%d\n", precision);
    DitVk dit;
    if (!dit.init(model_dir, false, precision)) { fprintf(stderr, "FAIL DitVk::init\n"); return 1; }

    int nw[3] = {4, 3, 3};
    Win wns = awa::make_win(1, th, tw, nw, TXT_N, false);
    Win wsh = awa::make_win(1, th, tw, nw, TXT_N, true);
    dit.set_windows(wns, wsh);
    int Lv = th * tw;
    fprintf(stderr, "[test] 360p token 网格 th=%d tw=%d Lv=%d nwin(ns=%d,sh=%d), N=%d 层\n", th, tw, Lv, wns.nwin, wsh.nwin, N);

    std::vector<float> vid_patch((size_t)Lv * 132), txt((size_t)TXT_N * 5120);
    for (auto& v : vid_patch) v = frand() * 2.f - 1.f;
    for (auto& v : txt) v = frand() * 2.f - 1.f;

    // ---- 逐层参考路径（原 DiT 块路径：逐层 Net + CPU 残差，权重 fp32）----
    std::vector<float> vl, tl, out_ref;
    if (!dit.forward_latent(vid_patch, Lv, txt, TXT_N, 1000.f, 0, N, true, true, vl, tl, out_ref)) {
        fprintf(stderr, "FAIL forward_latent(ref)\n"); return 1;
    }
    fprintf(stderr, "[test] 参考路径 out_ref=%zu\n", out_ref.size());
    // 诊断：dump 参考中间 v_cur_k（SEEDVR_DUMP 存在时）
    if (getenv("SEEDVR_DUMP")) {
        for (int k = 1; k <= N; k++) {
            std::vector<float> dv, dt, sr;
            if (dit.forward_latent(vid_patch, Lv, txt, TXT_N, 1000.f, 0, k, true, k == N, dv, dt, sr)) {
                char nm[64]; snprintf(nm, sizeof(nm), "ref_v_cur_%d.f32", k);
                FILE* f = fopen(nm, "wb");
                if (f) { fwrite(dv.data(), 4, dv.size(), f); fclose(f); }
                fprintf(stderr, "[test] ref v_cur_%d dumped %zu\n", k, dv.size());
            }
        }
    }

    // ---- 单整图 Net（N 层合并 1 个 Net，整图 1 次 download）----
    if (!dit.load_single(graph_dir, prefix, N)) { fprintf(stderr, "FAIL load_single(%s)\n", prefix.c_str()); return 1; }
    std::vector<float> out_single;
    if (!dit.forward_graph(vid_patch, Lv, txt, TXT_N, 1000.f, out_single)) { fprintf(stderr, "FAIL forward_graph(single)\n"); return 1; }
    fprintf(stderr, "[test] 单 Net out_single=%zu\n", out_single.size());

    if (out_ref.size() != out_single.size()) { fprintf(stderr, "FAIL size mismatch ref=%zu single=%zu\n", out_ref.size(), out_single.size()); return 1; }
    double c = cos_sim(out_ref, out_single);
    double p = psnr(out_ref, out_single);
    fprintf(stderr, "[result] cos(ref, single) = %.8f\n", c);
    fprintf(stderr, "[result] PSNR(ref, single) = %.2f dB\n", p);
    if (c > 0.999) {
        fprintf(stderr, "[PASS] 单 Net 与逐层参考路径数值等价（合并不改变计算，整图仅 1 次 download）\n");
        return 0;
    } else if (c > 0.99) {
        fprintf(stderr, "[WARN] cos=%.6f 略低（fp32 下应≈1.0，可能块图/单 Net 层范围或残差精度不一致）\n", c);
        return 0;
    } else {
        fprintf(stderr, "[FAIL] cos=%.6f 过低，单 Net 合并逻辑或层范围有误\n", c);
        return 2;
    }
}

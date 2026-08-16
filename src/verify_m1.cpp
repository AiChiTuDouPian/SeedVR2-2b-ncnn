// M1 验证：NaPatchIn(patchify 2x2 + vid_in.proj Linear) 与 fusedrms RMSNorm
// 对照 models/m1/*.bin 参考（raw 格式：int64 ndim + int64[ndim] shape + float32 数据）
#include <ncnn/net.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cstring>

static std::vector<float> load_raw(const char* path, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[FAIL] 无法打开 %s\n", path); exit(1); }
    int64_t ndim; f.read((char*)&ndim, 8);
    shape.resize(ndim);
    for (auto& d : shape) f.read((char*)&d, 8);
    int64_t total = 1; for (auto d : shape) total *= d;
    std::vector<float> data(total);
    f.read((char*)data.data(), total * 4);
    return data;
}

static std::vector<float> run_linear(const char* param, const char* bin, const std::vector<float>& x) {
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.load_param(param);
    net.load_model(bin);
    int n = (int)x.size();  // 单个样本：w=num_in, h=1
    ncnn::Mat in(n, 1, 1);
    memcpy(in, x.data(), n * sizeof(float));
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);
    std::vector<float> res(out.w * out.h * out.c);
    memcpy(res.data(), out, res.size() * sizeof(float));
    return res;
}

// patchify: (T,H,W,c) -> 2x2 空间收集 -> (T*H/2*W/2, 4*c)，t=1
static std::vector<float> patchify_2x2(const std::vector<float>& vid, int T, int H, int W, int c) {
    int H2 = H / 2, W2 = W / 2;
    int Lp = T * H2 * W2;
    std::vector<float> patches(Lp * (2 * 2 * c));
    for (int t = 0; t < T; t++)
        for (int hh = 0; hh < H2; hh++)
            for (int ww = 0; ww < W2; ww++) {
                int pid = t * H2 * W2 + hh * W2 + ww;
                for (int dh = 0; dh < 2; dh++)
                    for (int dw = 0; dw < 2; dw++)
                        for (int cc = 0; cc < c; cc++) {
                            int h = hh * 2 + dh, w = ww * 2 + dw;
                            int src = (t * H * W + h * W + w) * c + cc;
                            int dst = pid * (2 * 2 * c) + (dh * 2 + dw) * c + cc;
                            patches[dst] = vid[src];
                        }
            }
    return patches;
}

static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return (float)(dot / (sqrt(na) * sqrt(nb) + 1e-12));
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static int check(const char* name, const std::vector<float>& got, const std::vector<float>& ref,
                 float cos_thr, float diff_thr) {
    float c = cosine(got, ref);
    float d = max_abs_diff(got, ref);
    fprintf(stderr, "[%s] cos=%.6f  max|diff|=%.5f  (ref 范数 %.3f)\n", name, c, d,
            sqrtf((float)0));
    if (c >= cos_thr && d <= diff_thr) { fprintf(stderr, "   -> PASS\n"); return 0; }
    fprintf(stderr, "   -> FAIL\n"); return 1;
}

int main() {
    int rc = 0;

    // ---- 1) NaPatchIn: patchify + vid_in.proj ----
    std::vector<int64_t> sh;
    auto vid = load_raw("models/m1/vid_in_vidseq.bin", sh);   // (L=16, 33)
    int L = (int)sh[0], c = (int)sh[1];
    int T = 1, H = 4, W = 4;  // M1 固定测试形状
    auto patches = patchify_2x2(vid, T, H, W, c);
    int n_in = 2 * 2 * c;                 // 132
    int n_rows = (int)patches.size() / n_in;  // 4 个 patch
    // ncnn InnerProduct 不保留 batch 维，逐 patch 调用并拼接
    std::vector<float> got_in(n_rows * 2560);
    for (int r = 0; r < n_rows; r++) {
        std::vector<float> one(patches.begin() + r * n_in, patches.begin() + (r + 1) * n_in);
        auto o = run_linear("models/m1/vid_in_proj.param", "models/m1/vid_in_proj.bin", one);
        memcpy(got_in.data() + r * o.size(), o.data(), o.size() * sizeof(float));
    }
    auto ref_in = load_raw("models/m1/vid_in_expected.bin", sh);
    fprintf(stderr, "[info] NaPatchIn: patches=%zu got=%zu ref=%zu\n", patches.size(), got_in.size(), ref_in.size());
    rc |= check("NaPatchIn(patchify+linear)", got_in, ref_in, 0.999f, 0.05f);

    // ---- 2) fusedrms RMSNorm ----
    auto x = load_raw("models/m1/rmsnorm_x.bin", sh);          // (N,128)
    auto w = load_raw("models/m1/rmsnorm_w.bin", sh);          // (128,)
    auto ref_rn = load_raw("models/m1/rmsnorm_expected.bin", sh);
    int N = (int)sh[0], D = (int)sh[1];
    std::vector<float> got_rn(N * D);
    float eps = 1e-6f;
    for (int i = 0; i < N; i++) {
        float ss = 0;
        for (int j = 0; j < D; j++) { float v = x[i * D + j]; ss += v * v; }
        float rms = sqrtf(ss / D + eps);
        for (int j = 0; j < D; j++) got_rn[i * D + j] = x[i * D + j] / rms * w[j];
    }
    rc |= check("fusedrms RMSNorm", got_rn, ref_rn, 0.9999f, 0.01f);

    if (rc == 0) fprintf(stderr, "\n[PASS] M1 验证全部通过\n");
    else fprintf(stderr, "\n[FAIL] M1 存在失败项\n");
    return rc;
}

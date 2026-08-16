// M2 验证：单个 block 的 attn(qkv/qk_norm/out) + mlp(SwiGLU)
// 非窗口 attention（window=(1,1,1) 退化 -> 全局 attention，不含 RoPE）
// 对照 models/m2/block{N}_*.bin 参考。验证 block 0(dual) 与 block 20(shared)。
#include <ncnn/net.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cstring>

static std::vector<float> load_raw(const char* path, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[FAIL] 打开 %s\n", path); exit(1); }
    int64_t ndim; f.read((char*)&ndim, 8);
    shape.resize(ndim);
    for (auto& d : shape) f.read((char*)&d, 8);
    int64_t total = 1; for (auto d : shape) total *= d;
    std::vector<float> data(total);
    f.read((char*)data.data(), total * 4);
    return data;
}

struct Linear {
    ncnn::Net net;
    int num_out;
    void load(const char* param, const char* bin) {
        net.opt.use_vulkan_compute = false;
        net.load_param(param); net.load_model(bin);
    }
    std::vector<float> run(const std::vector<float>& x) {
        int n = (int)x.size();
        ncnn::Mat in(n, 1, 1);
        memcpy(in, x.data(), n * sizeof(float));
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", in);
        ncnn::Mat out; ex.extract("out0", out);
        std::vector<float> r(out.w * out.h * out.c);
        memcpy(r.data(), out, r.size() * sizeof(float));
        return r;
    }
};

static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return (float)(dot / (sqrt(na) * sqrt(nb) + 1e-12));
}
static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0; for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i] - b[i])); return m;
}
static int check(const char* name, const std::vector<float>& got, const std::vector<float>& ref,
                 float cos_thr, float diff_thr) {
    float c = cosine(got, ref), d = max_abs_diff(got, ref);
    fprintf(stderr, "[%s] cos=%.6f max|diff|=%.5f\n", name, c, d);
    if (c >= cos_thr && d <= diff_thr) { fprintf(stderr, "   -> PASS\n"); return 0; }
    fprintf(stderr, "   -> FAIL\n"); return 1;
}

static const int HEADS = 20, HEAD_D = 128, DIM = 2560;
static const float EPS = 1e-6f;

// 对 (L,2560) 里每个 token 跑 attn 子图（退化全局窗口），返回 (L,2560)
// 注意：attention 必须逐 head 计算（与 Python 参考一致）——每个 head 用各自的 Q/K/V
// 与 softmax，绝不能把 20 个 head 的 dot 先求和再做一次 softmax（会溢出成 NaN）。
static std::vector<float> run_attn(Linear& qkv, Linear& outp,
                                   const std::vector<float>& normq,
                                   const std::vector<float>& normk,
                                   const std::vector<float>& X, int L) {
    // 1) qkv 投影
    std::vector<float> Q(L * DIM), K(L * DIM), V(L * DIM);
    for (int i = 0; i < L; i++) {
        std::vector<float> x(X.begin() + i * DIM, X.begin() + (i + 1) * DIM);
        auto qkv_i = qkv.run(x);  // 7680
        for (int j = 0; j < DIM; j++) { Q[i * DIM + j] = qkv_i[j]; K[i * DIM + j] = qkv_i[DIM + j]; V[i * DIM + j] = qkv_i[2 * DIM + j]; }
    }
    // 2) qk_norm（每 head 128 维，rms 不减均值）
    // normq/normk 权重形状为 (128,) —— 跨所有 head 共享的 per-head_dim 权重，
    // 故下标只用 d，不要写成 normq[h*HEAD_D+d]（越界读到垃圾 -> NaN）。
    for (int i = 0; i < L; i++) {
        for (int h = 0; h < HEADS; h++) {
            float* q = &Q[i * DIM + h * HEAD_D];
            float* k = &K[i * DIM + h * HEAD_D];
            float sq = 0, sk = 0;
            for (int d = 0; d < HEAD_D; d++) { sq += q[d] * q[d]; sk += k[d] * k[d]; }
            float rq = sqrtf(sq / HEAD_D + EPS), rk = sqrtf(sk / HEAD_D + EPS);
            for (int d = 0; d < HEAD_D; d++) { q[d] = q[d] / rq * normq[d]; k[d] = k[d] / rk * normk[d]; }
        }
    }
    // 3) 逐 head attention（每个 head 独立 softmax）= softmax(QK^T/sqrt(d)) V
    std::vector<float> attn(L * DIM);
    for (int i = 0; i < L; i++) {
        for (int h = 0; h < HEADS; h++) {
            const float* qi = &Q[i * DIM + h * HEAD_D];
            std::vector<float> scores(L);
            float mx = -1e30f;
            for (int j = 0; j < L; j++) {
                const float* kj = &K[j * DIM + h * HEAD_D];
                float dot = 0; for (int d = 0; d < HEAD_D; d++) dot += qi[d] * kj[d];
                scores[j] = dot / sqrtf((float)HEAD_D);
                if (scores[j] > mx) mx = scores[j];
            }
            float sum = 0; for (int j = 0; j < L; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
            for (int j = 0; j < L; j++) scores[j] /= sum;
            float* o = &attn[i * DIM + h * HEAD_D];
            for (int d = 0; d < HEAD_D; d++) o[d] = 0;
            for (int j = 0; j < L; j++) {
                const float* vj = &V[j * DIM + h * HEAD_D];
                for (int d = 0; d < HEAD_D; d++) o[d] += scores[j] * vj[d];
            }
        }
    }
    // 4) proj_out（逐 token）
    std::vector<float> final(L * DIM);
    for (int i = 0; i < L; i++) {
        std::vector<float> a(attn.begin() + i * DIM, attn.begin() + (i + 1) * DIM);
        auto o = outp.run(a);
        memcpy(&final[i * DIM], o.data(), DIM * sizeof(float));
    }
    return final;
}

static std::vector<float> run_mlp(Linear& in_g, Linear& in_u, Linear& outp,
                                  const std::vector<float>& X, int L) {
    std::vector<float> final(L * DIM);
    for (int i = 0; i < L; i++) {
        std::vector<float> x(X.begin() + i * DIM, X.begin() + (i + 1) * DIM);
        auto gate = in_g.run(x);   // 6912
        auto up = in_u.run(x);     // 6912
        std::vector<float> h(gate.size());
        for (size_t j = 0; j < h.size(); j++) h[j] = (gate[j] / (1.f + expf(-gate[j]))) * up[j];
        auto o = outp.run(h);
        memcpy(&final[i * DIM], o.data(), DIM * sizeof(float));
    }
    return final;
}

static int verify_block(int block) {
    char base[64]; snprintf(base, sizeof(base), "models/m2/block%d", block);
    Linear qkv, outp, m_in_g, m_in_u, m_out;
    qkv.load((std::string(base) + "_attn_qkv.param").c_str(), (std::string(base) + "_attn_qkv.bin").c_str());
    outp.load((std::string(base) + "_attn_out.param").c_str(), (std::string(base) + "_attn_out.bin").c_str());
    m_in_g.load((std::string(base) + "_mlp_in_gate.param").c_str(), (std::string(base) + "_mlp_in_gate.bin").c_str());
    m_in_u.load((std::string(base) + "_mlp_in.param").c_str(), (std::string(base) + "_mlp_in.bin").c_str());
    m_out.load((std::string(base) + "_mlp_out.param").c_str(), (std::string(base) + "_mlp_out.bin").c_str());
    std::vector<int64_t> sh;
    auto normq = load_raw((std::string(base) + "_normq.bin").c_str(), sh);
    auto normk = load_raw((std::string(base) + "_normk.bin").c_str(), sh);
    auto Xa = load_raw((std::string(base) + "_attn_x.bin").c_str(), sh);
    auto ref_a = load_raw((std::string(base) + "_attn_expected.bin").c_str(), sh);
    auto Xm = load_raw((std::string(base) + "_mlp_x.bin").c_str(), sh);
    auto ref_m = load_raw((std::string(base) + "_mlp_expected.bin").c_str(), sh);
    int L = (int)Xa.size() / DIM;

    auto got_a = run_attn(qkv, outp, normq, normk, Xa, L);
    auto got_m = run_mlp(m_in_g, m_in_u, m_out, Xm, L);
    char nm[64];
    snprintf(nm, sizeof(nm), "block%d attn", block);
    int rc = check(nm, got_a, ref_a, 0.999f, 0.05f);
    snprintf(nm, sizeof(nm), "block%d mlp", block);
    rc |= check(nm, got_m, ref_m, 0.999f, 0.05f);
    return rc;
}

int main() {
    int rc = 0;
    rc |= verify_block(0);    // dual/vid
    rc |= verify_block(20);   // shared/.all
    if (rc == 0) fprintf(stderr, "\n[PASS] M2 验证全部通过\n");
    else fprintf(stderr, "\n[FAIL] M2 存在失败项\n");
    return rc;
}

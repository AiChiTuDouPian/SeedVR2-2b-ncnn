// M4 验证：shifted NaSwinAttention（720pswin_by_size_bysize）的 AWA。
// 与 M3 区别：27 个变长窗口（边界更小）；文本逐窗口重复，attention 后跨窗口平均(coalesce)。
// 视频 reverse 仍精确逆（shifted 窗口非重叠、完美平铺 M=L）。
#include <ncnn/net.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <fstream>

static const int HEADS = 20, HEAD_D = 128, DIM = 2560;
static const float EPS = 1e-6f;

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
static std::vector<int64_t> load_raw_i64(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[FAIL] 打开 %s\n", path); exit(1); }
    int64_t n; f.read((char*)&n, 8);
    std::vector<int64_t> data(n);
    f.read((char*)data.data(), n * 8);
    return data;
}

struct Linear {
    ncnn::Net net;
    void load(const char* param, const char* bin) {
        net.opt.use_vulkan_compute = false;
        net.load_param(param); net.load_model(bin);
    }
    std::vector<float> run(const std::vector<float>& x) {
        int n = (int)x.size();
        ncnn::Mat in(n, 1, 1); memcpy(in, x.data(), n * sizeof(float));
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
    for (size_t i = 0; i < a.size(); i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return (float)(dot / (sqrt(na)*sqrt(nb) + 1e-12));
}
static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0; for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::fabs(a[i]-b[i])); return m;
}
static int check(const char* name, const std::vector<float>& got, const std::vector<float>& ref,
                 float cos_thr, float diff_thr) {
    float c = cosine(got, ref), d = max_abs_diff(got, ref);
    fprintf(stderr, "[%s] cos=%.6f max|diff|=%.5f\n", name, c, d);
    if (c >= cos_thr && d <= diff_thr) { fprintf(stderr, "   -> PASS\n"); return 0; }
    fprintf(stderr, "   -> FAIL\n"); return 1;
}

static void qk_norm(const std::vector<float>& x, const float* w, std::vector<float>& out) {
    for (int h = 0; h < HEADS; h++) {
        const float* xp = &x[h * HEAD_D];
        float* op = &out[h * HEAD_D];
        float sq = 0; for (int d = 0; d < HEAD_D; d++) sq += xp[d]*xp[d];
        float r = sqrtf(sq / HEAD_D + EPS);
        for (int d = 0; d < HEAD_D; d++) op[d] = xp[d] / r * w[d];
    }
}

static void apply_rope(const float* freq, const float* q, float* out) {
    const int ROT = 126, HALF = ROT / 2;
    for (int k = 0; k < ROT; k++) {
        float c = cosf(freq[k]), s = sinf(freq[k]);
        float rh = (k % 2 == 0) ? -q[k + 1] : q[k - 1];  // 相邻配对
        out[k] = q[k] * c + rh * s;
    }
    for (int k = ROT; k < HEAD_D; k++) out[k] = q[k];
}

int main() {
    const char* M4 = "models/m4/";
    Linear qkv_vid, qkv_txt, out_vid, out_txt;
    qkv_vid.load((std::string(M4)+"b0_qkv_vid.param").c_str(), (std::string(M4)+"b0_qkv_vid.bin").c_str());
    qkv_txt.load((std::string(M4)+"b0_qkv_txt.param").c_str(), (std::string(M4)+"b0_qkv_txt.bin").c_str());
    out_vid.load((std::string(M4)+"b0_out_vid.param").c_str(), (std::string(M4)+"b0_out_vid.bin").c_str());
    out_txt.load((std::string(M4)+"b0_out_txt.param").c_str(), (std::string(M4)+"b0_out_txt.bin").c_str());
    std::vector<int64_t> sh;
    auto normq_v = load_raw((std::string(M4)+"normq_vid.bin").c_str(), sh);
    auto normk_v = load_raw((std::string(M4)+"normk_vid.bin").c_str(), sh);
    auto normq_t = load_raw((std::string(M4)+"normq_txt.bin").c_str(), sh);
    auto normk_t = load_raw((std::string(M4)+"normk_txt.bin").c_str(), sh);

    auto Xv = load_raw((std::string(M4)+"x_vid.bin").c_str(), sh);
    auto Xt = load_raw((std::string(M4)+"x_txt.bin").c_str(), sh);
    int Lv = (int)Xv.size() / DIM;
    int TXT = (int)Xt.size() / DIM;

    auto vid_freq = load_raw((std::string(M4)+"vid_freq.bin").c_str(), sh);   // (M,66)
    auto txt_freq = load_raw((std::string(M4)+"txt_freq.bin").c_str(), sh);   // (Nwin*l,66)
    auto cu = load_raw_i64((std::string(M4)+"cu_seqlens.bin").c_str());
    auto windef = load_raw_i64((std::string(M4)+"windows.bin").c_str());
    int t = (int)windef[0], h = (int)windef[1], w = (int)windef[2];
    int nwin = (int)windef[9];
    std::vector<int> wst(nwin), sen(nwin), sth(nwin), she(nwin), swt(nwin), swe(nwin);
    std::vector<int> f_list(nwin);
    for (int i = 0; i < nwin; i++) {
        wst[i] = (int)windef[10 + i*6 + 0]; sen[i] = (int)windef[10 + i*6 + 1];
        sth[i] = (int)windef[10 + i*6 + 2]; she[i] = (int)windef[10 + i*6 + 3];
        swt[i] = (int)windef[10 + i*6 + 4]; swe[i] = (int)windef[10 + i*6 + 5];
        f_list[i] = (sen[i]-wst[i]) * (she[i]-sth[i]) * (swe[i]-swt[i]);
    }
    int M = 0; for (int i = 0; i < nwin; i++) M += f_list[i];
    fprintf(stderr, "[info] shifted t=%d h=%d w=%d nwin=%d M=%d Lv=%d txt=%d\n", t,h,w,nwin,M,Lv,TXT);

    // ---- proj_qkv + qk_norm (video) ----
    std::vector<std::vector<float>> vq(Lv, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> vk(Lv, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> vv(Lv, std::vector<float>(HEADS*HEAD_D));
    for (int i = 0; i < Lv; i++) {
        auto qkv = qkv_vid.run(std::vector<float>(Xv.begin()+i*DIM, Xv.begin()+(i+1)*DIM));
        std::vector<float> q(HEADS*HEAD_D), k(HEADS*HEAD_D), v(HEADS*HEAD_D);
        for (int hh = 0; hh < HEADS; hh++)
            for (int d = 0; d < HEAD_D; d++) {
                q[hh*HEAD_D+d] = qkv[hh*HEAD_D+d];
                k[hh*HEAD_D+d] = qkv[HEADS*HEAD_D + hh*HEAD_D+d];
                v[hh*HEAD_D+d] = qkv[2*HEADS*HEAD_D + hh*HEAD_D+d];
            }
        qk_norm(q, normq_v.data(), vq[i]);
        qk_norm(k, normk_v.data(), vk[i]);
        for (int hh = 0; hh < HEADS*HEAD_D; hh++) vv[i][hh] = v[hh];
    }
    std::vector<std::vector<float>> tq(TXT, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> tk(TXT, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> tv(TXT, std::vector<float>(HEADS*HEAD_D));
    for (int i = 0; i < TXT; i++) {
        auto qkv = qkv_txt.run(std::vector<float>(Xt.begin()+i*DIM, Xt.begin()+(i+1)*DIM));
        std::vector<float> q(HEADS*HEAD_D), k(HEADS*HEAD_D), v(HEADS*HEAD_D);
        for (int hh = 0; hh < HEADS; hh++)
            for (int d = 0; d < HEAD_D; d++) {
                q[hh*HEAD_D+d] = qkv[hh*HEAD_D+d];
                k[hh*HEAD_D+d] = qkv[HEADS*HEAD_D + hh*HEAD_D+d];
                v[hh*HEAD_D+d] = qkv[2*HEADS*HEAD_D + hh*HEAD_D+d];
            }
        qk_norm(q, normq_t.data(), tq[i]);
        qk_norm(k, normk_t.data(), tk[i]);
        for (int hh = 0; hh < HEADS*HEAD_D; hh++) tv[i][hh] = v[hh];
    }

    // ---- window partition (变长) -> vqw/gkw/vvw (M tokens, 全局计数 g) ----
    std::vector<std::vector<float>> vqw(M, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> vkw(M, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> vvw(M, std::vector<float>(HEADS*HEAD_D));
    int g = 0;
    for (int wi = 0; wi < nwin; wi++) {
        int lt_max = sen[wi]-wst[wi], lh_max = she[wi]-sth[wi], lw_max = swe[wi]-swt[wi];
        for (int lt = 0; lt < lt_max; lt++)
            for (int lh = 0; lh < lh_max; lh++)
                for (int lw = 0; lw < lw_max; lw++) {
                    int it = wst[wi]+lt, ih = sth[wi]+lh, iw = swt[wi]+lw;
                    int idx = ((it*h) + ih)*w + iw;
                    vqw[g] = vq[idx]; vkw[g] = vk[idx]; vvw[g] = vv[idx];
                    g++;
                }
    }

    // ---- RoPE apply (video, 全局 g) ----
    for (int i = 0; i < M; i++) {
        const float* fr = &vid_freq[i*66];
        for (int hh = 0; hh < HEADS; hh++) {
            float in[HEAD_D], out[HEAD_D];
            memcpy(in, &vqw[i][hh*HEAD_D], HEAD_D*4);
            apply_rope(fr, in, out); memcpy(&vqw[i][hh*HEAD_D], out, HEAD_D*4);
            memcpy(in, &vkw[i][hh*HEAD_D], HEAD_D*4);
            apply_rope(fr, in, out); memcpy(&vkw[i][hh*HEAD_D], out, HEAD_D*4);
        }
    }
    // 文本逐窗口重复 + RoPE（文本 freq 与窗口无关）
    std::vector<std::vector<float>> tqw(nwin*TXT, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> tkw(nwin*TXT, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> tvw(nwin*TXT, std::vector<float>(HEADS*HEAD_D));
    for (int wi = 0; wi < nwin; wi++) {
        for (int i = 0; i < TXT; i++) {
            int dst = wi*TXT + i;
            tqw[dst] = tq[i]; tkw[dst] = tk[i]; tvw[dst] = tv[i];
            const float* fr = &txt_freq[dst*66];
            for (int hh = 0; hh < HEADS; hh++) {
                float in[HEAD_D], out[HEAD_D];
                memcpy(in, &tqw[dst][hh*HEAD_D], HEAD_D*4);
                apply_rope(fr, in, out); memcpy(&tqw[dst][hh*HEAD_D], out, HEAD_D*4);
                memcpy(in, &tkw[dst][hh*HEAD_D], HEAD_D*4);
                apply_rope(fr, in, out); memcpy(&tkw[dst][hh*HEAD_D], out, HEAD_D*4);
            }
        }
    }

    // ---- varlen concat + 每窗口 SDPA（变长）----
    std::vector<std::vector<float>> out_v(M, std::vector<float>(HEADS*HEAD_D));   // 视频输出(全局g)
    std::vector<std::vector<float>> out_t(nwin*TXT, std::vector<float>(HEADS*HEAD_D)); // 文本输出
    int g0 = 0;
    for (int wi = 0; wi < nwin; wi++) {
        int f_i = f_list[wi];
        int S = f_i + TXT;
        std::vector<std::vector<float>> segq(S), segk(S), segv(S);
        for (int j = 0; j < f_i; j++) { segq[j]=vqw[g0+j]; segk[j]=vkw[g0+j]; segv[j]=vvw[g0+j]; }
        for (int j = 0; j < TXT; j++) { segq[f_i+j]=tqw[wi*TXT+j]; segk[f_i+j]=tkw[wi*TXT+j]; segv[f_i+j]=tvw[wi*TXT+j]; }
        for (int hh = 0; hh < HEADS; hh++) {
            std::vector<std::vector<float>> sc(S, std::vector<float>(S));
            float mx = -1e30f;
            for (int a = 0; a < S; a++) {
                const float* qa = &segq[a][hh*HEAD_D];
                for (int b = 0; b < S; b++) {
                    const float* kb = &segk[b][hh*HEAD_D];
                    float dot = 0; for (int d = 0; d < HEAD_D; d++) dot += qa[d]*kb[d];
                    dot /= sqrtf((float)HEAD_D);
                    sc[a][b] = dot; if (dot > mx) mx = dot;
                }
            }
            float sum = 0; std::vector<float> p(S);
            for (int a = 0; a < S; a++) {
                float sm = 0; for (int b = 0; b < S; b++) { float e = expf(sc[a][b]-mx); p[b]=e; sm+=e; }
                for (int b = 0; b < S; b++) p[b] /= sm;
                std::vector<float> o(HEAD_D, 0.f);
                for (int b = 0; b < S; b++) {
                    const float* vb = &segv[b][hh*HEAD_D];
                    for (int d = 0; d < HEAD_D; d++) o[d] += p[b]*vb[d];
                }
                // a<f_i 写视频输出；a>=f_i 写文本输出（跨窗口后平均）
                if (a < f_i) memcpy(&out_v[g0 + a][hh*HEAD_D], o.data(), HEAD_D*4);
                else memcpy(&out_t[wi*TXT + (a - f_i)][hh*HEAD_D], o.data(), HEAD_D*4);
            }
        }
        g0 += f_i;
    }

    // ---- 视频 reverse（写回）----
    std::vector<std::vector<float>> vout_rev(Lv, std::vector<float>(HEADS*HEAD_D));
    g = 0;
    for (int wi = 0; wi < nwin; wi++) {
        int lt_max = sen[wi]-wst[wi], lh_max = she[wi]-sth[wi], lw_max = swe[wi]-swt[wi];
        for (int lt = 0; lt < lt_max; lt++)
            for (int lh = 0; lh < lh_max; lh++)
                for (int lw = 0; lw < lw_max; lw++) {
                    int it = wst[wi]+lt, ih = sth[wi]+lh, iw = swt[wi]+lw;
                    int idx = ((it*h) + ih)*w + iw;
                    vout_rev[idx] = out_v[g];
                    g++;
                }
    }

    // ---- 文本 coalesce 平均（跨窗口）----
    std::vector<std::vector<float>> tout_avg(TXT, std::vector<float>(HEADS*HEAD_D));
    for (int tkn = 0; tkn < TXT; tkn++) {
        for (int hh = 0; hh < HEADS*HEAD_D; hh++) {
            float s = 0;
            for (int wi = 0; wi < nwin; wi++) s += out_t[wi*TXT + tkn][hh];
            tout_avg[tkn][hh] = s / (float)nwin;
        }
    }

    // ---- proj_out ----
    std::vector<float> vid_out(Lv*DIM);
    for (int i = 0; i < Lv; i++) {
        std::vector<float> tok(HEADS*HEAD_D);
        for (int hh = 0; hh < HEADS*HEAD_D; hh++) tok[hh] = vout_rev[i][hh];
        auto o = out_vid.run(tok);
        memcpy(&vid_out[i*DIM], o.data(), DIM*sizeof(float));
    }
    std::vector<float> txt_out(TXT*DIM);
    for (int i = 0; i < TXT; i++) {
        std::vector<float> tok(HEADS*HEAD_D);
        for (int hh = 0; hh < HEADS*HEAD_D; hh++) tok[hh] = tout_avg[i][hh];
        auto o = out_txt.run(tok);
        memcpy(&txt_out[i*DIM], o.data(), DIM*sizeof(float));
    }

    // ---- 对照 ----
    auto ref_v = load_raw((std::string(M4)+"ref_vid_out.bin").c_str(), sh);
    auto ref_t = load_raw((std::string(M4)+"ref_txt_out.bin").c_str(), sh);
    int rc = 0;
    rc |= check("awa_shifted vid_out", vid_out, ref_v, 0.995f, 0.05f);
    rc |= check("awa_shifted txt_out", txt_out, ref_t, 0.995f, 0.05f);
    if (rc == 0) fprintf(stderr, "\n[PASS] M4 shifted-AWA 验证通过\n");
    else fprintf(stderr, "\n[FAIL] M4 shifted-AWA 存在失败项\n");
    return rc;
}

// M3 验证：NaSwinAttention（非 shifted）的 AWA 算子
// window partition -> mmrope3d -> varlen concat(文本逐窗口重复) -> 窗口内 SDPA
// -> window reverse -> proj_out。对照 models/m3/ 参考。
#include <ncnn/net.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cstring>

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

// fusedrms qk_norm（权重 128，跨 head 共享）
static void qk_norm(const std::vector<float>& x /* (2560,) */, const float* w /*128*/,
                    std::vector<float>& out /* (20,128) */) {
    for (int h = 0; h < HEADS; h++) {
        const float* xp = &x[h * HEAD_D];
        float* op = &out[h * HEAD_D];
        float sq = 0; for (int d = 0; d < HEAD_D; d++) sq += xp[d]*xp[d];
        float r = sqrtf(sq / HEAD_D + EPS);
        for (int d = 0; d < HEAD_D; d++) op[d] = xp[d] / r * w[d];
    }
}

// mmrope3d apply：每头旋转 3×42=126 维，相邻配对 rotate_half，其余不变
static void apply_rope(const float* freq /*126*/, const float* q /*128*/, float* out /*128*/) {
    const int ROT = 126, HALF = ROT / 2;  // 63
    for (int k = 0; k < ROT; k++) {
        float c = cosf(freq[k]), s = sinf(freq[k]);
        // rotary_embedding_torch 的 freq 布局 [f0,f0,f1,f1,...]：相邻配对 (2i,2i+1)
        float rh = (k % 2 == 0) ? -q[k + 1] : q[k - 1];
        out[k] = q[k] * c + rh * s;
    }
    for (int k = ROT; k < HEAD_D; k++) out[k] = q[k];
}

int main() {
    const char* M3 = "models/m3/";
    // ---- 权重 ----
    Linear qkv_vid, qkv_txt, out_vid, out_txt;
    qkv_vid.load((std::string(M3)+"b0_qkv_vid.param").c_str(), (std::string(M3)+"b0_qkv_vid.bin").c_str());
    qkv_txt.load((std::string(M3)+"b0_qkv_txt.param").c_str(), (std::string(M3)+"b0_qkv_txt.bin").c_str());
    out_vid.load((std::string(M3)+"b0_out_vid.param").c_str(), (std::string(M3)+"b0_out_vid.bin").c_str());
    out_txt.load((std::string(M3)+"b0_out_txt.param").c_str(), (std::string(M3)+"b0_out_txt.bin").c_str());
    std::vector<int64_t> sh;
    auto normq_v = load_raw((std::string(M3)+"normq_vid.bin").c_str(), sh);
    auto normk_v = load_raw((std::string(M3)+"normk_vid.bin").c_str(), sh);
    auto normq_t = load_raw((std::string(M3)+"normq_txt.bin").c_str(), sh);
    auto normk_t = load_raw((std::string(M3)+"normk_txt.bin").c_str(), sh);

    // ---- 输入 ----
    auto Xv = load_raw((std::string(M3)+"x_vid.bin").c_str(), sh);   // (Lv,2560)
    auto Xt = load_raw((std::string(M3)+"x_txt.bin").c_str(), sh);   // (l,2560)
    int Lv = (int)Xv.size() / DIM;
    int TXT = (int)Xt.size() / DIM;

    auto vid_freq = load_raw((std::string(M3)+"vid_freq.bin").c_str(), sh);  // (Nwin*f,66)
    auto txt_freq = load_raw((std::string(M3)+"txt_freq.bin").c_str(), sh);  // (Nwin*l,66)
    auto cu = load_raw_i64((std::string(M3)+"cu_seqlens.bin").c_str());
    auto windef = load_raw_i64((std::string(M3)+"windows.bin").c_str());     // 10 header + nwin*6
    int t = (int)windef[0], h = (int)windef[1], w = (int)windef[2];
    int wt = (int)windef[3], wh = (int)windef[4], ww = (int)windef[5];
    int nnt = (int)windef[6], nnh = (int)windef[7], nnw = (int)windef[8];
    int nwin = (int)windef[9];
    int per_win_f = wt * wh * ww;
    std::vector<int> wst(nwin), sen(nwin), sth(nwin), she(nwin), swt(nwin), swe(nwin);
    for (int i = 0; i < nwin; i++) {
        wst[i] = (int)windef[10 + i*6 + 0]; sen[i] = (int)windef[10 + i*6 + 1];
        sth[i] = (int)windef[10 + i*6 + 2]; she[i] = (int)windef[10 + i*6 + 3];
        swt[i] = (int)windef[10 + i*6 + 4]; swe[i] = (int)windef[10 + i*6 + 5];
    }
    fprintf(stderr, "[info] t=%d h=%d w=%d nwin=%d per_win_f=%d txt=%d\n", t,h,w,nwin,per_win_f,TXT);
    { double sx=0; for(float vv:Xv) sx+=vv; fprintf(stderr, "[DIAG] Xv sum=%.3f Xv[0:3]=%.3f %.3f %.3f\n", sx, Xv[0],Xv[1],Xv[2]); }
    { double sn=0; for(float vv:normq_v) sn+=vv; fprintf(stderr, "[DIAG] normq_v sum=%.3f normq_v[0:3]=%.3f %.3f %.3f shape=%lld\n", sn, normq_v[0],normq_v[1],normq_v[2],(long long)normq_v.size()); }

    // ---- proj_qkv + qk_norm ----
    std::vector<std::vector<float>> vq(Lv, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> vk(Lv, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> vv(Lv, std::vector<float>(HEADS*HEAD_D));
    for (int i = 0; i < Lv; i++) {
        auto qkv = qkv_vid.run(std::vector<float>(Xv.begin()+i*DIM, Xv.begin()+(i+1)*DIM)); // 7680
        if (i == 0) fprintf(stderr, "[DIAG] vid qkv_i.size=%d\n", (int)qkv.size());
        std::vector<float> q(HEADS*HEAD_D), k(HEADS*HEAD_D), v(HEADS*HEAD_D);
        for (int hh = 0; hh < HEADS; hh++)
            for (int d = 0; d < HEAD_D; d++) {
                q[hh*HEAD_D+d] = qkv[hh*HEAD_D+d];
                k[hh*HEAD_D+d] = qkv[HEADS*HEAD_D + hh*HEAD_D+d];
                v[hh*HEAD_D+d] = qkv[2*HEADS*HEAD_D + hh*HEAD_D+d];
            }
        qk_norm(q, normq_v.data(), vq[i]);
        qk_norm(k, normk_v.data(), vk[i]);
        // v 不需要 norm（模型 norm 只作用于 q,k）；直接拷贝
        for (int hh = 0; hh < HEADS*HEAD_D; hh++) vv[i][hh] = v[hh];
        if (i == 0) {
            double sq=0,sk=0,sv=0;
            for(float vv:q) sq+=vv; for(float vv:k) sk+=vv; for(float vv:v) sv+=vv;
            fprintf(stderr, "[DIAG] i0 qkv q sum=%.3f k sum=%.3f v sum=%.3f\n", sq,sk,sv);
            fprintf(stderr, "[DIAG] i0 q[0:3]=%.4f %.4f %.4f  vq[0:3]=%.4f %.4f %.4f\n", q[0],q[1],q[2], vq[0][0],vq[0][1],vq[0][2]);
        }
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

    // ---- window partition (video) -> (nwin*f, 20, 128) ----
    std::vector<std::vector<float>> vqw(nwin*per_win_f, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> vkw(nwin*per_win_f, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> vvw(nwin*per_win_f, std::vector<float>(HEADS*HEAD_D));
    for (int wi = 0; wi < nwin; wi++) {
        int lt_max = sen[wi]-wst[wi], lh_max = she[wi]-sth[wi], lw_max = swe[wi]-swt[wi];
        int local = 0;
        for (int lt = 0; lt < lt_max; lt++)
            for (int lh = 0; lh < lh_max; lh++)
                for (int lw = 0; lw < lw_max; lw++) {
                    int it = wst[wi]+lt, ih = sth[wi]+lh, iw = swt[wi]+lw;
                    int idx = ((it*h) + ih)*w + iw;
                    int dst = wi*per_win_f + local;
                    vqw[dst] = vq[idx]; vkw[dst] = vk[idx]; vvw[dst] = vv[idx];
                    local++;
                }
    }

    // ---- 诊断：partition+norm+qkv 是否对齐（与 python vqw_norope 比）----
    {
        auto ref_vqw = load_raw((std::string(M3)+"vqw_norope.bin").c_str(), sh);
        std::vector<float> flat(nwin*per_win_f*DIM);
        for (int i = 0; i < nwin*per_win_f; i++)
            memcpy(&flat[i*DIM], vqw[i].data(), DIM*sizeof(float));
        fprintf(stderr, "[DIAG pre-rope vid cos] = %.6f\n", cosine(flat, ref_vqw));
        fprintf(stderr, "[DIAG] cpp flat[0:4] = %.3f %.3f %.3f %.3f\n", flat[0], flat[1], flat[2], flat[3]);
        fprintf(stderr, "[DIAG] ref vqw[0:4] = %.3f %.3f %.3f %.3f\n", ref_vqw[0], ref_vqw[1], ref_vqw[2], ref_vqw[3]);
        // 也比对 raw qkv（不 norm）是否对齐
        float s_cpp=0, s_ref=0;
        for (size_t i=0;i<flat.size();i++){ s_cpp+=flat[i]; s_ref+=ref_vqw[i]; }
        fprintf(stderr, "[DIAG] sum cpp=%.3f sum ref=%.3f\n", s_cpp, s_ref);
    }

    // ---- RoPE apply (video) ----
    for (int i = 0; i < nwin*per_win_f; i++) {
        const float* fr = &vid_freq[i*66];
        for (int hh = 0; hh < HEADS; hh++) {
            float in[HEAD_D], out[HEAD_D];
            memcpy(in, &vqw[i][hh*HEAD_D], HEAD_D*4);
            apply_rope(fr, in, out); memcpy(&vqw[i][hh*HEAD_D], out, HEAD_D*4);
            memcpy(in, &vkw[i][hh*HEAD_D], HEAD_D*4);
            apply_rope(fr, in, out); memcpy(&vkw[i][hh*HEAD_D], out, HEAD_D*4);
        }
    }
    // 文本逐窗口重复 + RoPE
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

    // ---- varlen concat + 窗口内 SDPA ----
    int S = per_win_f + TXT;
    std::vector<std::vector<float>> out_all(nwin*S, std::vector<float>(HEADS*HEAD_D));
    for (int wi = 0; wi < nwin; wi++) {
        // 组装 q/k/v (S, 20, 128)
        std::vector<std::vector<float>> segq(S), segk(S), segv(S);
        for (int j = 0; j < per_win_f; j++) { segq[j]=vqw[wi*per_win_f+j]; segk[j]=vkw[wi*per_win_f+j]; segv[j]=vvw[wi*per_win_f+j]; }
        for (int j = 0; j < TXT; j++)    { segq[per_win_f+j]=tqw[wi*TXT+j]; segk[per_win_f+j]=tkw[wi*TXT+j]; segv[per_win_f+j]=tvw[wi*TXT+j]; }
        for (int hh = 0; hh < HEADS; hh++) {
            // scores (S,S)
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
                // out_a = sum_b p[b]*v[b]
                std::vector<float> o(HEAD_D, 0.f);
                for (int b = 0; b < S; b++) {
                    const float* vb = &segv[b][hh*HEAD_D];
                    for (int d = 0; d < HEAD_D; d++) o[d] += p[b]*vb[d];
                }
                memcpy(&out_all[wi*S + a][hh*HEAD_D], o.data(), HEAD_D*4);
            }
        }
    }

    // ---- unconcat: vid 取每段前 per_win_f，txt 取每段后 TXT ----
    std::vector<std::vector<float>> vout_win(nwin*per_win_f, std::vector<float>(HEADS*HEAD_D));
    std::vector<std::vector<float>> tout_win(nwin*TXT, std::vector<float>(HEADS*HEAD_D));
    for (int wi = 0; wi < nwin; wi++) {
        for (int j = 0; j < per_win_f; j++) vout_win[wi*per_win_f+j] = out_all[wi*S + j];
        for (int j = 0; j < TXT; j++)       tout_win[wi*TXT+j]      = out_all[wi*S + per_win_f + j];
    }

    // ---- window reverse (video) ----
    std::vector<std::vector<float>> vout_rev(Lv, std::vector<float>(HEADS*HEAD_D));
    for (int wi = 0; wi < nwin; wi++) {
        int lt_max = sen[wi]-wst[wi], lh_max = she[wi]-sth[wi], lw_max = swe[wi]-swt[wi];
        int local = 0;
        for (int lt = 0; lt < lt_max; lt++)
            for (int lh = 0; lh < lh_max; lh++)
                for (int lw = 0; lw < lw_max; lw++) {
                    int it = wst[wi]+lt, ih = sth[wi]+lh, iw = swt[wi]+lw;
                    int idx = ((it*h)+ih)*w + iw;
                    vout_rev[idx] = vout_win[wi*per_win_f + local];
                    local++;
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
        for (int hh = 0; hh < HEADS*HEAD_D; hh++) tok[hh] = tout_win[i][hh]; // 第一段文本
        auto o = out_txt.run(tok);
        memcpy(&txt_out[i*DIM], o.data(), DIM*sizeof(float));
    }

    // ---- 对照 ----
    auto ref_v = load_raw((std::string(M3)+"ref_vid_out.bin").c_str(), sh);
    auto ref_t = load_raw((std::string(M3)+"ref_txt_out.bin").c_str(), sh);
    int rc = 0;
    rc |= check("awa vid_out", vid_out, ref_v, 0.995f, 0.05f);
    rc |= check("awa txt_out", txt_out, ref_t, 0.995f, 0.05f);
    if (rc == 0) fprintf(stderr, "\n[PASS] M3 AWA 验证通过\n");
    else fprintf(stderr, "\n[FAIL] M3 AWA 存在失败项\n");
    return rc;
}

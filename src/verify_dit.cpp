// M5 全网络验证：完整 32 层 NaDiT 单次前向（C++ 忠实复现）对拍 models/m5/ref_vid_out.bin
// 权重直接解码 ncnn fp16 Linear（权重 tag 0x01306B47 + 可选 fp32 bias blob）。
// 与 m5_export.py 完全一致：patch_in -> 32 blocks -> vid_out_norm+ada -> vid_out.proj
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cstring>
#include <string>
#include <sstream>

static const int HEADS = 20, HEAD_D = 128, DIM = 2560, QKV = HEADS*HEAD_D*3;
static const float EPS = 1e-5f;
static const int NUM_LAYERS = 32, MM_LAYERS = 10;
static const int TXT_LEN = 8;
static const int ROPE_ROT = 126, ROPE_HALF = 63;

// ---------- fp16 解码 ----------
static inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t expo = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t f;
    if (expo == 0) {
        if (mant == 0) f = sign << 31;
        else {
            int e = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; e--; }
            mant &= 0x3ff;
            f = (sign << 31) | (e << 23) | (mant << 13);
        }
    } else if (expo == 0x1f) {
        f = (sign << 31) | 0x7f800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((expo - 15 + 127) << 23) | (mant << 13);
    }
    return reinterpret_cast<float&>(f);
}

// ---------- 文件读取 ----------
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
static std::vector<float> load_raw_f(const char* path) {
    std::vector<int64_t> sh; return load_raw(path, sh);
}

// ---------- Linear (ncnn fp16 解码) ----------
struct LinearRaw {
    int in = 0, out = 0;
    std::vector<float> W;   // (out, in) row-major
    std::vector<float> b;   // (out,) 或空
    void run(const float* x, float* y, int Ln) const {
        #pragma omp parallel for
        for (int t = 0; t < Ln; t++) {
            const float* xi = x + t*in;
            float* yo = y + t*out;
            for (int o = 0; o < out; o++) {
                const float* wi = &W[o*in];
                float acc = 0.f;
                for (int i = 0; i < in; i++) acc += xi[i]*wi[i];
                yo[o] = acc + (b.empty() ? 0.f : b[o]);
            }
        }
    }
    std::vector<float> batch(const std::vector<float>& x, int Ln) const {
        std::vector<float> y(Ln*out); run(x.data(), y.data(), Ln); return y;
    }
};

static LinearRaw load_linear(const std::string& base, bool has_bias) {
    // 从 .param 解析 num_output(0=) 与 total(2=)
    std::ifstream pf((base + ".param").c_str());
    std::string line; int numout = 0, total = 0;
    while (std::getline(pf, line)) {
        if (line.find("InnerProduct") != std::string::npos) {
            std::istringstream ss(line); std::string tok;
            while (ss >> tok) {
                if (tok.rfind("0=", 0) == 0) numout = std::stoi(tok.substr(2));
                if (tok.rfind("2=", 0) == 0) total = std::stoi(tok.substr(2));
            }
        }
    }
    int in = total / numout;
    LinearRaw L; L.in = in; L.out = numout;
    std::ifstream bf((base + ".bin").c_str(), std::ios::binary);
    uint32_t tag; bf.read((char*)&tag, 4);
    int n = numout*in;
    std::vector<uint16_t> hw(n); bf.read((char*)hw.data(), n*2);
    L.W.resize(n); for (int i = 0; i < n; i++) L.W[i] = fp16_to_float(hw[i]);
    if (has_bias) { L.b.resize(numout); bf.read((char*)L.b.data(), numout*4); }
    return L;
}

// ---------- 指标 ----------
static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return (float)(dot / (sqrt(na)*sqrt(nb) + 1e-12));
}
static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0; size_t n = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; i++) m = std::max(m, std::fabs(a[i]-b[i])); return m;
}

// ---------- 算子 ----------
static void rmsnorm(const float* x, int n, const float* w, float* out) {
    double s = 0; for (int i = 0; i < n; i++) s += (double)x[i]*x[i];
    float r = 1.0f / sqrtf((float)(s/n) + EPS);
    for (int i = 0; i < n; i++) out[i] = x[i]*r*(w ? w[i] : 1.f);
}
static void silu(std::vector<float>& x) {
    for (auto& v : x) v = v / (1.f + expf(-v));
}
// fusedrms qk_norm：权重 128（跨 head 共享）；in 为 (HEADS*HEAD_D)
static void qk_norm(const float* x, const float* w, float* out) {
    for (int h = 0; h < HEADS; h++) {
        const float* xp = x + h*HEAD_D; float* op = out + h*HEAD_D;
        float sq = 0; for (int d = 0; d < HEAD_D; d++) sq += xp[d]*xp[d];
        float r = sqrtf(sq / HEAD_D + EPS);
        for (int d = 0; d < HEAD_D; d++) op[d] = xp[d]/r * w[d];
    }
}
static void apply_rope(const float* freq, const float* q, float* out) {
    for (int k = 0; k < ROPE_ROT; k++) {
        float c = cosf(freq[k]), s = sinf(freq[k]);
        float rh = (k % 2 == 0) ? -q[k + 1] : q[k - 1];
        out[k] = q[k]*c + rh*s;
    }
    for (int k = ROPE_ROT; k < HEAD_D; k++) out[k] = q[k];
}

// ---------- 窗口几何 ----------
struct Win {
    int t, h, w, nwin;
    std::vector<int> st, en, sh, eh, sw, ew;
    std::vector<float> vid_freq, txt_freq;
};
static Win load_win(const char* path) {
    std::ifstream f(path, std::ios::binary);
    int64_t v[10]; f.read((char*)v, 80);
    Win w; w.t = (int)v[0]; w.h = (int)v[1]; w.w = (int)v[2]; w.nwin = (int)v[9];
    int ntot = w.nwin*6; std::vector<int64_t> sl(ntot); f.read((char*)sl.data(), ntot*8);
    for (int i = 0; i < w.nwin; i++) {
        w.st.push_back((int)sl[i*6]); w.en.push_back((int)sl[i*6+1]);
        w.sh.push_back((int)sl[i*6+2]); w.eh.push_back((int)sl[i*6+3]);
        w.sw.push_back((int)sl[i*6+4]); w.ew.push_back((int)sl[i*6+5]);
    }
    f.seekg(0, std::ios::end); int64_t fsz = f.tellg(); f.seekg(80 + ntot*8, std::ios::beg);
    int64_t rem = fsz - (80 + ntot*8); int nf = (int)(rem/4);
    std::vector<float> allf(nf); f.read((char*)allf.data(), nf*4);
    int vidn = nf - w.nwin*TXT_LEN*ROPE_ROT;
    w.vid_freq.assign(allf.begin(), allf.begin()+vidn);
    w.txt_freq.assign(allf.begin()+vidn, allf.end());
    return w;
}

// ---------- 分支权重 ----------
struct BranchW {
    LinearRaw qkv, out, mlp_in, mlp_g, mlp_out;
    std::vector<float> nq, nk, attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate;
};
static void load_branch(BranchW& W, const std::string& base) {
    W.qkv = load_linear(base + "_qkv", false);
    W.out = load_linear(base + "_out", true);
    W.mlp_in = load_linear(base + "_mlp_in", false);
    W.mlp_g = load_linear(base + "_mlp_g", false);
    W.mlp_out = load_linear(base + "_mlp_out", false);
    W.nq = load_raw_f((base + "_nq.bin").c_str());
    W.nk = load_raw_f((base + "_nk.bin").c_str());
    W.attn_shift = load_raw_f((base + "_attn_shift.bin").c_str());
    W.attn_scale = load_raw_f((base + "_attn_scale.bin").c_str());
    W.attn_gate  = load_raw_f((base + "_attn_gate.bin").c_str());
    W.mlp_shift  = load_raw_f((base + "_mlp_shift.bin").c_str());
    W.mlp_scale  = load_raw_f((base + "_mlp_scale.bin").c_str());
    W.mlp_gate   = load_raw_f((base + "_mlp_gate.bin").c_str());
}

// ---------- AWA 单 block 前向 ----------
static void awa_forward(const std::vector<float>& vid, const std::vector<float>& txt,
                        const Win& win, const BranchW& vW, const BranchW& tW,
                        std::vector<float>& vid_out, std::vector<float>& txt_out) {
    int Lv = (int)vid.size()/DIM, TXT = (int)txt.size()/DIM;
    int T = win.t, H = win.h, Wd = win.w;
    // proj_qkv + qk_norm（分支各自，batched）
    std::vector<float> vqkv = vW.qkv.batch(vid, Lv);
    std::vector<float> tqkv = tW.qkv.batch(txt, TXT);
    std::vector<float> vq(Lv*DIM), vk(Lv*DIM), vv(Lv*DIM);
    std::vector<float> tq(TXT*DIM), tk(TXT*DIM), tv(TXT*DIM);
    std::vector<float> qtmp(DIM), ktmp(DIM), vtmp(DIM);
    for (int i = 0; i < Lv; i++) {
        const float* q = &vqkv[i*QKV];
        qk_norm(q, vW.nq.data(), &vq[i*DIM]);
        qk_norm(q + 2560, vW.nk.data(), &vk[i*DIM]);
        memcpy(&vv[i*DIM], q + 5120, DIM*4);
    }
    for (int i = 0; i < TXT; i++) {
        const float* q = &tqkv[i*QKV];
        qk_norm(q, tW.nq.data(), &tq[i*DIM]);
        qk_norm(q + 2560, tW.nk.data(), &tk[i*DIM]);
        memcpy(&tv[i*DIM], q + 5120, DIM*4);
    }
    // 累计 f
    std::vector<int> cumf(win.nwin + 1, 0);
    for (int wi = 0; wi < win.nwin; wi++) {
        int f_i = (win.en[wi]-win.st[wi])*(win.eh[wi]-win.sh[wi])*(win.ew[wi]-win.sw[wi]);
        cumf[wi+1] = cumf[wi] + f_i;
    }
    int sumf = cumf[win.nwin];
    // partition (video)
    std::vector<float> vqw(sumf*DIM), vkw(sumf*DIM), vvw(sumf*DIM);
    for (int wi = 0; wi < win.nwin; wi++) {
        int local = 0;
        for (int lt = win.st[wi]; lt < win.en[wi]; lt++)
            for (int lh = win.sh[wi]; lh < win.eh[wi]; lh++)
                for (int lw = win.sw[wi]; lw < win.ew[wi]; lw++) {
                    int idx = ((lt*H) + lh)*Wd + lw;
                    int dst = cumf[wi] + local;
                    memcpy(&vqw[dst*DIM], &vq[idx*DIM], DIM*4);
                    memcpy(&vkw[dst*DIM], &vk[idx*DIM], DIM*4);
                    memcpy(&vvw[dst*DIM], &vv[idx*DIM], DIM*4);
                    local++;
                }
    }
    // RoPE (video)
    for (int dst = 0; dst < sumf; dst++) {
        const float* fr = &win.vid_freq[dst*ROPE_ROT];
        for (int hh = 0; hh < HEADS; hh++) {
            float inb[HEAD_D], outb[HEAD_D];
            memcpy(inb, &vqw[dst*DIM + hh*HEAD_D], HEAD_D*4);
            apply_rope(fr, inb, outb); memcpy(&vqw[dst*DIM + hh*HEAD_D], outb, HEAD_D*4);
            memcpy(inb, &vkw[dst*DIM + hh*HEAD_D], HEAD_D*4);
            apply_rope(fr, inb, outb); memcpy(&vkw[dst*DIM + hh*HEAD_D], outb, HEAD_D*4);
        }
    }
    // 文本逐窗口重复 + RoPE (q,k)
    std::vector<float> tqw(win.nwin*TXT*DIM), tkw(win.nwin*TXT*DIM), tvw(win.nwin*TXT*DIM);
    for (int wi = 0; wi < win.nwin; wi++) {
        for (int i = 0; i < TXT; i++) {
            int dst = wi*TXT + i;
            memcpy(&tqw[dst*DIM], &tq[i*DIM], DIM*4);
            memcpy(&tkw[dst*DIM], &tk[i*DIM], DIM*4);
            memcpy(&tvw[dst*DIM], &tv[i*DIM], DIM*4);
            const float* fr = &win.txt_freq[dst*ROPE_ROT];
            for (int hh = 0; hh < HEADS; hh++) {
                float inb[HEAD_D], outb[HEAD_D];
                memcpy(inb, &tqw[dst*DIM + hh*HEAD_D], HEAD_D*4);
                apply_rope(fr, inb, outb); memcpy(&tqw[dst*DIM + hh*HEAD_D], outb, HEAD_D*4);
                memcpy(inb, &tkw[dst*DIM + hh*HEAD_D], HEAD_D*4);
                apply_rope(fr, inb, outb); memcpy(&tkw[dst*DIM + hh*HEAD_D], outb, HEAD_D*4);
            }
        }
    }
    // 窗口内 SDPA
    std::vector<float> vout_win(sumf*DIM), tout_win(win.nwin*TXT*DIM);
    float scale = 1.0f/sqrtf((float)HEAD_D);
    for (int wi = 0; wi < win.nwin; wi++) {
        int f_i = cumf[wi+1] - cumf[wi];
        int S = f_i + TXT;
        for (int hh = 0; hh < HEADS; hh++) {
            std::vector<float> qseg(S*HEAD_D), kseg(S*HEAD_D), vseg(S*HEAD_D);
            for (int j = 0; j < f_i; j++) {
                memcpy(&qseg[j*HEAD_D], &vqw[(cumf[wi]+j)*DIM + hh*HEAD_D], HEAD_D*4);
                memcpy(&kseg[j*HEAD_D], &vkw[(cumf[wi]+j)*DIM + hh*HEAD_D], HEAD_D*4);
                memcpy(&vseg[j*HEAD_D], &vvw[(cumf[wi]+j)*DIM + hh*HEAD_D], HEAD_D*4);
            }
            for (int i = 0; i < TXT; i++) {
                int s = f_i + i;
                memcpy(&qseg[s*HEAD_D], &tqw[(wi*TXT+i)*DIM + hh*HEAD_D], HEAD_D*4);
                memcpy(&kseg[s*HEAD_D], &tkw[(wi*TXT+i)*DIM + hh*HEAD_D], HEAD_D*4);
                memcpy(&vseg[s*HEAD_D], &tvw[(wi*TXT+i)*DIM + hh*HEAD_D], HEAD_D*4);
            }
            std::vector<float> sc(S*S);
            float mx = -1e30f;
            for (int a = 0; a < S; a++) {
                const float* qa = &qseg[a*HEAD_D];
                for (int b = 0; b < S; b++) {
                    const float* kb = &kseg[b*HEAD_D];
                    float dot = 0; for (int d = 0; d < HEAD_D; d++) dot += qa[d]*kb[d];
                    dot *= scale; sc[a*S+b] = dot; if (dot > mx) mx = dot;
                }
            }
            for (int a = 0; a < S; a++) {
                float sm = 0; std::vector<float> p(S);
                for (int b = 0; b < S; b++) { float e = expf(sc[a*S+b]-mx); p[b]=e; sm+=e; }
                for (int b = 0; b < S; b++) p[b] /= sm;
                float o[HEAD_D] = {0};
                for (int b = 0; b < S; b++) {
                    const float* vb = &vseg[b*HEAD_D];
                    for (int d = 0; d < HEAD_D; d++) o[d] += p[b]*vb[d];
                }
                if (a < f_i) memcpy(&vout_win[(cumf[wi]+a)*DIM + hh*HEAD_D], o, HEAD_D*4);
                else         memcpy(&tout_win[(wi*TXT+(a-f_i))*DIM + hh*HEAD_D], o, HEAD_D*4);
            }
        }
    }
    // 文本平均（跨窗口）
    std::vector<float> tout(TXT*DIM, 0.f);
    for (int wi = 0; wi < win.nwin; wi++)
        for (int i = 0; i < TXT; i++)
            for (int d = 0; d < DIM; d++)
                tout[i*DIM + d] += tout_win[(wi*TXT+i)*DIM + d];
    for (size_t i = 0; i < tout.size(); i++) tout[i] /= (float)win.nwin;
    // window reverse (video)
    std::vector<float> vout_rev(Lv*DIM, 0.f);
    for (int wi = 0; wi < win.nwin; wi++) {
        int local = 0;
        for (int lt = win.st[wi]; lt < win.en[wi]; lt++)
            for (int lh = win.sh[wi]; lh < win.eh[wi]; lh++)
                for (int lw = win.sw[wi]; lw < win.ew[wi]; lw++) {
                    int idx = ((lt*H) + lh)*Wd + lw;
                    memcpy(&vout_rev[idx*DIM], &vout_win[(cumf[wi]+local)*DIM], DIM*4);
                    local++;
                }
    }
    // proj_out（分支各自，带 bias）
    vid_out = vW.out.batch(vout_rev, Lv);
    txt_out = tW.out.batch(tout, TXT);
}

// ---------- sinusoidal + TimeEmbedding ----------
static std::vector<float> sinusoidal(float t, int dim) {
    std::vector<float> e(dim);
    int half = dim/2;
    for (int j = 0; j < dim; j++) {
        float freq = (j < half) ? (float)exp(-logf(10000.f)*j/half) : (float)exp(-logf(10000.f)*(j-half)/half);
        e[j] = (j < half) ? sinf(t*freq) : cosf(t*freq);
    }
    return e;
}
static LinearRaw g_emb_in, g_emb_hid, g_emb_out;
static std::vector<float> time_embedding(float t) {
    std::vector<float> e = sinusoidal(t, 256);
    e = g_emb_in.batch(e, 1); silu(e);
    e = g_emb_hid.batch(e, 1); silu(e);
    e = g_emb_out.batch(e, 1);
    return e;
}

// ---------- 主流程 ----------
int main() {
    const char* M5 = "models/m5/";
    // 顶层权重
    LinearRaw vid_in_proj = load_linear(std::string(M5)+"vid_in_proj", true);
    LinearRaw txt_in      = load_linear(std::string(M5)+"txt_in", true);
    LinearRaw vid_out_proj= load_linear(std::string(M5)+"vid_out_proj", true);
    g_emb_in  = load_linear(std::string(M5)+"emb_proj_in", true);
    g_emb_hid = load_linear(std::string(M5)+"emb_proj_hid", true);
    g_emb_out = load_linear(std::string(M5)+"emb_proj_out", true);
    std::vector<float> Wvon = load_raw_f((std::string(M5)+"vid_out_norm_w.bin").c_str());
    std::vector<float> Wvoa_s = load_raw_f((std::string(M5)+"vid_out_ada_shift.bin").c_str());
    std::vector<float> Wvoa_sc = load_raw_f((std::string(M5)+"vid_out_ada_scale.bin").c_str());

    // 输入
    std::vector<int64_t> sh;
    auto x_patch = load_raw((std::string(M5)+"x_patch.bin").c_str(), sh);
    auto x_txt   = load_raw((std::string(M5)+"x_txt.bin").c_str(), sh);
    // 注意：x_patch 是 patchify 后的 RAW 输入（通道=33*1*2*2=132），x_txt 通道=5120；
    // 须按输入维度求 token 数，不能用 DIM(2560)。（proj 之后才是 2560）
    const int VID_IN_CH = 33*1*2*2;   // 132
    const int TXT_IN_CH = 5120;
    int Lv = (int)x_patch.size()/VID_IN_CH, TXT = (int)x_txt.size()/TXT_IN_CH;

    // 加载 32 个 block 权重
    std::vector<BranchW> bvid(NUM_LAYERS), btxt(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (i < MM_LAYERS) {
            load_branch(bvid[i], std::string(M5)+"b"+std::to_string(i)+"_vid");
            load_branch(btxt[i], std::string(M5)+"b"+std::to_string(i)+"_txt");
        } else {
            load_branch(bvid[i], std::string(M5)+"b"+std::to_string(i)+"_all");
            btxt[i] = bvid[i];
        }
    }
    Win win_ns = load_win((std::string(M5)+"win_nonshifted.bin").c_str());
    Win win_sh = load_win((std::string(M5)+"win_shifted.bin").c_str());
    fprintf(stderr, "[info] Lv=%d TXT=%d win_ns.nwin=%d win_sh.nwin=%d\n", Lv, TXT, win_ns.nwin, win_sh.nwin);

    // patch in
    std::vector<float> vid = vid_in_proj.batch(x_patch, Lv);
    std::vector<float> txt = txt_in.batch(x_txt, TXT);
    std::vector<float> emb = time_embedding(500.0f);

    int rc = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        const Win& win = (i % 2 == 0) ? win_ns : win_sh;
        const BranchW& vW = bvid[i]; const BranchW& tW = btxt[i];
        // attn_norm
        std::vector<float> vid_an(Lv*DIM), txt_an(TXT*DIM);
        for (int t = 0; t < Lv; t++) rmsnorm(&vid[t*DIM], DIM, nullptr, &vid_an[t*DIM]);
        for (int t = 0; t < TXT; t++) rmsnorm(&txt[t*DIM], DIM, nullptr, &txt_an[t*DIM]);
        // ada attn in
        for (int t = 0; t < Lv; t++)
            for (int d = 0; d < DIM; d++) {
                float sA=emb[d*6], scA=emb[d*6+1], gA=emb[d*6+2];
                float shB=vW.attn_shift[d], scB=vW.attn_scale[d], gB=vW.attn_gate[d];
                vid_an[t*DIM+d] = vid_an[t*DIM+d]*(scA+scB)+(sA+shB);
            }
        for (int t = 0; t < TXT; t++)
            for (int d = 0; d < DIM; d++) {
                float sA=emb[d*6], scA=emb[d*6+1], gA=emb[d*6+2];
                float thB=tW.attn_shift[d], tcB=tW.attn_scale[d], tgB=tW.attn_gate[d];
                txt_an[t*DIM+d] = txt_an[t*DIM+d]*(scA+tcB)+(sA+thB);
            }
        // AWA
        std::vector<float> vid_at, txt_at;
        awa_forward(vid_an, txt_an, win, vW, tW, vid_at, txt_at);
        // ada attn out
        for (int t = 0; t < Lv; t++)
            for (int d = 0; d < DIM; d++) { float gA=emb[d*6+2], gB=vW.attn_gate[d]; vid_at[t*DIM+d]*=(gA+gB); }
        for (int t = 0; t < TXT; t++)
            for (int d = 0; d < DIM; d++) { float gA=emb[d*6+2], tgB=tW.attn_gate[d]; txt_at[t*DIM+d]*=(gA+tgB); }
        for (int k = 0; k < Lv*DIM; k++) vid[k] = vid_at[k] + vid[k];
        for (int k = 0; k < TXT*DIM; k++) txt[k] = txt_at[k] + txt[k];
        // mlp_norm
        std::vector<float> vid_mn(Lv*DIM), txt_mn(TXT*DIM);
        for (int t = 0; t < Lv; t++) rmsnorm(&vid[t*DIM], DIM, nullptr, &vid_mn[t*DIM]);
        for (int t = 0; t < TXT; t++) rmsnorm(&txt[t*DIM], DIM, nullptr, &txt_mn[t*DIM]);
        // ada mlp in
        for (int t = 0; t < Lv; t++)
            for (int d = 0; d < DIM; d++) {
                float sAm=emb[d*6+3], scAm=emb[d*6+4], gAm=emb[d*6+5];
                float shB=vW.mlp_shift[d], scB=vW.mlp_scale[d], gB=vW.mlp_gate[d];
                vid_mn[t*DIM+d] = vid_mn[t*DIM+d]*(scAm+scB)+(sAm+shB);
            }
        for (int t = 0; t < TXT; t++)
            for (int d = 0; d < DIM; d++) {
                float sAm=emb[d*6+3], scAm=emb[d*6+4], gAm=emb[d*6+5];
                float thB=tW.mlp_shift[d], tcB=tW.mlp_scale[d], tgB=tW.mlp_gate[d];
                txt_mn[t*DIM+d] = txt_mn[t*DIM+d]*(scAm+tcB)+(sAm+thB);
            }
        // SwiGLU：官方 proj_out( silu(proj_in_gate) * proj_in )
        std::vector<float> vid_m = vW.mlp_out.batch(
            [&]{ std::vector<float> h = vW.mlp_in.batch(vid_mn, Lv);
                 std::vector<float> g = vW.mlp_g.batch(vid_mn, Lv);
                 for (size_t k = 0; k < h.size(); k++) g[k] = (g[k]/(1.f+expf(-g[k])))*h[k];
                 return g; }(), Lv);
        std::vector<float> txt_m = tW.mlp_out.batch(
            [&]{ std::vector<float> h = tW.mlp_in.batch(txt_mn, TXT);
                 std::vector<float> g = tW.mlp_g.batch(txt_mn, TXT);
                 for (size_t k = 0; k < h.size(); k++) g[k] = (g[k]/(1.f+expf(-g[k])))*h[k];
                 return g; }(), TXT);
        // ada mlp out
        for (int t = 0; t < Lv; t++)
            for (int d = 0; d < DIM; d++) { float gAm=emb[d*6+5], gB=vW.mlp_gate[d]; vid_m[t*DIM+d]*=(gAm+gB); }
        for (int t = 0; t < TXT; t++)
            for (int d = 0; d < DIM; d++) { float gAm=emb[d*6+5], gB=tW.mlp_gate[d]; txt_m[t*DIM+d]*=(gAm+gB); }
        for (int k = 0; k < Lv*DIM; k++) vid[k] = vid_m[k] + vid[k];
        for (int k = 0; k < TXT*DIM; k++) txt[k] = txt_m[k] + txt[k];

        // 逐 block 对拍（debug）
        if (i < 4 || i == 15 || i == 31) {
            auto refb = load_raw_f((std::string(M5)+"vid_b"+std::to_string(i)+".bin").c_str());
            float c = cosine(vid, refb), dd = max_abs_diff(vid, refb);
            fprintf(stderr, "[block %d] vid cos=%.6f max|diff|=%.5f %s\n", i, c, dd, (c>=0.995f?"OK":"<<<FAIL"));
            if (c < 0.99f) rc |= 2;
        }
    }

    // vid_out
    std::vector<float> vn(Lv*DIM);
    for (int t = 0; t < Lv; t++) rmsnorm(&vid[t*DIM], DIM, Wvon.data(), &vn[t*DIM]);
    for (int t = 0; t < Lv; t++)
        for (int d = 0; d < DIM; d++) {
            // [FIX 2026-09-10] emb 6 槽布局：out 层(attn 组) shift=emb[6d+0], scale=emb[6d+1]
            float sAo = emb[d*6+0], scAo = emb[d*6+1];
            vn[t*DIM+d] = vn[t*DIM+d]*(scAo + Wvoa_sc[d]) + (sAo + Wvoa_s[d]);
        }
    std::vector<float> vid_out = vid_out_proj.batch(vn, Lv);

    auto ref = load_raw_f((std::string(M5)+"ref_vid_out.bin").c_str());
    float c = cosine(vid_out, ref), dd = max_abs_diff(vid_out, ref);
    fprintf(stderr, "[FINAL vid_out] cos=%.6f max|diff|=%.5f\n", c, dd);
    if (c >= 0.99f && dd <= 0.5f) fprintf(stderr, "\n[PASS] M5 全网络验证通过\n");
    else { fprintf(stderr, "\n[FAIL] M5 全网络 cos 不足\n"); rc |= 1; }
    return rc;
}

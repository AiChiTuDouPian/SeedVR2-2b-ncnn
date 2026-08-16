// dit_vk.cpp — SeedVR2 NaDiT 的 Vulkan 加速推理引擎实现
#include "dit_vk.h"
#include <ncnn/platform.h>
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <memory>
#include <map>
#include <chrono>

// ---------- 常量 ----------
static const int HEADS = 20, HEAD_D = 128, DIM = 2560, QKV = HEADS*HEAD_D*3;
static const float EPS = 1e-5f;
static const int NUM_LAYERS = 32, MM_LAYERS = 10;
static const int MLP_DIM = 6912;                      // SwiGLU 中间维度（proj_in/proj_in_gate 各 2560->6912）
static const int ROPE_ROT = 126, ROPE_HALF = 63;      // 每头实际旋转 3 轴×42=126 通道（修复：原 66 为半分错误布局）

// ---------- 性能计时（诊断用，全局累计，单线程） ----------
static double g_lin_time = 0.0;   // 所有 lin()（ncnn GEMM）累计耗时
static double g_awa_time = 0.0;   // 所有 awa.forward()（GPU shader）累计耗时

// ---------- 文件读取 ----------
static std::string read_text(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[FAIL] 读 %s\n", path.c_str()); exit(1); }
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}
static std::vector<unsigned char> read_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[FAIL] 读 %s\n", path.c_str()); exit(1); }
    f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg(); f.seekg(0);
    std::vector<unsigned char> b(n); f.read((char*)b.data(), n); return b;
}
static std::vector<float> read_raw_f(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[FAIL] 读 %s\n", path.c_str()); exit(1); }
    int64_t ndim; f.read((char*)&ndim, 8);
    std::vector<int64_t> sh(ndim); for (auto& d : sh) f.read((char*)&d, 8);
    int64_t total = 1; for (auto d : sh) total *= d;
    std::vector<float> d(total); f.read((char*)d.data(), total*4); return d;
}

// ---------- CPU 算子（与 M5 验证一致） ----------
static void rmsnorm(const float* x, int n, const float* w, float* out) {
    double s = 0; for (int i = 0; i < n; i++) s += (double)x[i]*x[i];
    float r = 1.0f / sqrtf((float)(s/n) + EPS);
    for (int i = 0; i < n; i++) out[i] = x[i]*r*(w ? w[i] : 1.f);
}
static void silu(std::vector<float>& x) { for (auto& v : x) v = v / (1.f + expf(-v)); }
static void qk_norm(const float* x, const float* w, float* out) {
    for (int h = 0; h < HEADS; h++) {
        const float* xp = x + h*HEAD_D; float* op = out + h*HEAD_D;
        float sq = 0; for (int d = 0; d < HEAD_D; d++) sq += xp[d]*xp[d];
        float r = sqrtf(sq / HEAD_D + EPS);
        for (int d = 0; d < HEAD_D; d++) op[d] = xp[d]/r * w[d];
    }
}
static void apply_rope(const float* freq, const float* q, float* out) {
    // rotary_embedding_torch 的 freq 布局为 [f0,f0,f1,f1,...]，故 rotate_half 必须按
    // 「相邻配对」(2i,2i+1) 交织：out[2i]=-x[2i+1], out[2i+1]=x[2i]。
    for (int k = 0; k < ROPE_ROT; k++) {
        float c = cosf(freq[k]), s = sinf(freq[k]);
        float rh = (k % 2 == 0) ? -q[k + 1] : q[k - 1];
        out[k] = q[k]*c + rh*s;
    }
    for (int k = ROPE_ROT; k < HEAD_D; k++) out[k] = q[k];
}

// ---------- 窗口加载 ----------
Win load_win(const char* path, int txt_len) {
    std::ifstream f(path, std::ios::binary);
    int64_t v[10]; f.read((char*)v, 80);
    Win w; w.t = (int)v[0]; w.h = (int)v[1]; w.w = (int)v[2]; w.nwin = (int)v[9];
    w.txt_len = txt_len;
    int ntot = w.nwin*6; std::vector<int64_t> sl(ntot); f.read((char*)sl.data(), ntot*8);
    for (int i = 0; i < w.nwin; i++) {
        w.st.push_back((int)sl[i*6]); w.en.push_back((int)sl[i*6+1]);
        w.sh.push_back((int)sl[i*6+2]); w.eh.push_back((int)sl[i*6+3]);
        w.sw.push_back((int)sl[i*6+4]); w.ew.push_back((int)sl[i*6+5]);
    }
    f.seekg(0, std::ios::end); int64_t fsz = f.tellg(); f.seekg(80 + ntot*8, std::ios::beg);
    int64_t rem = fsz - (80 + ntot*8); int nf = (int)(rem/4);
    std::vector<float> allf(nf); f.read((char*)allf.data(), nf*4);
    // 真实模型 txt_len=58：vid_freq=sum(f_i)*66，txt_freq=nwin*txt_len*66
    int vidn = nf - w.nwin*txt_len*ROPE_ROT;
    w.vid_freq.assign(allf.begin(), allf.begin()+vidn);
    w.txt_freq.assign(allf.begin()+vidn, allf.end());
    return w;
}

// ---------- DitVk ----------
DitVk::DitVk() {}
DitVk::~DitVk() {
    // 必须先释放所有引用 GPU 设备的资源，再销毁 GPU 实例，否则 use-after-free -> SIGSEGV。
    // 顺序很关键：net_cache（含权重分配器/pipeline）与 awa pipeline 都要在 destroy_gpu_instance 之前析构。
    net_cache.clear();                 // 析构所有缓存的 ncnn::Net -> 释放权重 VkWeightAllocator + pipeline
    awa.release();                     // 删除 AWA compute pipeline（引用 device）
    blob_alloc = nullptr;
    staging_alloc = nullptr;
    if (vkdev) ncnn::destroy_gpu_instance();
    vkdev = nullptr;
}

void DitVk::reclaim_vram() {
    // 每层所有 Net 已析构、AwaVk 的 VkMat 已出作用域，此时 blob/staging allocator 的 free-list
    // 只含已释放块，clear() 把整块已提交 Vulkan 显存真正还给驱动，避免 32 层累计资源耗尽。
    // （权重 VkWeightAllocator 各 Net 私有，随 Net 析构自动 clear，这里无需处理。）
    if (blob_alloc)    blob_alloc->clear();
    if (staging_alloc) staging_alloc->clear();
}

void DitVk::reset_vulkan_device() {
    // 块间重置：先释放所有引用旧设备的资源，再销毁/重建 GPU 实例（全新 VkDevice），
    // 最后重新 acquire allocator 并重建 AWA pipeline。CPU 侧权重（Wvon/Wvoa_*/b*_raw）
    // 与 fcache 字节保留在内存，不重读磁盘，下一次 get_net 从 fcache 重新上传权重到新设备。
    fprintf(stderr, "[reset] 释放旧 net_cache / AWA pipeline...\n"); fflush(stderr);
    net_cache.clear();                  // 析构旧设备上的 ncnn::Net（权重分配器/pipeline）
    awa.release();                      // 删除 AWA compute pipeline
    blob_alloc = nullptr;
    staging_alloc = nullptr;
    ncnn::destroy_gpu_instance();       // 销毁 VkDevice，清空设备级命令池/描述符池/pipeline
    if (ncnn::create_gpu_instance() != 0) { fprintf(stderr, "[FAIL] create_gpu_instance (reset)\n"); }
    vkdev = ncnn::get_gpu_device(0);
    blob_alloc    = vkdev->acquire_blob_allocator();
    staging_alloc = vkdev->acquire_staging_allocator();
    std::string spv = dir + "awa.spv"; { std::ifstream t(spv); if (!t) spv = "awa.spv"; }
    awa.init(vkdev, spv);
    awa.set_allocators(blob_alloc, staging_alloc);
    fprintf(stderr, "[reset] 新 Vulkan 设备就绪 (awa.ready=%d)\n", (int)awa.ready()); fflush(stderr);
}

bool DitVk::init(const std::string& model_dir, bool fp16_arith, int precision) {
    dir = model_dir;
    if (dir.empty() || dir.back() != '/') dir += '/';
    this->fp16_arith = fp16_arith;
    this->precision_ = precision;

    // Vulkan 设备
    if (ncnn::create_gpu_instance() != 0) { fprintf(stderr, "[FAIL] create_gpu_instance\n"); return false; }
    vkdev = ncnn::get_gpu_device(0);
    if (!vkdev) { fprintf(stderr, "[FAIL] get_gpu_device\n"); return false; }

    // CPU 侧权重
    Wvon    = read_raw_f(dir + "vid_out_norm_w.bin");
    Wvoa_s  = read_raw_f(dir + "vid_out_ada_shift.bin");
    Wvoa_sc = read_raw_f(dir + "vid_out_ada_scale.bin");

    bvid_raw.resize(NUM_LAYERS); btxt_raw.resize(NUM_LAYERS);
    for (int i = 0; i < NUM_LAYERS; i++) {
        std::string bv = (i < MM_LAYERS) ? ("b"+std::to_string(i)+"_vid") : ("b"+std::to_string(i)+"_all");
        std::string bt = (i < MM_LAYERS) ? ("b"+std::to_string(i)+"_txt") : ("b"+std::to_string(i)+"_all");
        auto load_raw_branch = [&](const std::string& base, RawBranch& R) {
            R.nq  = read_raw_f(dir + base + "_nq.bin");
            R.nk  = read_raw_f(dir + base + "_nk.bin");
            R.attn_shift = read_raw_f(dir + base + "_attn_shift.bin");
            R.attn_scale = read_raw_f(dir + base + "_attn_scale.bin");
            R.attn_gate  = read_raw_f(dir + base + "_attn_gate.bin");
            R.mlp_shift  = read_raw_f(dir + base + "_mlp_shift.bin");
            R.mlp_scale  = read_raw_f(dir + base + "_mlp_scale.bin");
            R.mlp_gate   = read_raw_f(dir + base + "_mlp_gate.bin");
        };
        load_raw_branch(bv, bvid_raw[i]);
        load_raw_branch(bt, btxt_raw[i]);
    }

    // 若目录里有固定窗口（GRID=2,40,40）则默认加载
    {
        std::ifstream fns(dir + "win_nonshifted.bin", std::ios::binary);
        if (fns) { fns.close(); win_ns = load_win((dir + "win_nonshifted.bin").c_str()); }
        std::ifstream fsh(dir + "win_shifted.bin", std::ios::binary);
        if (fsh) { fsh.close(); win_sh = load_win((dir + "win_shifted.bin").c_str()); }
    }
    fprintf(stderr, "[DitVk] init ok (fp16_arith=%d) win_ns.nwin=%d win_sh.nwin=%d\n",
            fp16_arith, win_ns.nwin, win_sh.nwin);

    // 自定义 Vulkan AWA 模块：优先 dir/awa.spv，否则 cwd/awa.spv
    {
        // 全局只 acquire 一次 blob/staging allocator，所有 Net 与 AwaVk 复用，避免显存泄漏
        blob_alloc    = vkdev->acquire_blob_allocator();
        staging_alloc = vkdev->acquire_staging_allocator();
        std::string spv = dir + "awa.spv";
        std::ifstream t(spv);
        if (!t) spv = "awa.spv";
        awa.init(vkdev, spv);
        awa.set_allocators(blob_alloc, staging_alloc);
    }
    return true;
}

bool DitVk::cache_file(const std::string& base) {
    if (fcache.count(base)) return true;
    std::string p = read_text(dir + base + ".param");
    std::vector<unsigned char> m = read_bin(dir + base + ".bin");
    fcache[base] = {p, m};
    return true;
}
ncnn::Net* DitVk::get_net(const std::string& base) {
    auto it = net_cache.find(base);
    if (it != net_cache.end()) {
        // 更新 LRU：移到末尾（最近使用）
        net_lru.remove(base);
        net_lru.push_back(base);
        return it->second.get();
    }
    cache_file(base);
    auto& f = fcache[base];
    std::unique_ptr<ncnn::Net> net(new ncnn::Net);
    net->set_vulkan_device(vkdev);
    net->opt.use_vulkan_compute = true;
    net->opt.use_fp16_arithmetic = fp16_arith;
    // 存储精度（precision_：0=fp32 默认逐位对齐 / 1=fp16 / 2=bf16）。
    // fp16 在真实大激活 >65504 时会溢出 Inf/NaN（历史踩坑）；bf16 指数位同 fp32 范围安全，
    // 对齐官方 PyTorch bf16，且经 cooperative matrix(tensor core) 加速、累加器仍 fp32。
    net->opt.use_fp16_storage = (precision_ == 1);
    net->opt.use_fp16_packed  = (precision_ == 1);
    net->opt.use_bf16_storage = (precision_ == 2);
    net->opt.use_bf16_packed  = (precision_ == 2);
    // 复用 DitVk 缓存的 blob/staging allocator（set_vulkan_device 内部会 acquire 新实例，这里覆盖回去）
    if (blob_alloc)    net->opt.blob_vkallocator    = blob_alloc;
    if (staging_alloc) net->opt.staging_vkallocator = staging_alloc;
    net->load_param_mem(f.first.c_str());
    net->load_model(f.second.data());
    ncnn::Net* p = net.get();
    net_cache[base] = std::move(net);
    net_lru.remove(base);
    net_lru.push_back(base);
    evict_nets();
    return p;
}

void DitVk::evict_nets() {
    while (net_cache.size() > NET_CACHE_CAP) {
        std::string old = net_lru.front();
        net_lru.pop_front();
        auto it = net_cache.find(old);
        if (it != net_cache.end()) {
            // unique_ptr 析构 -> ncnn::Net 析构 -> 释放其私有权重 VkWeightAllocator（device-local 显存）
            net_cache.erase(it);
        }
    }
}

std::vector<float> DitVk::lin(const std::string& base, const float* x, int Ln, int in_dim, int out_dim) {
    auto _t0 = std::chrono::high_resolution_clock::now();
    ncnn::Net* net = get_net(base);
    ncnn::Mat in(in_dim, Ln, (void*)x);
    ncnn::Mat out;
    {
        ncnn::Extractor ex = net->create_extractor();
        ex.input("in0", in);
        ex.extract("out0", out);
    }
    std::vector<float> y(Ln*out_dim);
    const float* od = out;
    if (out.w == out_dim && out.h == Ln && out.c == 1) {
        for (int t = 0; t < Ln; t++) for (int o = 0; o < out_dim; o++) y[t*out_dim+o] = od[t*out_dim+o];
    } else if (out.w == Ln && out.h == out_dim && out.c == 1) {
        for (int t = 0; t < Ln; t++) for (int o = 0; o < out_dim; o++) y[t*out_dim+o] = od[o*Ln+t];
    } else {
        fprintf(stderr, "[FAIL] %s 输出维度异常 w=%d h=%d c=%d (期望 %d x %d)\n",
                base.c_str(), out.w, out.h, out.c, out_dim, Ln);
        exit(1);
    }
    g_lin_time += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - _t0).count();
    return y;
}

static std::vector<float> sinusoidal(float t, int dim) {
    std::vector<float> e(dim); int half = dim/2;
    for (int j = 0; j < dim; j++) {
        float freq = (j < half) ? (float)exp(-logf(10000.f)*j/half) : (float)exp(-logf(10000.f)*(j-half)/half);
        e[j] = (j < half) ? sinf(t*freq) : cosf(t*freq);
    }
    return e;
}
std::vector<float> DitVk::time_embedding(float t) {
    std::vector<float> e = sinusoidal(t, 256);
    e = lin("emb_proj_in",  e.data(), 1, 256,  2560); silu(e);
    e = lin("emb_proj_hid", e.data(), 1, 2560, 2560); silu(e);
    e = lin("emb_proj_out", e.data(), 1, 2560, 15360);
    return e;  // (1,15360)
}

// 单 block 窗口注意力（qkv / proj_out 走 Vulkan；其余 CPU）
void DitVk::awa_forward(int i, const std::vector<float>& vid, const std::vector<float>& txt, const Win& win,
                        const RawBranch& vR, const RawBranch& tR,
                        std::vector<float>& vid_out, std::vector<float>& txt_out) {
    bool dual = (i < MM_LAYERS);
    std::string bv = dual ? ("b"+std::to_string(i)+"_vid") : ("b"+std::to_string(i)+"_all");
    std::string bt = dual ? ("b"+std::to_string(i)+"_txt") : ("b"+std::to_string(i)+"_all");
    int Lv = (int)vid.size()/DIM, TXT = (int)txt.size()/DIM;
    int T = win.t, H = win.h, Wd = win.w;

    std::vector<float> vqkv = lin(bv+"_qkv", vid.data(), Lv, DIM, QKV);
    std::vector<float> tqkv = lin(bt+"_qkv", txt.data(), TXT, DIM, QKV);
    if (getenv("SEEDVR_DUMP_AWA")) {
        { std::ofstream fo("dbg_vqkv.bin", std::ios::binary); fo.write((const char*)vqkv.data(), vqkv.size()*4); }
        { std::ofstream fo("dbg_tqkv.bin", std::ios::binary); fo.write((const char*)tqkv.data(), tqkv.size()*4); }
    }

    bool force_cpu_awa = (getenv("SEEDVR_FORCE_CPU_AWA") != nullptr);
    if (awa.ready() && !force_cpu_awa) {
        // ★ 自定义 Vulkan AWA：partition+qk_norm+RoPE+varlen SDPA+unpartition+coalesce 全在 GPU shader
        std::vector<float> vattn, tattn;
        auto _t0 = std::chrono::high_resolution_clock::now();
        awa.forward(vqkv, tqkv, win, vR.nq, vR.nk, tR.nq, tR.nk, Lv, TXT, vattn, tattn);
        g_awa_time += std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - _t0).count();
        if (getenv("SEEDVR_DUMP_AWA")) {
            { std::ofstream fo("dbg_gpu_vattn.bin", std::ios::binary); fo.write((const char*)vattn.data(), vattn.size()*4); }
            { std::ofstream fo("dbg_gpu_tattn.bin", std::ios::binary); fo.write((const char*)tattn.data(), tattn.size()*4); }
            fprintf(stderr, "[dump] gpu vattn=%zu tattn=%zu\n", vattn.size(), tattn.size());
        }
        vid_out = lin(bv+"_out", vattn.data(), Lv, DIM, DIM);
        txt_out = lin(bt+"_out", tattn.data(), TXT, DIM, DIM);
        return;
    }

    std::vector<float> vq(Lv*DIM), vk(Lv*DIM), vv(Lv*DIM);
    std::vector<float> tq(TXT*DIM), tk(TXT*DIM), tv(TXT*DIM);
    std::vector<float> qtmp(DIM), ktmp(DIM), vtmp(DIM);
    for (int ii = 0; ii < Lv; ii++) {
        const float* q = &vqkv[ii*QKV];
        qk_norm(q, vR.nq.data(), &vq[ii*DIM]);
        qk_norm(q + 2560, vR.nk.data(), &vk[ii*DIM]);
        memcpy(&vv[ii*DIM], q + 5120, DIM*4);
    }
    for (int ii = 0; ii < TXT; ii++) {
        const float* q = &tqkv[ii*QKV];
        qk_norm(q, tR.nq.data(), &tq[ii*DIM]);
        qk_norm(q + 2560, tR.nk.data(), &tk[ii*DIM]);
        memcpy(&tv[ii*DIM], q + 5120, DIM*4);
    }
    std::vector<int> cumf(win.nwin + 1, 0);
    for (int wi = 0; wi < win.nwin; wi++) {
        int f_i = (win.en[wi]-win.st[wi])*(win.eh[wi]-win.sh[wi])*(win.ew[wi]-win.sw[wi]);
        cumf[wi+1] = cumf[wi] + f_i;
    }
    int sumf = cumf[win.nwin];
    std::vector<float> vqw(sumf*DIM), vkw(sumf*DIM), vvw(sumf*DIM);
    for (int wi = 0; wi < win.nwin; wi++) {
        int local = 0;
        for (int lt = win.st[wi]; lt < win.en[wi]; lt++)
            for (int lh = win.sh[wi]; lh < win.eh[wi]; lh++)
                for (int lw = win.sw[wi]; lw < win.ew[wi]; lw++) {
                    int idx = ((lt*H) + lh)*Wd + lw; int dst = cumf[wi] + local;
                    memcpy(&vqw[dst*DIM], &vq[idx*DIM], DIM*4);
                    memcpy(&vkw[dst*DIM], &vk[idx*DIM], DIM*4);
                    memcpy(&vvw[dst*DIM], &vv[idx*DIM], DIM*4);
                    local++;
                }
    }
    for (int dst = 0; dst < sumf; dst++) {
        const float* fr = &win.vid_freq[dst*ROPE_ROT];
        for (int hh = 0; hh < HEADS; hh++) {
            float inb[HEAD_D], outb[HEAD_D];
            memcpy(inb, &vqw[dst*DIM + hh*HEAD_D], HEAD_D*4); apply_rope(fr, inb, outb); memcpy(&vqw[dst*DIM + hh*HEAD_D], outb, HEAD_D*4);
            memcpy(inb, &vkw[dst*DIM + hh*HEAD_D], HEAD_D*4); apply_rope(fr, inb, outb); memcpy(&vkw[dst*DIM + hh*HEAD_D], outb, HEAD_D*4);
        }
    }
    std::vector<float> tqw(win.nwin*TXT*DIM), tkw(win.nwin*TXT*DIM), tvw(win.nwin*TXT*DIM);
    for (int wi = 0; wi < win.nwin; wi++) {
        for (int ii = 0; ii < TXT; ii++) {
            int dst = wi*TXT + ii;
            memcpy(&tqw[dst*DIM], &tq[ii*DIM], DIM*4);
            memcpy(&tkw[dst*DIM], &tk[ii*DIM], DIM*4);
            memcpy(&tvw[dst*DIM], &tv[ii*DIM], DIM*4);
            const float* fr = &win.txt_freq[dst*ROPE_ROT];
            for (int hh = 0; hh < HEADS; hh++) {
                float inb[HEAD_D], outb[HEAD_D];
                memcpy(inb, &tqw[dst*DIM + hh*HEAD_D], HEAD_D*4); apply_rope(fr, inb, outb); memcpy(&tqw[dst*DIM + hh*HEAD_D], outb, HEAD_D*4);
                memcpy(inb, &tkw[dst*DIM + hh*HEAD_D], HEAD_D*4); apply_rope(fr, inb, outb); memcpy(&tkw[dst*DIM + hh*HEAD_D], outb, HEAD_D*4);
            }
        }
    }
    std::vector<float> vout_win(sumf*DIM), tout_win(win.nwin*TXT*DIM);
    float scale = 1.0f/sqrtf((float)HEAD_D);
    std::vector<float> qseg, kseg, vseg;
    for (int wi = 0; wi < win.nwin; wi++) {
        int f_i = cumf[wi+1] - cumf[wi]; int S = f_i + TXT;
        qseg.resize(S*HEAD_D); kseg.resize(S*HEAD_D); vseg.resize(S*HEAD_D);
        for (int hh = 0; hh < HEADS; hh++) {
            for (int j = 0; j < f_i; j++) {
                memcpy(&qseg[j*HEAD_D], &vqw[(cumf[wi]+j)*DIM + hh*HEAD_D], HEAD_D*4);
                memcpy(&kseg[j*HEAD_D], &vkw[(cumf[wi]+j)*DIM + hh*HEAD_D], HEAD_D*4);
                memcpy(&vseg[j*HEAD_D], &vvw[(cumf[wi]+j)*DIM + hh*HEAD_D], HEAD_D*4);
            }
            for (int ii = 0; ii < TXT; ii++) {
                int s = f_i + ii;
                memcpy(&qseg[s*HEAD_D], &tqw[(wi*TXT+ii)*DIM + hh*HEAD_D], HEAD_D*4);
                memcpy(&kseg[s*HEAD_D], &tkw[(wi*TXT+ii)*DIM + hh*HEAD_D], HEAD_D*4);
                memcpy(&vseg[s*HEAD_D], &tvw[(wi*TXT+ii)*DIM + hh*HEAD_D], HEAD_D*4);
            }
            std::vector<float> sc(S*S); float mx = -1e30f;
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
                for (int b = 0; b < S; b++) { const float* vb = &vseg[b*HEAD_D]; for (int d = 0; d < HEAD_D; d++) o[d] += p[b]*vb[d]; }
                if (a < f_i) memcpy(&vout_win[(cumf[wi]+a)*DIM + hh*HEAD_D], o, HEAD_D*4);
                else         memcpy(&tout_win[(wi*TXT+(a-f_i))*DIM + hh*HEAD_D], o, HEAD_D*4);
            }
        }
    }
    std::vector<float> tout(TXT*DIM, 0.f);
    for (int wi = 0; wi < win.nwin; wi++)
        for (int ii = 0; ii < TXT; ii++)
            for (int d = 0; d < DIM; d++) tout[ii*DIM + d] += tout_win[(wi*TXT+ii)*DIM + d];
    for (size_t ii = 0; ii < tout.size(); ii++) tout[ii] /= (float)win.nwin;
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
    if (getenv("SEEDVR_DUMP_AWA")) {
        { std::ofstream fo("dbg_cpu_vout_rev.bin", std::ios::binary); fo.write((const char*)vout_rev.data(), vout_rev.size()*4); }
        { std::ofstream fo("dbg_cpu_tout.bin", std::ios::binary); fo.write((const char*)tout.data(), tout.size()*4); }
        fprintf(stderr, "[dump] cpu vout_rev=%zu tout=%zu\n", vout_rev.size(), tout.size());
    }
    vid_out = lin(bv+"_out", vout_rev.data(), Lv, DIM, DIM);
    txt_out = lin(bt+"_out", tout.data(),     TXT, DIM, DIM);
}

std::vector<float> DitVk::swiglu(const std::string& b, const float* x, int Ln) {
    // 官方 SwiGLUMLP: proj_out( silu(proj_in_gate(x)) * proj_in(x) )
    // 注意 silu 作用在 proj_in_gate（g）上，而非 proj_in（h）。
    std::vector<float> h = lin(b+"_mlp_in", x, Ln, DIM, MLP_DIM);   // proj_in
    std::vector<float> g = lin(b+"_mlp_g",  x, Ln, DIM, MLP_DIM);   // proj_in_gate
    for (size_t k = 0; k < h.size(); k++) g[k] = (g[k]/(1.f+expf(-g[k])))*h[k];  // silu(g)*h
    return lin(b+"_mlp_out", g.data(), Ln, MLP_DIM, DIM);
}

bool DitVk::forward_latent(const std::vector<float>& vid_in, int Lv,
                           const std::vector<float>& txt_in, int TXT, float timestep,
                           int l0, int l1, bool do_init, bool do_final,
                           std::vector<float>& vid_out_latent, std::vector<float>& txt_out_latent,
                           std::vector<float>& sr_out) {
    std::vector<float> vid, txt;
    auto _pf_t0 = std::chrono::high_resolution_clock::now();
    double _pf_lin0 = g_lin_time, _pf_awa0 = g_awa_time;
    if (do_init) {
        fprintf(stderr, "[fl] vid_in_proj / txt_in...\n"); fflush(stderr);
        vid = lin("vid_in_proj", vid_in.data(), Lv, 33*1*2*2, DIM);
        txt = lin("txt_in",      txt_in.data(), TXT, 5120, DIM);
    } else {
        vid = vid_in;   // 已是 (Lv, DIM) 残差 latent（上一块传出）
        txt = txt_in;   // 已是 (TXT, DIM) 残差 latent
    }
    std::vector<float> emb = time_embedding(timestep);

    int max_layers = NUM_LAYERS;
    if (const char* ml = getenv("SEEDVR_MAX_LAYERS")) { max_layers = atoi(ml); if (max_layers < 1) max_layers = NUM_LAYERS; }
    int hi = (l1 < max_layers) ? l1 : max_layers;

    for (int i = l0; i < hi; i++) {
        const Win& win = (i % 2 == 0) ? win_ns : win_sh;
        RawBranch& vR = bvid_raw[i]; RawBranch& tR = btxt_raw[i];
        bool dual = (i < MM_LAYERS);
        std::string bv = dual ? ("b"+std::to_string(i)+"_vid") : ("b"+std::to_string(i)+"_all");
        std::string bt = dual ? ("b"+std::to_string(i)+"_txt") : ("b"+std::to_string(i)+"_all");
        if ((i - l0) % 4 == 0) { fprintf(stderr, "[fl] layer %d/%d\n", i, NUM_LAYERS); fflush(stderr); }

        std::vector<float> vid_an(Lv*DIM), txt_an(TXT*DIM);
        for (int t = 0; t < Lv; t++) rmsnorm(&vid[t*DIM], DIM, nullptr, &vid_an[t*DIM]);
        for (int t = 0; t < TXT; t++) rmsnorm(&txt[t*DIM], DIM, nullptr, &txt_an[t*DIM]);
        for (int t = 0; t < Lv; t++)
            for (int d = 0; d < DIM; d++) {
                float sA=emb[d*6], scA=emb[d*6+1], gA=emb[d*6+2];
                float shB=vR.attn_shift[d], scB=vR.attn_scale[d], gB=vR.attn_gate[d];
                vid_an[t*DIM+d] = vid_an[t*DIM+d]*(scA+scB)+(sA+shB);
            }
        for (int t = 0; t < TXT; t++)
            for (int d = 0; d < DIM; d++) {
                float sA=emb[d*6], scA=emb[d*6+1], gA=emb[d*6+2];
                float thB=tR.attn_shift[d], tcB=tR.attn_scale[d], tgB=tR.attn_gate[d];
                txt_an[t*DIM+d] = txt_an[t*DIM+d]*(scA+tcB)+(sA+thB);
            }
        std::vector<float> vid_at, txt_at;
        awa_forward(i, vid_an, txt_an, win, vR, tR, vid_at, txt_at);
        for (int t = 0; t < Lv; t++)
            for (int d = 0; d < DIM; d++) { float gA=emb[d*6+2], gB=vR.attn_gate[d]; vid_at[t*DIM+d]*=(gA+gB); }
        for (int t = 0; t < TXT; t++)
            for (int d = 0; d < DIM; d++) { float gA=emb[d*6+2], tgB=tR.attn_gate[d]; txt_at[t*DIM+d]*=(gA+tgB); }
        for (int k = 0; k < Lv*DIM; k++) vid[k] = vid_at[k] + vid[k];
        for (int k = 0; k < TXT*DIM; k++) txt[k] = txt_at[k] + txt[k];

        std::vector<float> vid_mn(Lv*DIM), txt_mn(TXT*DIM);
        for (int t = 0; t < Lv; t++) rmsnorm(&vid[t*DIM], DIM, nullptr, &vid_mn[t*DIM]);
        for (int t = 0; t < TXT; t++) rmsnorm(&txt[t*DIM], DIM, nullptr, &txt_mn[t*DIM]);
        for (int t = 0; t < Lv; t++)
            for (int d = 0; d < DIM; d++) {
                float sAm=emb[d*6+3], scAm=emb[d*6+4], gAm=emb[d*6+5];
                float shB=vR.mlp_shift[d], scB=vR.mlp_scale[d], gB=vR.mlp_gate[d];
                vid_mn[t*DIM+d] = vid_mn[t*DIM+d]*(scAm+scB)+(sAm+shB);
            }
        for (int t = 0; t < TXT; t++)
            for (int d = 0; d < DIM; d++) {
                float sAm=emb[d*6+3], scAm=emb[d*6+4], gAm=emb[d*6+5];
                float thB=tR.mlp_shift[d], tcB=tR.mlp_scale[d], tgB=tR.mlp_gate[d];
                txt_mn[t*DIM+d] = txt_mn[t*DIM+d]*(scAm+tcB)+(sAm+thB);
            }
        std::vector<float> vid_m = swiglu(bv, vid_mn.data(), Lv);
        std::vector<float> txt_m = swiglu(bt, txt_mn.data(), TXT);
        for (int t = 0; t < Lv; t++)
            for (int d = 0; d < DIM; d++) { float gAm=emb[d*6+5], gB=vR.mlp_gate[d]; vid_m[t*DIM+d]*=(gAm+gB); }
        for (int t = 0; t < TXT; t++)
            for (int d = 0; d < DIM; d++) { float gAm=emb[d*6+5], gB=tR.mlp_gate[d]; txt_m[t*DIM+d]*=(gAm+gB); }
        for (int k = 0; k < Lv*DIM; k++) vid[k] = vid_m[k] + vid[k];
        for (int k = 0; k < TXT*DIM; k++) txt[k] = txt_m[k] + txt[k];

        reclaim_vram();   // 本层所有 Net 已析构、AwaVk VkMat 已出作用域，回收激活显存
    }

    if (do_final) {
        std::vector<float> vn(Lv*DIM);
        for (int t = 0; t < Lv; t++) rmsnorm(&vid[t*DIM], DIM, Wvon.data(), &vn[t*DIM]);
        for (int t = 0; t < Lv; t++)
            for (int d = 0; d < DIM; d++) {
                float sAo = emb[d*3], scAo = emb[d*3+1];
                vn[t*DIM+d] = vn[t*DIM+d]*(scAo + Wvoa_sc[d]) + (sAo + Wvoa_s[d]);
            }
        sr_out = lin("vid_out_proj", vn.data(), Lv, DIM, 64);
        vid_out_latent.clear(); txt_out_latent.clear();
        fprintf(stderr, "[fl] 块末 norm+vid_out_proj ok (layers [%d,%d))\n", l0, hi); fflush(stderr);
    } else {
        vid_out_latent = vid;
        txt_out_latent = txt;
    }
    {
        double _pf_total = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - _pf_t0).count();
        double _pf_lin = g_lin_time - _pf_lin0;
        double _pf_awa = g_awa_time - _pf_awa0;
        double _pf_cpu = _pf_total - _pf_lin - _pf_awa;
        if (_pf_total < 1e-9) _pf_total = 1e-9;
        fprintf(stderr, "[perf] DiT块[%d,%d): total=%.3fs  GEMM=%.3fs(%.0f%%)  AWA=%.3fs(%.0f%%)  CPU循环=%.3fs(%.0f%%)\n",
                l0, hi, _pf_total, _pf_lin, 100*_pf_lin/_pf_total,
                _pf_awa, 100*_pf_awa/_pf_total, _pf_cpu, 100*_pf_cpu/_pf_total);
    }
    return true;
}

bool DitVk::core_forward(std::vector<float>& vid, int Lv, std::vector<float>& txt, int TXT,
                         float timestep, std::vector<float>& out_sr) {
    // 单次全量前向：等价于 forward_latent(0, NUM_LAYERS, do_init=true, do_final=true)
    std::vector<float> vl, tl, sr;
    if (!forward_latent(vid, Lv, txt, TXT, timestep, 0, NUM_LAYERS, true, true, vl, tl, sr)) return false;
    out_sr = sr;
    return true;
}

bool DitVk::forward(const std::vector<float>& vid_patch, int Lv,
                    const std::vector<float>& txt, int TXT, float timestep,
                    std::vector<float>& out_sr) {
    std::vector<float> vid = vid_patch, t = txt;
    return core_forward(vid, Lv, t, TXT, timestep, out_sr);
}

std::vector<float> DitVk::patchify_grid(const std::vector<float>& vid_grid, int T, int H, int W) {
    int H2 = H/2, W2 = W/2;
    int Lv = T*H2*W2;
    // patchify: (T,H,W,33) -> (Lv, 132), 顺序 (ph,pw,c)
    std::vector<float> vid_patch(Lv*132);
    for (int t = 0; t < T; t++)
        for (int hh = 0; hh < H2; hh++)
            for (int ww = 0; ww < W2; ww++)
                for (int ph = 0; ph < 2; ph++)
                    for (int pw = 0; pw < 2; pw++)
                        for (int c = 0; c < 33; c++) {
                            int gi = ((t*H + (hh*2+ph))*W + (ww*2+pw))*33 + c;
                            int pi = ((t*H2 + hh)*W2 + ww)*132 + ((ph*2+pw)*33 + c);
                            vid_patch[pi] = vid_grid[gi];
                        }
    return vid_patch;
}

std::vector<float> DitVk::unpatchify_latent(const std::vector<float>& sr_latent, int T, int H, int W) {
    int H2 = H/2, W2 = W/2;
    // unpatchify: (Lv,64) -> (T,H,W,16), 顺序 (ph,pw,c)
    std::vector<float> out_sr((size_t)T*H*W*16, 0.f);
    for (int t = 0; t < T; t++)
        for (int hh = 0; hh < H2; hh++)
            for (int ww = 0; ww < W2; ww++)
                for (int ph = 0; ph < 2; ph++)
                    for (int pw = 0; pw < 2; pw++)
                        for (int c = 0; c < 16; c++) {
                            int li = ((t*H2 + hh)*W2 + ww)*64 + ((ph*2+pw)*16 + c);
                            int gi = ((t*H + (hh*2+ph))*W + (ww*2+pw))*16 + c;
                            out_sr[gi] = sr_latent[li];
                        }
    return out_sr;
}

bool DitVk::forward_grid(const std::vector<float>& vid_grid, int T, int H, int W,
                         const std::vector<float>& txt, int TXT, float timestep,
                         std::vector<float>& sr_latent) {
    int H2 = H/2, W2 = W/2;
    int Lv = T*H2*W2;
    std::vector<float> vid_patch = patchify_grid(vid_grid, T, H, W);
    std::vector<float> out_sr, dummy_v, dummy_t;
    if (!forward_latent(vid_patch, Lv, txt, TXT, timestep, 0, NUM_LAYERS, true, true,
                        dummy_v, dummy_t, out_sr)) return false;
    sr_latent = unpatchify_latent(out_sr, T, H, W);
    return true;
}

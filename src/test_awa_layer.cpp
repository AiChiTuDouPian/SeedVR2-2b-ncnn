// test_awa_layer.cpp — 阶段1 单测：AwaLayer（自定义 ncnn::Layer）vs AwaVk 参考
// 合成小窗口（nwin=2, f_i=32, TXT=8, Lv=64），随机 qkv/nq/nk/freq，
// GPU Layer 输出与参考（awa_vk.cpp 的 AwaVk.forward）比对 cos，期望 = 1.0。
//
// 覆盖点：
//   - AwaLayer.create_pipeline / upload_model（常量预上传 GPU）
//   - forward(VkMat in -> VkMat out，全程 GPU 无 download)
//   - awa.comp（partition+qk_norm+3D mmrope+varlen SDPA+unpartition）
//   - awa_coalesce.comp（txt 跨窗口平均）
//   - ncnn dispatcher 语义：dispatcher.w 是【元素数】而非 workgroup 数（历史大坑）
//
// SEEDVR_TEST_REALWIN=1 时用真实 480p 变长窗口（nwin=4, TXT=58, Lv=1350），复现分块图场景。
//
// 用法：seedvr2_test_awa_layer [model_dir]   （model_dir 含 awa.spv / awa_coalesce.spv）
#include "awa_layer.h"
#include "awa_vk.h"
#include "awa_window.h"
#include "dit_vk.h"
#include <ncnn/gpu.h>
#include <ncnn/mat.h>
#include <ncnn/command.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>

static const int HEADS = 20, HEAD_D = 128, DIM = 2560, QKV = 7680;
static const int ROPE_ROT = 126;

// LCG 固定种子随机（可复现）
static unsigned int g_seed = 12345;
static float frand() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return ((g_seed >> 8) & 0xFFFFFF) / (float)0x1000000;  // [0,1)
}

static double cos_sim(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) { fprintf(stderr, "  size mismatch %zu vs %zu\n", a.size(), b.size()); return -1; }
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return dot / sqrt(na * nb);
}

int main(int argc, char** argv)
{
    std::string model_dir = argc > 1 ? argv[1] : "models/m5/";
    if (model_dir.empty() || model_dir.back() != '/') model_dir += '/';
    bool ok = false;
    // SEEDVR_TEST_LP: 0=默认 fp32, 1=fp16, 2=bf16（层内低16->fp32 cast 链验证）
    int lp = 0;
    if (const char* lpe = getenv("SEEDVR_TEST_LP")) lp = atoi(lpe);
    {
    // ---- 窗口几何：合成（默认）或真实 480p 变长（SEEDVR_TEST_REALWIN=1）----
    int TXT = 8, Lv = 64, nwin = 2;
    Win w;
    if (getenv("SEEDVR_TEST_REALWIN")) {
        // 真实 480p：token 网格 (1,30,45)，num_windows={4,3,3}，TXT=58
        int nw[3] = {4, 3, 3};
        w = awa::make_win(1, 30, 45, nw, 58, false);
        TXT = 58; Lv = 30 * 45; nwin = w.nwin;
        fprintf(stderr, "[win] 真实变长窗口 nwin=%d TXT=%d Lv=%d\n", nwin, TXT, Lv);
    } else {
        w.t = 1; w.h = 8; w.w = 8; w.nwin = nwin; w.txt_len = TXT;
        w.st = {0, 0}; w.en = {4, 4}; w.sh = {0, 0}; w.eh = {4, 4}; w.sw = {0, 2}; w.ew = {2, 4};
    }
    int sumf = 0;
    for (int wi = 0; wi < nwin; wi++) sumf += (w.en[wi]-w.st[wi]) * (w.eh[wi]-w.sh[wi]) * (w.ew[wi]-w.sw[wi]);
    // vidx / cumf（与 awa_vk.cpp / shader 相同公式）
    std::vector<float> vidx; vidx.reserve(sumf);
    std::vector<float> cumf(nwin + 1, 0.f);
    for (int wi = 0; wi < nwin; wi++) {
        int f_i = (w.en[wi]-w.st[wi]) * (w.eh[wi]-w.sh[wi]) * (w.ew[wi]-w.sw[wi]);
        cumf[wi + 1] = cumf[wi] + (float)f_i;
        for (int lt = w.st[wi]; lt < w.en[wi]; lt++)
            for (int lh = w.sh[wi]; lh < w.eh[wi]; lh++)
                for (int lw = w.sw[wi]; lw < w.ew[wi]; lw++)
                    vidx.push_back((float)(((lt * w.h) + lh) * w.w + lw));
    }
    // freq：合成用随机 0..2π；真实窗口用 build_freqs 生成的实际角度（w.vid_freq 已填充）
    if (!getenv("SEEDVR_TEST_REALWIN")) {
        w.vid_freq.resize(sumf * ROPE_ROT);
        w.txt_freq.resize(nwin * TXT * ROPE_ROT);
        for (auto& v : w.vid_freq) v = frand() * 6.2831853f;
        for (auto& v : w.txt_freq) v = frand() * 6.2831853f;
    }

    // ---- 输入：合成随机或真实 b0mid_v_qkv_0（SEEDVR_TEST_REALWIN 时）----
    std::vector<float> vqkv((size_t)Lv * QKV), tqkv((size_t)TXT * QKV);
    std::vector<float> nq_v(HEAD_D), nk_v(HEAD_D), nq_t(HEAD_D), nk_t(HEAD_D);
    if (getenv("SEEDVR_TEST_REALWIN") && getenv("SEEDVR_TEST_REALDATA")) {
        // 从 b0mid dump 读真实 qkv（v_qkv_0: Lv*QKV 展平 fp32 raw），qk_norm 用真实权重
        FILE* fq = fopen("b0mid_v_qkv_0.f32", "rb");
        if (fq) { fread(vqkv.data(), sizeof(float), vqkv.size(), fq); fclose(fq); }
        else { fprintf(stderr, "[data] 缺 b0mid_v_qkv_0.f32，退回随机\n"); }
        for (auto& v : tqkv) v = frand() * 2.f - 1.f;
        // 真实 qk_norm 权重（b0_vid_nq/nk, b0_txt_nq/nk：read_raw 格式）
        auto load_raw = [](const char* p, std::vector<float>& dst) {
            FILE* f = fopen(p, "rb"); if (!f) { fprintf(stderr, "[data] 缺 %s\n", p); return false; }
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            long skip = sz - (long)dst.size() * 4; if (skip < 0) skip = 0;
            fread((char*)dst.data(), 1, sz - skip, f); fclose(f); return true;
        };
        load_raw("models/m5/b0_vid_nq.bin", nq_v);
        load_raw("models/m5/b0_vid_nk.bin", nk_v);
        load_raw("models/m5/b0_txt_nq.bin", nq_t);
        load_raw("models/m5/b0_txt_nk.bin", nk_t);
        fprintf(stderr, "[data] 用真实 v_qkv_0 + 真实 qk_norm (Lv=%d QKV=%d)\n", Lv, QKV);
    } else {
        for (auto& v : vqkv) v = frand() * 2.f - 1.f;
        for (auto& v : tqkv) v = frand() * 2.f - 1.f;
        for (auto& v : nq_v) v = frand() * 2.f - 1.f;
        for (auto& v : nk_v) v = frand() * 2.f - 1.f;
        for (auto& v : nq_t) v = frand() * 2.f - 1.f;
        for (auto& v : nk_t) v = frand() * 2.f - 1.f;
    }

    if (ncnn::create_gpu_instance() != 0) { fprintf(stderr, "FAIL create_gpu_instance\n"); return 1; }
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(0);
    ncnn::VkAllocator* blob_alloc = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_alloc = vkdev->acquire_staging_allocator();

    // ---- 参考：AwaVk（awa_vk.cpp）----
    AwaVk ref;
    if (!ref.init(vkdev, model_dir + "awa.spv")) { fprintf(stderr, "FAIL AwaVk init\n"); return 1; }
    ref.set_allocators(blob_alloc, staging_alloc);
    std::vector<float> vattn_ref, tattn_ref;
    ref.forward(vqkv, tqkv, w, nq_v, nk_v, nq_t, nk_t, Lv, TXT, vattn_ref, tattn_ref);

    // 灵敏度自检：把输入在宿主侧量化成 fp16/bf16 后重跑 fp32 参考，
    // 评估「低精度舍入本身」对 AWA 输出的影响（排除 GPU cast 链问题）
    if (lp) {
        std::vector<float> vq_q = vqkv, tq_q = tqkv;
        if (lp == 2) {
            auto qb = [](float x) -> float {
                uint32_t u; memcpy(&u, &x, 4);
                uint32_t r = (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
                u = r << 16; float y; memcpy(&y, &u, 4); return y;
            };
            for (auto& v : vq_q) v = qb(v);
            for (auto& v : tq_q) v = qb(v);
        } else {
            auto rt16 = [](std::vector<float>& v) {
                ncnn::Mat m32; m32.create((int)v.size(), 1, (size_t)4u, 1);
                memcpy(m32.data, v.data(), v.size() * sizeof(float));
                ncnn::Mat m16; ncnn::cast_float32_to_float16(m32, m16);
                ncnn::Mat m32b; ncnn::cast_float16_to_float32(m16, m32b);
                memcpy(v.data(), m32b.data, v.size() * sizeof(float));
            };
            rt16(vq_q); rt16(tq_q);
        }
        std::vector<float> vr_q, tr_q;
        ref.forward(vq_q, tq_q, w, nq_v, nk_v, nq_t, nk_t, Lv, TXT, vr_q, tr_q);
        double cv_q = cos_sim(vr_q, vattn_ref), ct_q = cos_sim(tr_q, tattn_ref);
        fprintf(stderr, "  [灵敏度] 宿主%s量化输入 重跑 fp32 参考: vid cos=%.6f txt cos=%.6f\n",
                lp == 2 ? "bf16" : "fp16", cv_q, ct_q);
    }

    // ---- GPU：AwaLayer ----
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_storage = (lp == 1);
    opt.use_fp16_packed = false;
    opt.use_bf16_storage = (lp == 2);
    opt.use_bf16_packed = false;
    opt.blob_vkallocator = blob_alloc;
    opt.staging_vkallocator = staging_alloc;
    if (lp) fprintf(stderr, "[lp] 低精度模式 = %s\n", lp == 1 ? "fp16" : "bf16");
    // fp32 下载/独立验证用 option（flat-float shader 自检不允许低精度存储）
    ncnn::Option o32 = opt;
    o32.use_fp16_storage = o32.use_fp16_packed = false;
    o32.use_bf16_storage = o32.use_bf16_packed = false;

    AwaLayer layer;
    layer.set_vulkan_device(vkdev);
    layer.set_config(model_dir, 0, Lv, TXT);
    layer.set_qk_norm_host(nq_v.data(), nk_v.data(), nq_t.data(), nk_t.data());
    layer.set_window_geometry(vidx, cumf, w.vid_freq, w.txt_freq, nwin, Lv, TXT);
    if (layer.create_pipeline(opt) != 0) { fprintf(stderr, "FAIL create_pipeline\n"); return 1; }
    ncnn::VkTransfer vt(vkdev);
    if (layer.upload_model(vt, opt) != 0) { fprintf(stderr, "FAIL upload_model\n"); return 1; }
    vt.submit_and_wait();

    // bottom 上传（2D Mat 避免 pack4）
    auto upload = [&](const std::vector<float>& src, ncnn::VkMat& dst, ncnn::VkCompute& c, const ncnn::Option& uo) {
        ncnn::Mat m; m.create((int)src.size(), 1, (size_t)4u, 1);
        memcpy(m.data, src.data(), src.size() * sizeof(float));
        c.record_upload(m, dst, uo);
    };
    ncnn::VkCompute up(vkdev);
    ncnn::VkMat vqkv_vk, tqkv_vk;
    upload(vqkv, vqkv_vk, up, opt);
    upload(tqkv, tqkv_vk, up, opt);
    up.submit_and_wait();
    up.reset();
    // DIAG：dump 上传后原始字节（验证 fp16/bf16 打包布局：2 half/uint？字节流？）
    if (getenv("SEEDVR_DUMP_AWAIN")) {
        ncnn::VkCompute dc(vkdev);
        ncnn::Mat dm;
        ncnn::Option du = opt; du.use_packing_layout = false;
        dc.record_download(vqkv_vk, dm, du);
        dc.submit_and_wait(); dc.reset();
        FILE* fv = fopen("lp_in_vqkv.bin", "wb");
        if (fv) { fwrite(dm.data, 1, dm.total() * (size_t)dm.elemsize, fv); fclose(fv);
            fprintf(stderr, "[lpIn] dumped lp_in_vqkv.bin elemsize=%zu total=%zu ep=%d\n",
                    (size_t)dm.elemsize, dm.total(), dm.elempack); }
    }

    ncnn::VkCompute cmd(vkdev);
    std::vector<ncnn::VkMat> bottoms = { vqkv_vk, tqkv_vk };
    std::vector<ncnn::VkMat> tops(2);
    if (layer.forward(bottoms, tops, cmd, opt) != 0) { fprintf(stderr, "FAIL forward\n"); return 1; }
    ncnn::Mat m_vattn, m_tattn;
    cmd.record_download(tops[0], m_vattn, o32);
    cmd.record_download(tops[1], m_tattn, o32);
    cmd.submit_and_wait();
    cmd.reset();
    // 下载回的 top 在低精度模式下可能是 fp16/bf16（elemsize=2），需显式转 fp32 才能按 float 读
    auto ensure_f32 = [&](ncnn::Mat& m) {
        if (m.elemsize == 2) { ncnn::Mat m32; 
            if (lp == 2) ncnn::cast_bfloat16_to_float32(m, m32, o32);
            else         ncnn::cast_float16_to_float32(m, m32, o32);
            m = m32; }
    };
    ensure_f32(m_vattn);
    ensure_f32(m_tattn);

    // 诊断：dump 上传后的输入 vqkv_vk，确认 shader 读到的输入是否 = b0mid_v_qkv_0
    if (getenv("SEEDVR_TEST_REALDATA") && getenv("SEEDVR_DUMP_AWAIN")) {
        ncnn::VkCompute dc(vkdev);
        ncnn::Mat dm;
        dc.record_download(vqkv_vk, dm, opt);
        dc.submit_and_wait(); dc.reset();
        FILE* fv = fopen("awa_in_vqkv_vk.f32", "wb");
        if (fv) { fwrite(dm.data, 1, dm.total()*dm.elemsize, fv); fclose(fv);
            float mn=1e30f,mx=-1e30f; const float* dd=(const float*)dm.data;
            for (size_t q=0;q<(size_t)dm.w*dm.h;q++){float v=dd[q];if(v<mn)mn=v;if(v>mx)mx=v;}
            fprintf(stderr, "[AwaIn] dumped awa_in_vqkv_vk.f32 w=%d h=%d ep=%d range[%.4f,%.4f]\n", dm.w, dm.h, dm.elempack, mn, mx);
        }
    }

    std::vector<float> vattn_gpu((const float*)m_vattn.data, (const float*)m_vattn.data + (size_t)Lv * DIM);
    std::vector<float> tattn_gpu((const float*)m_tattn.data, (const float*)m_tattn.data + (size_t)TXT * DIM);

    // ---- 独立验证 coalesce shader（已知输入 + 深部检查，防 dispatcher 假象）----
    bool coal_ok = true;
    {
        const int nwin_c = 2, txt_c = 8, dim_c = 2560;
        std::vector<float> inw((size_t)nwin_c * txt_c * dim_c);
        for (size_t i = 0; i < inw.size(); i++) inw[i] = (float)i;
        std::ifstream f(model_dir + "awa_coalesce.spv", std::ios::binary);
        std::vector<uint32_t> cspv;
        f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg(); f.seekg(0);
        cspv.resize(n / 4); f.read((char*)cspv.data(), n);
        ncnn::Pipeline cp(vkdev);
        cp.set_local_size_xyz(256, 1, 1);
        if (cp.create(cspv.data(), cspv.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0)
        { fprintf(stderr, "FAIL coal pipeline\n"); return 1; }
        ncnn::VkCompute cc(vkdev);
        ncnn::VkMat vk_in, vk_res;
        upload(inw, vk_in, cc, o32);
        vk_res.create(dim_c, txt_c, 4u, blob_alloc);
        std::vector<ncnn::VkMat> cb = { vk_in, vk_res };
        std::vector<ncnn::vk_constant_type> ccst(3);
        ccst[0].i = nwin_c; ccst[1].i = txt_c; ccst[2].i = dim_c;
        ncnn::VkMat cdis; cdis.w = txt_c * dim_c; cdis.h = 1; cdis.c = 1;  // 元素数语义！
        cc.record_pipeline(&cp, cb, ccst, cdis);
        ncnn::Mat m_res;
        cc.record_download(vk_res, m_res, o32);
        cc.submit_and_wait(); cc.reset();
        auto check = [&](int i) {
            float expect = (inw[i] + inw[(size_t)nwin_c * txt_c * dim_c / 2 + i]) * 0.5f;
            float got = ((const float*)m_res.data)[i];
            if (fabsf(got - expect) > 1e-4f) { coal_ok = false; fprintf(stderr, "  coal[%d] got=%.4f expect=%.4f\n", i, got, expect); }
        };
        for (int i = 0; i < 32; i++) check(i);
        check(3000);
        check(txt_c * dim_c - 1);   // 尾部
        fprintf(stderr, "  [coalesce 独立验证] %s\n", coal_ok ? "PASS ✅" : "FAIL ❌");

        // ---- 低精度 cast shader 直接自测（隔离 AWA 核心）：读取侧 b2f/h2f + 写入侧 f2b/f2h ----
        if (lp) {
            const int NC = 8192;
            std::vector<float> src(NC);
            for (auto& v : src) v = frand() * 4.f - 2.f;
            ncnn::Mat m32; m32.create(NC, 1, (size_t)4u, 1);
            memcpy(m32.data, src.data(), NC * sizeof(float));
            ncnn::Mat mlp, mref;
            if (lp == 2) { ncnn::cast_float32_to_bfloat16(m32, mlp); ncnn::cast_bfloat16_to_float32(mlp, mref); }
            else         { ncnn::cast_float32_to_float16(m32, mlp); ncnn::cast_float16_to_float32(mlp, mref); }
            const float* refd = (const float*)mref.data;
            // 期望：host 量化往返
            auto mkpipe = [&](const char* fn, ncnn::Pipeline& pp, int ls) {
                std::ifstream fs(model_dir + fn, std::ios::binary);
                std::vector<uint32_t> sv;
                fs.seekg(0, std::ios::end); size_t nn = (size_t)fs.tellg(); fs.seekg(0);
                sv.resize(nn / 4); fs.read((char*)sv.data(), nn);
                pp.set_local_size_xyz(ls, 1, 1);
                return pp.create(sv.data(), sv.size() * 4, std::vector<ncnn::vk_specialization_type>()) == 0;
            };
            // (1) 读侧：fp32 按 lp 上传（storage 低16） -> b2f/h2f -> fp32 下载，对比 host 量化
            {
                ncnn::VkCompute c1(vkdev);
                ncnn::VkMat vk_lp;
                c1.record_upload(m32, vk_lp, opt);   // lp opt -> 上传即转 fp16/bf16
                c1.submit_and_wait(); c1.reset();
                ncnn::Pipeline pp(vkdev);
                if (mkpipe(lp == 2 ? "cast_bf16_f32.spv" : "cast_f16_f32.spv", pp, 128)) {
                    ncnn::VkMat vk_f;
                    vk_f.create(NC, 1, 4u, blob_alloc);
                    std::vector<ncnn::VkMat> b = { vk_lp, vk_f };
                    std::vector<ncnn::vk_constant_type> cs(1); cs[0].i = NC;
                    ncnn::VkMat d; d.w = NC; d.h = 1; d.c = 1;
                    ncnn::VkCompute c2(vkdev);
                    c2.record_pipeline(&pp, b, cs, d);
                    ncnn::Mat mout;
                    c2.record_download(vk_f, mout, o32);
                    c2.submit_and_wait(); c2.reset();
                    const float* od = (const float*)mout.data;
                    double bad = 0; float mx = 0;
                    for (int i = 0; i < NC; i++) { float e = fabsf(od[i] - refd[i]); if (e > mx) mx = e; if (e > 1e-3f) bad++; }
                    fprintf(stderr, "  [cast读侧 %s] %s (max|d|=%.5f 坏点=%d/%d)\n",
                            lp == 2 ? "bf16" : "fp16", bad == 0 ? "PASS ✅" : "FAIL ❌", mx, (int)bad, NC);
                }
            }
            // (2) 写侧：fp32 随机 -> f2b/f2h -> 下载原始低16 字节（o32 不转）-> host 解回，对比 host 量化
            {
                ncnn::VkCompute c1(vkdev);
                ncnn::VkMat vk_f;
                c1.record_upload(m32, vk_f, o32);
                c1.submit_and_wait(); c1.reset();
                ncnn::Pipeline pp(vkdev);
                if (mkpipe(lp == 2 ? "cast_f32_bf16.spv" : "cast_f32_f16.spv", pp, 128)) {
                    ncnn::VkMat vk_lp;
                    vk_lp.create(NC, 1, 2u, blob_alloc);   // 2B/元素输出
                    std::vector<ncnn::VkMat> b = { vk_f, vk_lp };
                    std::vector<ncnn::vk_constant_type> cs(1); cs[0].i = NC;
                    ncnn::VkMat d; d.w = NC; d.h = 1; d.c = 1;
                    ncnn::VkCompute c2(vkdev);
                    c2.record_pipeline(&pp, b, cs, d);
                    ncnn::Mat mout;
                    c2.record_download(vk_lp, mout, o32);
                    c2.submit_and_wait(); c2.reset();
                    ncnn::Mat mdec;
                    if (lp == 2) ncnn::cast_bfloat16_to_float32(mout, mdec, o32);
                    else         ncnn::cast_float16_to_float32(mout, mdec, o32);
                    const float* od = (const float*)mdec.data;
                    double bad = 0; float mx = 0;
                    for (int i = 0; i < NC; i++) { float e = fabsf(od[i] - refd[i]); if (e > mx) mx = e; if (e > 1e-3f) bad++; }
                    fprintf(stderr, "  [cast写侧 %s] %s (max|d|=%.5f 坏点=%d/%d)\n",
                            lp == 2 ? "bf16" : "fp16", bad == 0 ? "PASS ✅" : "FAIL ❌", mx, (int)bad, NC);
                }
            }
        }
    }

    if (getenv("SEEDVR_TEST_REALWIN") && getenv("SEEDVR_TEST_REALDATA")) {
        FILE* fg = fopen("awa_out_vattn.f32", "wb");
        if (fg) { fwrite(vattn_gpu.data(), sizeof(float), vattn_gpu.size(), fg); fclose(fg);
            fprintf(stderr, "[data] dump awa_out_vattn.f32 (%zu floats)\n", vattn_gpu.size()); }
    }
    double cv = cos_sim(vattn_gpu, vattn_ref);
    double ct = cos_sim(tattn_gpu, tattn_ref);
    fprintf(stderr, "\n========== 阶段1 AWA Layer 单测结果 ==========\n");
    fprintf(stderr, "  vid 输出 cos = %.10f   (期望 1.0)\n", cv);
    fprintf(stderr, "  txt 输出 cos = %.10f   (期望 1.0)\n", ct);
    ok = coal_ok && (cv > 1 - 1e-6) && (ct > 1 - 1e-6);
    if (lp == 2) ok = coal_ok && (cv > 0.999) && (ct > 0.999);   // bf16 量化级误差阈值
    fprintf(stderr, "  ==> %s\n", ok ? "PASS ✅" : "FAIL ❌");
    fprintf(stderr, "==============================================\n");

    // 析构顺序坑：必须先释放引用 GPU 设备的资源，再销毁 GPU 实例；
    // 且 layer 的 VkMat 常量成员也要在 GPU 实例销毁前析构 -> 包在作用域块里
    layer.destroy_pipeline(opt);
    ref.release();
    }   // 作用域结束：layer（含 VkMat 成员）/ref 先析构，GPU 实例仍活

    ncnn::destroy_gpu_instance();
    return ok ? 0 : 1;
}

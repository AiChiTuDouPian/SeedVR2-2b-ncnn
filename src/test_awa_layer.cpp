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
// 用法：seedvr2_test_awa_layer [model_dir]   （model_dir 含 awa.spv / awa_coalesce.spv）
#include "awa_layer.h"
#include "awa_vk.h"
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
    {
    // ---- 合成窗口几何：t=1,h=8,w=8, nwin=2, 每窗口 4*4*2=32 ----
    const int TXT = 8, Lv = 64, nwin = 2;
    Win w;
    w.t = 1; w.h = 8; w.w = 8; w.nwin = nwin; w.txt_len = TXT;
    w.st = {0, 0}; w.en = {4, 4}; w.sh = {0, 0}; w.eh = {4, 4}; w.sw = {0, 2}; w.ew = {2, 4};
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
    // freq（随机 0..2π）
    w.vid_freq.resize(sumf * ROPE_ROT);
    w.txt_freq.resize(nwin * TXT * ROPE_ROT);
    for (auto& v : w.vid_freq) v = frand() * 6.2831853f;
    for (auto& v : w.txt_freq) v = frand() * 6.2831853f;

    // ---- 随机输入（均匀 [-1,1]）----
    std::vector<float> vqkv((size_t)Lv * QKV), tqkv((size_t)TXT * QKV);
    std::vector<float> nq_v(HEAD_D), nk_v(HEAD_D), nq_t(HEAD_D), nk_t(HEAD_D);
    for (auto& v : vqkv) v = frand() * 2.f - 1.f;
    for (auto& v : tqkv) v = frand() * 2.f - 1.f;
    for (auto& v : nq_v) v = frand() * 2.f - 1.f;
    for (auto& v : nk_v) v = frand() * 2.f - 1.f;
    for (auto& v : nq_t) v = frand() * 2.f - 1.f;
    for (auto& v : nk_t) v = frand() * 2.f - 1.f;

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

    // ---- GPU：AwaLayer ----
    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.use_fp16_storage = opt.use_fp16_packed = false;
    opt.use_bf16_storage = opt.use_bf16_packed = false;
    opt.blob_vkallocator = blob_alloc;
    opt.staging_vkallocator = staging_alloc;

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
    auto upload = [&](const std::vector<float>& src, ncnn::VkMat& dst, ncnn::VkCompute& c) {
        ncnn::Mat m; m.create((int)src.size(), 1, (size_t)4u, 1);
        memcpy(m.data, src.data(), src.size() * sizeof(float));
        c.record_upload(m, dst, opt);
    };
    ncnn::VkCompute up(vkdev);
    ncnn::VkMat vqkv_vk, tqkv_vk;
    upload(vqkv, vqkv_vk, up);
    upload(tqkv, tqkv_vk, up);
    up.submit_and_wait();
    up.reset();

    ncnn::VkCompute cmd(vkdev);
    std::vector<ncnn::VkMat> bottoms = { vqkv_vk, tqkv_vk };
    std::vector<ncnn::VkMat> tops(2);
    if (layer.forward(bottoms, tops, cmd, opt) != 0) { fprintf(stderr, "FAIL forward\n"); return 1; }
    ncnn::Mat m_vattn, m_tattn;
    cmd.record_download(tops[0], m_vattn, opt);
    cmd.record_download(tops[1], m_tattn, opt);
    cmd.submit_and_wait();
    cmd.reset();

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
        upload(inw, vk_in, cc);
        vk_res.create(dim_c, txt_c, 4u, blob_alloc);
        std::vector<ncnn::VkMat> cb = { vk_in, vk_res };
        std::vector<ncnn::vk_constant_type> ccst(3);
        ccst[0].i = nwin_c; ccst[1].i = txt_c; ccst[2].i = dim_c;
        ncnn::VkMat cdis; cdis.w = txt_c * dim_c; cdis.h = 1; cdis.c = 1;  // 元素数语义！
        cc.record_pipeline(&cp, cb, ccst, cdis);
        ncnn::Mat m_res;
        cc.record_download(vk_res, m_res, opt);
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
    }

    double cv = cos_sim(vattn_gpu, vattn_ref);
    double ct = cos_sim(tattn_gpu, tattn_ref);
    fprintf(stderr, "\n========== 阶段1 AWA Layer 单测结果 ==========\n");
    fprintf(stderr, "  vid 输出 cos = %.10f   (期望 1.0)\n", cv);
    fprintf(stderr, "  txt 输出 cos = %.10f   (期望 1.0)\n", ct);
    ok = coal_ok && (cv > 1 - 1e-6) && (ct > 1 - 1e-6);
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

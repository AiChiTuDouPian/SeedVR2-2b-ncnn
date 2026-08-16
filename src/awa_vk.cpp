// awa_vk.cpp — Adaptive Window Attention 自定义 Vulkan 模块实现
#include "awa_vk.h"
#include "dit_vk.h"   // 完整 Win 定义
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/mat.h>
#include <ncnn/pipeline.h>
#include <ncnn/command.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>

// 与 dit_vk.cpp 一致的常量
static const int HEADS = 20, HEAD_D = 128, DIM = 2560;
static const int ROPE_ROT = 126;
static const float EPS = 1e-5f;

AwaVk::AwaVk() {}
AwaVk::~AwaVk() {
    if (pipe) { delete pipe; pipe = nullptr; }
}

static std::vector<uint32_t> read_spv(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[AwaVk] 无 spv %s（将走 CPU 参考分支）\n", path.c_str()); return {}; }
    f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint32_t> d(n / 4); f.read((char*)d.data(), n);
    return d;
}

bool AwaVk::init(ncnn::VulkanDevice* vkdev, const std::string& spv_path) {
    this->vkdev = vkdev;
    std::vector<uint32_t> spv = read_spv(spv_path);
    if (spv.empty()) { fprintf(stderr, "[AwaVk] init 跳过（spv 缺失）\n"); return false; }
    pipe = new ncnn::Pipeline(vkdev);
    // 多线程优化：每个 workgroup 128 线程处理一个 (window, head)，共享内存缓存 K/V 块。
    // 与 shader 声明的 local_size (128,1,1) 一致。
    pipe->set_local_size_xyz(128, 1, 1);
    if (pipe->create(spv.data(), spv.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0) {
        fprintf(stderr, "[AwaVk] FAIL create pipeline from %s\n", spv_path.c_str());
        return false;
    }
    fprintf(stderr, "[AwaVk] init ok (spv=%s, %zu dwords)\n", spv_path.c_str(), spv.size());
    return true;
}

bool AwaVk::forward(const std::vector<float>& vqkv, const std::vector<float>& tqkv,
                    const Win& win,
                    const std::vector<float>& nq_v, const std::vector<float>& nk_v,
                    const std::vector<float>& nq_t, const std::vector<float>& nk_t,
                    int Lv, int TXT,
                    std::vector<float>& vid_out, std::vector<float>& txt_out) {
    if (!pipe) { fprintf(stderr, "[AwaVk] pipe 未初始化\n"); return false; }
    int T = win.t, H = win.h, Wd = win.w;
    int nwin = win.nwin;

    // ---- 复刻 awa_forward 的窗口 partition 索引 + cumf ----
    std::vector<float> cumf(nwin + 1, 0.f);
    std::vector<float> vidx;
    vidx.reserve(Lv);
    int maxS = 1;
    for (int wi = 0; wi < nwin; wi++) {
        int f_i = (win.en[wi] - win.st[wi]) * (win.eh[wi] - win.sh[wi]) * (win.ew[wi] - win.sw[wi]);
        cumf[wi + 1] = cumf[wi] + (float)f_i;
        if (f_i + TXT > maxS) maxS = f_i + TXT;
        for (int lt = win.st[wi]; lt < win.en[wi]; lt++)
            for (int lh = win.sh[wi]; lh < win.eh[wi]; lh++)
                for (int lw = win.sw[wi]; lw < win.ew[wi]; lw++) {
                    int idx = ((lt * H) + lh) * Wd + lw;
                    vidx.push_back((float)idx);
                }
    }
    int sumf = (int)vidx.size();

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    // 关键：自定义 shader 以扁平 float[] 索引缓冲（vqkv[tok*QKV+...]），
    // 必须关掉 packing，否则 ncnn 会把 VkMat 按 elempack=4 打包，
    // 输入上传与输出回读都会错位（m_vattn.w 变成 1/4）。
    opt.use_packing_layout = false;
    // 关键2：离散 GPU 默认 use_fp16_packed=true，record_upload 会把 fp32 数据转成 fp16 存入 buffer；
    // 自定义 shader 用 flat float[] 读取时必须保持 fp32，否则读到的是 fp16 字节(被当 fp32 解读=乱码/全零)。
    opt.use_fp16_storage = false;
    opt.use_fp16_packed  = false;
    opt.use_bf16_storage = false;
    opt.use_bf16_packed  = false;
    opt.blob_vkallocator    = blob_alloc    ? blob_alloc    : vkdev->acquire_blob_allocator();
    opt.staging_vkallocator = staging_alloc ? staging_alloc : vkdev->acquire_staging_allocator();

    ncnn::VkCompute cmd(vkdev);

    auto upload = [&](const std::vector<float>& src, ncnn::VkMat& dst) {
        // 用 2 维 Mat(w=size, h=1)：record_upload 对 1 维 Mat 会按 elemcount=w 自动 pack4，
        // 当 w/4 > 65535(Vulkan maxComputeWorkGroupCount[y]) 时 pack4 静默失败 → 大 buffer 损坏。
        // 2 维 Mat 的 elemcount=h=1 → dst_elempack=1，不 pack，绕开该限制。
        ncnn::Mat m; m.create((int)src.size(), 1, sizeof(float), 1);
        memcpy(m.data, src.data(), src.size() * sizeof(float));
        cmd.record_upload(m, dst, opt);
    };

    ncnn::VkMat vk_vqkv, vk_tqkv, vk_vidx, vk_cumf, vk_vfreq, vk_tfreq;
    ncnn::VkMat vk_nqv, vk_nkv, vk_nqt, vk_nkt;
    upload(vqkv,  vk_vqkv);
    upload(tqkv,  vk_tqkv);
    upload(vidx,  vk_vidx);
    upload(cumf,  vk_cumf);
    upload(win.vid_freq, vk_vfreq);
    upload(win.txt_freq, vk_tfreq);
    upload(nq_v,  vk_nqv);
    upload(nk_v,  vk_nkv);
    upload(nq_t,  vk_nqt);
    upload(nk_t,  vk_nkt);
    fprintf(stderr, "[AwaVk] upload 后 vqkv.w=%d empty=%d  vidx.w=%d empty=%d  vfreq.w=%d empty=%d\n",
            vk_vqkv.w, (int)vk_vqkv.empty(), vk_vidx.w, (int)vk_vidx.empty(),
            vk_vfreq.w, (int)vk_vfreq.empty());

    ncnn::VkMat vk_vattn, vk_toutw;
    vk_vattn.create(Lv * DIM, (size_t)4, blob_alloc ? blob_alloc : vkdev->acquire_blob_allocator());
    vk_toutw.create((size_t)nwin * TXT * DIM, (size_t)4, blob_alloc ? blob_alloc : vkdev->acquire_blob_allocator());
    fprintf(stderr, "[AwaVk] upload/alloc done  Lv=%d TXT=%d nwin=%d sumf=%d vattn.w=%d toutw.w=%d\n",
            Lv, TXT, nwin, sumf, vk_vattn.w, vk_toutw.w);

    std::vector<ncnn::VkMat> bindings(12);
    bindings[0]  = vk_vqkv;
    bindings[1]  = vk_tqkv;
    bindings[2]  = vk_vidx;
    bindings[3]  = vk_cumf;
    bindings[4]  = vk_vfreq;
    bindings[5]  = vk_tfreq;
    bindings[6]  = vk_nqv;
    bindings[7]  = vk_nkv;
    bindings[8]  = vk_nqt;
    bindings[9]  = vk_nkt;
    bindings[10] = vk_vattn;
    bindings[11] = vk_toutw;

    std::vector<ncnn::vk_constant_type> constants(10);
    constants[0].i = DIM;
    constants[1].i = HEAD_D;
    constants[2].i = HEADS;
    constants[3].i = ROPE_ROT;
    constants[4].i = nwin;
    constants[5].i = TXT;
    constants[6].i = Lv;
    constants[7].i = sumf;
    constants[8].f = 1.0f / sqrtf((float)HEAD_D);
    constants[9].f = EPS;

    ncnn::VkMat dispatcher;
    dispatcher.w = 1;      // 每个 workgroup 处理一个 (window, head) 的全部 query（shader 内循环）
    dispatcher.h = HEADS;
    dispatcher.c = nwin;

    cmd.record_pipeline(pipe, bindings, constants, dispatcher);

    ncnn::Mat m_vattn, m_toutw;
    cmd.record_download(vk_vattn, m_vattn, opt);
    cmd.record_download(vk_toutw, m_toutw, opt);
    fprintf(stderr, "[AwaVk] dispatch + download recorded, submitting...\n");
    cmd.submit_and_wait();
    cmd.reset();
    fprintf(stderr, "[AwaVk] submit done  m_vattn.w=%d m_toutw.w=%d\n", m_vattn.w, m_toutw.w);
    if (m_vattn.w != Lv * DIM) { fprintf(stderr, "[AwaVk] FAIL m_vattn.w=%d != %d\n", m_vattn.w, Lv*DIM); return false; }

    vid_out.assign((const float*)m_vattn.data,
                   (const float*)m_vattn.data + (size_t)m_vattn.w);   // Lv*DIM
    // txt 跨窗口平均（coalesce）：toutw 形状 nwin*TXT*DIM -> TXT*DIM
    txt_out.assign((size_t)TXT * DIM, 0.f);
    const float* tw = (const float*)m_toutw.data;
    for (int wi = 0; wi < nwin; wi++)
        for (int ii = 0; ii < TXT; ii++)
            for (int d = 0; d < DIM; d++)
                txt_out[ii * DIM + d] += tw[((wi * TXT + ii) * DIM) + d];
    for (size_t i = 0; i < txt_out.size(); i++) txt_out[i] /= (float)nwin;

    return true;
}

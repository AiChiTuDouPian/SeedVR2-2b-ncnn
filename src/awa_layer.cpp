// awa_layer.cpp — AwaLayer：AWA 自定义 ncnn::Layer 实现（阶段1）
// 数据流（全程 GPU，无 download）：
//   bottom [vqkv(Lv*QKV), tqkv(TXT*QKV)]
//     -> awa.comp（partition+qk_norm+3D mmrope+varlen SDPA+unpartition）
//     -> toutw(nwin*TXT*DIM) -> awa_coalesce.comp（txt 跨窗口平均）
//     -> top [vattn(Lv*DIM), tattn(TXT*DIM)]
#include "awa_layer.h"
#include "dit_vk.h"   // Win / load_win（窗口几何与 freq 解析复用）
#include <ncnn/gpu.h>
#include <ncnn/pipeline.h>
#include <ncnn/command.h>
#include <ncnn/paramdict.h>
#include <ncnn/modelbin.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>

const float AwaLayer::SCALE = 1.0f / std::sqrt((float)AwaLayer::HEAD_D);
const float AwaLayer::EPS = 1e-5f;

// 构造函数在头文件内联（support_bf16_storage=true：低精度模式 I/O 为 bf16，层内自行转换）

AwaLayer::~AwaLayer()
{
    destroy_pipeline(ncnn::Option());
}

int AwaLayer::load_param(const ncnn::ParamDict& pd)
{
    spv_dir_  = pd.get(0, spv_dir_);
    win_type_ = pd.get(1, 0);
    Lv_       = pd.get(2, 0);
    TXT_      = pd.get(3, 0);
    return 0;
}

int AwaLayer::load_model(const ncnn::ModelBin& mb)
{
    ncnn::Mat m0 = mb.load(128, 0);
    ncnn::Mat m1 = mb.load(128, 0);
    ncnn::Mat m2 = mb.load(128, 0);
    ncnn::Mat m3 = mb.load(128, 0);
    if (m0.empty() || m1.empty() || m2.empty() || m3.empty())
    {
        fprintf(stderr, "[AwaLayer] load_model FAIL：4 个 qk_norm 权重缺失\n");
        return -1;
    }
    nq_v_.assign((const float*)m0.data, (const float*)m0.data + 128);
    nk_v_.assign((const float*)m1.data, (const float*)m1.data + 128);
    nq_t_.assign((const float*)m2.data, (const float*)m2.data + 128);
    nk_t_.assign((const float*)m3.data, (const float*)m3.data + 128);
    return 0;
}

void AwaLayer::set_config(const std::string& spv_dir, int win_type, int Lv, int TXT)
{
    spv_dir_ = spv_dir;
    if (!spv_dir_.empty() && spv_dir_.back() != '/') spv_dir_ += '/';
    win_type_ = win_type;
    Lv_ = Lv;
    TXT_ = TXT;
}

void AwaLayer::set_qk_norm_host(const float* nqv, const float* nkv, const float* nqt, const float* nkt)
{
    nq_v_.assign(nqv, nqv + 128);
    nk_v_.assign(nkv, nkv + 128);
    nq_t_.assign(nqt, nqt + 128);
    nk_t_.assign(nkt, nkt + 128);
}

void AwaLayer::set_window_geometry(const std::vector<float>& vidx, const std::vector<float>& cumf,
                                   const std::vector<float>& vfreq, const std::vector<float>& tfreq,
                                   int nwin, int Lv, int TXT)
{
    vidx_ = vidx;
    cumf_ = cumf;
    vfreq_ = vfreq;
    tfreq_ = tfreq;
    nwin_ = nwin;
    sumf_ = (int)vidx.size();
    Lv_ = Lv;
    TXT_ = TXT;
    geom_injected_ = true;
    // 窗口变化：失效已上传的 GPU 侧几何（next forward 重新上传）
    vk_vidx = ncnn::VkMat();
    vk_cumf = ncnn::VkMat();
    vk_vfreq = ncnn::VkMat();
    vk_tfreq = ncnn::VkMat();
}

bool AwaLayer::read_spv_file(const std::string& path, std::vector<uint32_t>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        fprintf(stderr, "[AwaLayer] 无 spv %s\n", path.c_str());
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    out.resize(n / 4);
    if (n) f.read((char*)out.data(), n);
    return !out.empty();
}

// 从 spv_dir 读 win_{nonshifted,shifted}.bin，构建 vidx/cumf/vfreq/tfreq（与 awa_vk.cpp 一致）
int AwaLayer::build_window_geometry()
{
    std::string winpath = spv_dir_ + (win_type_ == 0 ? "win_nonshifted.bin" : "win_shifted.bin");
    Win w = load_win(winpath.c_str(), TXT_);
    nwin_ = w.nwin;

    cumf_.assign(nwin_ + 1, 0.f);
    vidx_.clear();
    vidx_.reserve(Lv_);
    for (int wi = 0; wi < nwin_; wi++)
    {
        int f_i = (w.en[wi] - w.st[wi]) * (w.eh[wi] - w.sh[wi]) * (w.ew[wi] - w.sw[wi]);
        cumf_[wi + 1] = cumf_[wi] + (float)f_i;
        for (int lt = w.st[wi]; lt < w.en[wi]; lt++)
            for (int lh = w.sh[wi]; lh < w.eh[wi]; lh++)
                for (int lw = w.sw[wi]; lw < w.ew[wi]; lw++)
                    vidx_.push_back((float)(((lt * w.h) + lh) * w.w + lw));
    }
    sumf_ = (int)vidx_.size();
    vfreq_ = w.vid_freq;
    tfreq_ = w.txt_freq;
    fprintf(stderr, "[AwaLayer] win(%s) nwin=%d Lv=%d sumf=%d vfreq=%zu tfreq=%zu\n",
            win_type_ == 0 ? "nonshifted" : "shifted", nwin_, Lv_, sumf_,
            vfreq_.size(), tfreq_.size());
    if (sumf_ * ROPE_ROT != (int)vfreq_.size())
    {
        fprintf(stderr, "[AwaLayer] FAIL sumf*ROPE_ROT=%d != vfreq=%zu\n", sumf_ * ROPE_ROT, vfreq_.size());
        return -1;
    }
    geom_injected_ = false;
    return 0;
}

int AwaLayer::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute) return 0;

    if (!geom_injected_ && sumf_ <= 0)
    {
        // 尝试从 win bin 构建窗口几何；失败不致命（真实窗口由宿主运行时注入）
        if (build_window_geometry() != 0)
            fprintf(stderr, "[AwaLayer] 警告：窗口几何未就绪，等待运行时注入\n");
    }

    std::vector<uint32_t> spv;
    if (!read_spv_file(spv_dir_ + "awa.spv", spv)) return -1;
    awa_pipe_ = new ncnn::Pipeline(vkdev);
    awa_pipe_->set_local_size_xyz(128, 1, 1);
    if (awa_pipe_->create(spv.data(), spv.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0)
    {
        fprintf(stderr, "[AwaLayer] FAIL create awa pipeline\n");
        return -1;
    }

    std::vector<uint32_t> cspv;
    if (!read_spv_file(spv_dir_ + "awa_coalesce.spv", cspv)) return -1;
    coal_pipe_ = new ncnn::Pipeline(vkdev);
    coal_pipe_->set_local_size_xyz(256, 1, 1);
    if (coal_pipe_->create(cspv.data(), cspv.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0)
    {
        fprintf(stderr, "[AwaLayer] FAIL create coalesce pipeline\n");
        return -1;
    }

    std::vector<uint32_t> ispv;
    if (!read_spv_file(spv_dir_ + "awa_init.spv", ispv)) return -1;
    init_pipe_ = new ncnn::Pipeline(vkdev);
    init_pipe_->set_local_size_xyz(128, 1, 1);
    if (init_pipe_->create(ispv.data(), ispv.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0)
    {
        fprintf(stderr, "[AwaLayer] FAIL create init pipeline\n");
        return -1;
    }

    // 低精度（bf16）I/O 转换 pipeline（cast_bf16_f32 / cast_f32_bf16）
    {
        std::vector<uint32_t> s1;
        if (read_spv_file(spv_dir_ + "cast_bf16_f32.spv", s1)) {
            b2f_pipe_ = new ncnn::Pipeline(vkdev);
            b2f_pipe_->set_local_size_xyz(128, 1, 1);
            if (b2f_pipe_->create(s1.data(), s1.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0) {
                fprintf(stderr, "[AwaLayer] FAIL create bf16->f32 pipeline\n");
                delete b2f_pipe_; b2f_pipe_ = nullptr;
            }
        }
        std::vector<uint32_t> s2;
        if (read_spv_file(spv_dir_ + "cast_f32_bf16.spv", s2)) {
            f2b_pipe_ = new ncnn::Pipeline(vkdev);
            f2b_pipe_->set_local_size_xyz(128, 1, 1);
            if (f2b_pipe_->create(s2.data(), s2.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0) {
                fprintf(stderr, "[AwaLayer] FAIL create f32->bf16 pipeline\n");
                delete f2b_pipe_; f2b_pipe_ = nullptr;
            }
        }
    }
    // 低精度（fp16）I/O 转换 pipeline（cast_f16_f32 / cast_f32_f16）
    // fp16 存储也是 2 字节/元素，但 half->fp32 需要指数/尾数重排（unpackHalf2x16），
    // 不能复用 bf16 的纯位对齐 (u<<16)，故独立管线。
    {
        std::vector<uint32_t> s3;
        if (read_spv_file(spv_dir_ + "cast_f16_f32.spv", s3)) {
            h2f_pipe_ = new ncnn::Pipeline(vkdev);
            h2f_pipe_->set_local_size_xyz(128, 1, 1);
            if (h2f_pipe_->create(s3.data(), s3.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0) {
                fprintf(stderr, "[AwaLayer] FAIL create fp16->f32 pipeline\n");
                delete h2f_pipe_; h2f_pipe_ = nullptr;
            }
        }
        std::vector<uint32_t> s4;
        if (read_spv_file(spv_dir_ + "cast_f32_f16.spv", s4)) {
            f2h_pipe_ = new ncnn::Pipeline(vkdev);
            f2h_pipe_->set_local_size_xyz(128, 1, 1);
            if (f2h_pipe_->create(s4.data(), s4.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0) {
                fprintf(stderr, "[AwaLayer] FAIL create f32->fp16 pipeline\n");
                delete f2h_pipe_; f2h_pipe_ = nullptr;
            }
        }
    }
    return 0;
}

int AwaLayer::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    if (awa_pipe_) { delete awa_pipe_; awa_pipe_ = nullptr; }
    if (coal_pipe_) { delete coal_pipe_; coal_pipe_ = nullptr; }
    if (init_pipe_) { delete init_pipe_; init_pipe_ = nullptr; }
    if (b2f_pipe_) { delete b2f_pipe_; b2f_pipe_ = nullptr; }
    if (f2b_pipe_) { delete f2b_pipe_; f2b_pipe_ = nullptr; }
    if (h2f_pipe_) { delete h2f_pipe_; h2f_pipe_ = nullptr; }
    if (f2h_pipe_) { delete f2h_pipe_; f2h_pipe_ = nullptr; }
    return 0;
}

int AwaLayer::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    // fp16 模式下（use_fp16_storage=true）record_upload 会把 fp32 常量压成 half；
    // 自定义 shader 按 flat fp32 读，必须强制 fp32 上传。
    ncnn::Option o2 = opt;
    o2.use_fp16_storage = o2.use_fp16_packed = false;
    o2.use_bf16_storage = o2.use_bf16_packed = false;
    // 只上传 qk_norm 权重；窗口几何在首次 forward 按需上传（几何来源 = 宿主注入或文件 fallback）
    auto upload = [&](const std::vector<float>& src, ncnn::VkMat& dst) {
        if (src.empty()) return;
        // 坑：record_upload 对 1D Mat 会按 elemcount=w 自动 pack4（w/4>65535 时还静默失败）；
        // 用 2D Mat(w=n, h=1) 使 elemcount=h=1 -> dst_elempack=1，buffer 保持 flat fp32 线性布局。
        ncnn::Mat h;
        h.create((int)src.size(), 1, (size_t)4u, 1);
        memcpy(h.data, src.data(), src.size() * sizeof(float));
        cmd.record_upload(h, dst, o2);
    };
    upload(nq_v_,  vk_nqv);
    upload(nk_v_,  vk_nkv);
    upload(nq_t_,  vk_nqt);
    upload(nk_t_,  vk_nkt);
    fprintf(stderr, "[AwaLayer] upload_model done: nqv=%d nkv=%d nqt=%d nkt=%d (empty=%d%d%d%d)\n",
            (int)nq_v_.size(), (int)nk_v_.size(), (int)nq_t_.size(), (int)nk_t_.size(),
            (int)vk_nqv.empty(), (int)vk_nkv.empty(), (int)vk_nqt.empty(), (int)vk_nkt.empty());
    return 0;
}

int AwaLayer::forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                      std::vector<ncnn::VkMat>& top_blobs,
                      ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 2) { fprintf(stderr, "[AwaLayer] FAIL 需要 2 个输入\n"); return -1; }
    const ncnn::VkMat& vqkv = bottom_blobs[0];
    const ncnn::VkMat& tqkv = bottom_blobs[1];

    // 分辨率无关：从输入推断 Lv/TXT（覆盖 param 里的固定占位值）
    // 整图里 AWA 输入是上游 convert_packing 到 elempack=1 的 (w=QKV, h=Lv)；
    // 但阶段1 单测/其他路径可能传 ep4 —— 必须用 buffer_capacity()（真实字节数）/4 求元素数，
    // 不能用 total()（不含 elempack，ep4 时会低估 Lv -> shader 越界/输出 0）。
    {
        int in_esz = (opt.use_bf16_storage || opt.use_fp16_storage) ? 2 : 4;   // bf16/fp16 输入 2 字节/元素
        int Lv_in = (int)(vqkv.buffer_capacity() / in_esz / QKV);
        int TXT_in = (int)(tqkv.buffer_capacity() / in_esz / QKV);
        if (Lv_in <= 0 || TXT_in <= 0)
        {
            fprintf(stderr, "[AwaLayer] FAIL 输入形状异常 vqkv.cap=%zu tqkv.cap=%zu\n",
                    vqkv.buffer_capacity(), tqkv.buffer_capacity());
            return -1;
        }
        Lv_ = Lv_in;
        TXT_ = TXT_in;
    }

    // 窗口几何按需上传（首次 forward 时；几何来源 = 宿主 set_window_geometry 或 create_pipeline 的 fallback）
    if (vk_vidx.empty())
    {
        if (!geometry_ready())
        {
            fprintf(stderr, "[AwaLayer] FAIL 窗口几何未就绪（需 set_window_geometry 注入）\n");
            return -1;
        }
        // fp16 模式下强制 fp32 上传（flat-fp32 shader 约定）
        ncnn::Option o2 = opt;
        o2.use_fp16_storage = o2.use_fp16_packed = false;
        o2.use_bf16_storage = o2.use_bf16_packed = false;
        auto upload = [&](const std::vector<float>& src, ncnn::VkMat& dst) {
            ncnn::Mat h;
            h.create((int)src.size(), 1, (size_t)4u, 1);
            memcpy(h.data, src.data(), src.size() * sizeof(float));
            cmd.record_upload(h, dst, o2);
        };
        upload(vidx_,  vk_vidx);
        upload(cumf_,  vk_cumf);
        upload(vfreq_, vk_vfreq);
        upload(tfreq_, vk_tfreq);
        fprintf(stderr, "[AwaLayer] 窗口几何按需上传 OK (nwin=%d sumf=%d)\n", nwin_, sumf_);
    }

    // 输出（dims=2: w=DIM, h=Lv/TXT，供下游 InnerProduct 直接消费）
    ncnn::VkMat& vattn = top_blobs[0];
    ncnn::VkMat& tattn = top_blobs[1];
    // 低精度（bf16/fp16）模式：I/O 是 2 字节/元素 blob，层内 f32 中间计算 + 前后转换
    bool bf16 = (opt.use_bf16_storage && b2f_pipe_ && f2b_pipe_);
    bool f16  = (opt.use_fp16_storage && h2f_pipe_ && f2h_pipe_);
    bool low16 = bf16 || f16;
    ncnn::VkMat vqkv_f, tqkv_f, vattn_f, tattn_f, toutw_f;
    const ncnn::VkMat *pq = &vqkv, *pt = &tqkv;
    ncnn::VkMat *pv = &vattn, *ptt = &tattn, *ptw = &toutw_;
    auto cast_l2f = [&](const ncnn::VkMat& src, ncnn::VkMat& dst) {
        dst.create(src.w, src.h, 4u, opt.blob_vkallocator);
        std::vector<ncnn::VkMat> b(2); b[0] = src; b[1] = dst;
        std::vector<ncnn::vk_constant_type> c(1); c[0].i = src.w * src.h;
        ncnn::VkMat d; d.w = src.w * src.h; d.h = 1; d.c = 1;
        cmd.record_pipeline(bf16 ? b2f_pipe_ : h2f_pipe_, b, c, d);
    };
    auto cast_f2l = [&](const ncnn::VkMat& src, ncnn::VkMat& dst) {
        dst.create(src.w, src.h, 2u, opt.blob_vkallocator);
        std::vector<ncnn::VkMat> b(2); b[0] = src; b[1] = dst;
        std::vector<ncnn::vk_constant_type> c(1); c[0].i = src.w * src.h;
        ncnn::VkMat d; d.w = src.w * src.h; d.h = 1; d.c = 1;
        cmd.record_pipeline(bf16 ? f2b_pipe_ : f2h_pipe_, b, c, d);
    };
    if (low16) {
        cast_l2f(vqkv, vqkv_f);
        cast_l2f(tqkv, tqkv_f);
        vattn_f.create(DIM, Lv_, 4u, opt.blob_vkallocator);
        tattn_f.create(DIM, TXT_, 4u, opt.blob_vkallocator);
        toutw_f.create(DIM, nwin_ * TXT_, 4u, opt.blob_vkallocator);
        pq = &vqkv_f; pt = &tqkv_f; pv = &vattn_f; ptt = &tattn_f; ptw = &toutw_f;
    } else {
        vattn.create(DIM, Lv_, 4u, opt.blob_vkallocator);
        tattn.create(DIM, TXT_, 4u, opt.blob_vkallocator);
        toutw_.create(DIM, nwin_ * TXT_, 4u, opt.blob_vkallocator);
    }
    // 未覆盖 token（不在任何窗口内）无线程写入 -> 必须显式清零，否则脏数据
    // 注意：不能用 record_upload 清零（dims=2 时按 h%4 自动 pack4，会写坏 ep1 布局）
    {
        std::vector<ncnn::VkMat> ib(1);
        ib[0] = *pv;
        std::vector<ncnn::vk_constant_type> ic(1);
        ic[0].i = Lv_ * DIM;
        ncnn::VkMat idisp; idisp.w = Lv_ * DIM; idisp.h = 1; idisp.c = 1;  // 元素数语义
        cmd.record_pipeline(init_pipe_, ib, ic, idisp);
        ib[0] = *ptt;
        ic[0].i = TXT_ * DIM;
        ncnn::VkMat idisp2; idisp2.w = TXT_ * DIM; idisp2.h = 1; idisp2.c = 1;
        cmd.record_pipeline(init_pipe_, ib, ic, idisp2);
    }

    std::vector<ncnn::VkMat> bindings(12);
    bindings[0]  = *pq;
    bindings[1]  = *pt;
    bindings[2]  = vk_vidx;
    bindings[3]  = vk_cumf;
    bindings[4]  = vk_vfreq;
    bindings[5]  = vk_tfreq;
    bindings[6]  = vk_nqv;
    bindings[7]  = vk_nkv;
    bindings[8]  = vk_nqt;
    bindings[9]  = vk_nkt;
    bindings[10] = *pv;
    bindings[11] = *ptw;

    std::vector<ncnn::vk_constant_type> constants(10);
    constants[0].i = DIM;
    constants[1].i = HEAD_D;
    constants[2].i = HEADS;
    constants[3].i = ROPE_ROT;
    constants[4].i = nwin_;
    constants[5].i = TXT_;
    constants[6].i = Lv_;
    constants[7].i = sumf_;
    constants[8].f = SCALE;
    constants[9].f = EPS;

    ncnn::VkMat dispatcher;
    dispatcher.w = 1;
    dispatcher.h = HEADS;
    dispatcher.c = nwin_;
    cmd.record_pipeline(awa_pipe_, bindings, constants, dispatcher);

    // txt 跨窗口平均（coalesce）
    std::vector<ncnn::VkMat> cbindings(2);
    cbindings[0] = *ptw;
    cbindings[1] = *ptt;
    std::vector<ncnn::vk_constant_type> cconst(3);
    cconst[0].i = nwin_;
    cconst[1].i = TXT_;
    cconst[2].i = DIM;
    ncnn::VkMat cdisp;
    cdisp.w = TXT_ * DIM;   // dispatcher 语义 = 元素数；ncnn 内部再除 local_size(256) 得 workgroup 数
    cdisp.h = 1;
    cdisp.c = 1;
    cmd.record_pipeline(coal_pipe_, cbindings, cconst, cdisp);

    // 低精度：输出 f32 中间 -> bf16/fp16（top_blobs）
    if (low16) {
        cast_f2l(vattn_f, vattn);
        cast_f2l(tattn_f, tattn);
    }
    return 0;
}

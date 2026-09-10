// ada_compose.cpp — AdaCompose 自定义 ncnn::Layer 实现
// 把 ada 计算从 CPU + 386 次 upload 改为 GPU shader（1 次 emb upload + GPU 批量算）
#include "ada_compose.h"
#include <ncnn/gpu.h>
#include <ncnn/pipeline.h>
#include <ncnn/command.h>
#include <ncnn/paramdict.h>
#include <ncnn/modelbin.h>
#include <cstdio>
#include <cstring>
#include <fstream>

// ada 向量类型 -> emb 通道索引 k 的映射（与 compute_ada 一致）
//   0=a_sc -> k=1, 1=a_sh -> k=0, 2=a_g -> k=2
//   3=m_sc -> k=4, 4=m_sh -> k=3, 5=m_g -> k=5
static const int K_MAP[6] = {1, 0, 2, 4, 3, 5};

AdaComposeLayer::~AdaComposeLayer()
{
    destroy_pipeline(ncnn::Option());
}

int AdaComposeLayer::load_param(const ncnn::ParamDict& pd)
{
    spv_dir_  = pd.get(0, spv_dir_);
    nlayers_   = pd.get(1, 32);
    mm_layers_ = pd.get(2, 10);
    dim_       = pd.get(3, 2560);
    has_final_ = pd.get(4, 1);
    if (!spv_dir_.empty() && spv_dir_.back() != '/') spv_dir_ += '/';
    n_outputs_ = nlayers_ * 12 + has_final_ * 2;
    return 0;
}

int AdaComposeLayer::load_model(const ncnn::ModelBin& mb)
{
    // branch 权重：每层 2 流 × 6 向量 × DIM
    // 顺序与输出一致：[i*12 + flow*6 + k]
    //   k=0:a_sc, 1:a_sh, 2:a_g, 3:m_sc, 4:m_sh, 5:m_g
    // branch 存储顺序：attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate
    // 对应 k:            a_sh(1)    a_sc(0)    a_g(2)     m_sh(4)    m_sc(3)    m_g(5)
    // 但我们在 load_model 时按输出顺序(a_sc,a_sh,a_g,m_sc,m_sh,m_g)排列，
    // 所以读入时需重排：bin 顺序 = shift, scale, gate → 输出顺序 = scale(a_sc,k=1), shift(a_sh,k=0), gate(a_g,k=2)
    //                              mlp: shift, scale, gate → m_sc(k=4), m_sh(k=3), m_g(k=5)
    // 简化：直接按 K_MAP 读取——对每层每流读 6 个 DIM 向量，
    //   bin 顺序: attn_shift, attn_scale, attn_gate, mlp_shift, mlp_scale, mlp_gate
    //   输出顺序: a_sc(=attn_scale), a_sh(=attn_shift), a_g(=attn_gate),
    //             m_sc(=mlp_scale), m_sh(=mlp_shift), m_g(=mlp_gate)
    branch_host_.resize((size_t)n_outputs_ * dim_, 0.f);

    for (int i = 0; i < nlayers_; i++) {
        for (int flow = 0; flow < 2; flow++) {
            // bin 顺序: shift, scale, gate, shift, scale, gate
            ncnn::Mat m_shift = mb.load(dim_, 0);
            ncnn::Mat m_scale = mb.load(dim_, 0);
            ncnn::Mat m_gate  = mb.load(dim_, 0);
            ncnn::Mat m_msh   = mb.load(dim_, 0);
            ncnn::Mat m_msc   = mb.load(dim_, 0);
            ncnn::Mat m_mg    = mb.load(dim_, 0);
            if (m_shift.empty() || m_scale.empty() || m_gate.empty() ||
                m_msh.empty() || m_msc.empty() || m_mg.empty()) {
                fprintf(stderr, "[AdaCompose] load_model FAIL at layer %d flow %d\n", i, flow);
                return -1;
            }
            int base = (i * 12 + flow * 6) * dim_;
            // 输出顺序: a_sc, a_sh, a_g, m_sc, m_sh, m_g
            memcpy(&branch_host_[base + 0 * dim_], m_scale.data, dim_ * 4);  // a_sc = attn_scale
            memcpy(&branch_host_[base + 1 * dim_], m_shift.data, dim_ * 4);  // a_sh = attn_shift
            memcpy(&branch_host_[base + 2 * dim_], m_gate.data,  dim_ * 4);  // a_g  = attn_gate
            memcpy(&branch_host_[base + 3 * dim_], m_msc.data,   dim_ * 4);  // m_sc = mlp_scale
            memcpy(&branch_host_[base + 4 * dim_], m_msh.data,   dim_ * 4);  // m_sh = mlp_shift
            memcpy(&branch_host_[base + 5 * dim_], m_mg.data,    dim_ * 4);  // m_g  = mlp_gate
        }
    }
    // final: vid_out_ada_shift, vid_out_ada_scale
    if (has_final_) {
        ncnn::Mat m_fin_sh = mb.load(dim_, 0);  // vid_out_ada_shift
        ncnn::Mat m_fin_sc = mb.load(dim_, 0);  // vid_out_ada_scale
        if (m_fin_sh.empty() || m_fin_sc.empty()) {
            fprintf(stderr, "[AdaCompose] load_model FAIL: final weights missing\n");
            return -1;
        }
        int base = nlayers_ * 12 * dim_;
        memcpy(&branch_host_[base + 0 * dim_], m_fin_sc.data, dim_ * 4);  // fin_sc
        memcpy(&branch_host_[base + 1 * dim_], m_fin_sh.data, dim_ * 4);  // fin_sh
    }
    fprintf(stderr, "[AdaCompose] load_model OK: %d layers, %d outputs, %zu floats\n",
            nlayers_, n_outputs_, branch_host_.size());
    return 0;
}

bool AdaComposeLayer::read_spv_file(const std::string& path, std::vector<uint32_t>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[AdaCompose] 无 spv %s\n", path.c_str());
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    out.resize(n / 4);
    if (n) f.read((char*)out.data(), n);
    return !out.empty();
}

int AdaComposeLayer::create_pipeline(const ncnn::Option& opt)
{
    if (!opt.use_vulkan_compute) return 0;

    // spv 目录：与 AWA 层相同的约定（param 里的 spv_dir 或默认 models/m5/）
    std::vector<uint32_t> spv;
    if (!read_spv_file(spv_dir_ + "ada_compose.spv", spv)) return -1;
    pipe_ = new ncnn::Pipeline(vkdev);
    pipe_->set_local_size_xyz(128, 1, 1);
    if (pipe_->create(spv.data(), spv.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0) {
        fprintf(stderr, "[AdaCompose] FAIL create pipeline\n");
        return -1;
    }

    // 低精度图的下游 BinaryOp 按 bf16/fp16 读；使用专用 pair shader，
    // 每个线程独占写一个 uint，避免通用 cast shader 的 read-modify-write 竞争。
    std::vector<uint32_t> cspv;
    if (read_spv_file(spv_dir_ + "ada_compose_bf16.spv", cspv)) {
        pipe_bf16_ = new ncnn::Pipeline(vkdev);
        pipe_bf16_->set_local_size_xyz(128, 1, 1);
        if (pipe_bf16_->create(cspv.data(), cspv.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0) {
            delete pipe_bf16_; pipe_bf16_ = nullptr;
        }
    }
    cspv.clear();
    if (read_spv_file(spv_dir_ + "ada_compose_fp16.spv", cspv)) {
        pipe_fp16_ = new ncnn::Pipeline(vkdev);
        pipe_fp16_->set_local_size_xyz(128, 1, 1);
        if (pipe_fp16_->create(cspv.data(), cspv.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0) {
            delete pipe_fp16_; pipe_fp16_ = nullptr;
        }
    }
    return 0;
}

int AdaComposeLayer::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    if (pipe_) { delete pipe_; pipe_ = nullptr; }
    if (pipe_bf16_) { delete pipe_bf16_; pipe_bf16_ = nullptr; }
    if (pipe_fp16_) { delete pipe_fp16_; pipe_fp16_ = nullptr; }
    return 0;
}

int AdaComposeLayer::upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt)
{
    // 强制 fp32 上传（flat-fp32 shader 约定）
    ncnn::Option o2 = opt;
    o2.use_fp16_storage = o2.use_fp16_packed = false;
    o2.use_bf16_storage = o2.use_bf16_packed = false;

    ncnn::Mat h;
    h.create((int)branch_host_.size(), 1, (size_t)4u, 1);
    memcpy(h.data, branch_host_.data(), branch_host_.size() * sizeof(float));
    cmd.record_upload(h, vk_branch_, o2);
    fprintf(stderr, "[AdaCompose] upload_model OK: %zu floats -> GPU\n", branch_host_.size());
    return 0;
}

int AdaComposeLayer::forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                              std::vector<ncnn::VkMat>& top_blobs,
                              ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 1) {
        fprintf(stderr, "[AdaCompose] FAIL 需要 1 个输入(emb), got %zu\n", bottom_blobs.size());
        return -1;
    }
    const ncnn::VkMat& emb = bottom_blobs[0];
    if (top_blobs.size() != (size_t)n_outputs_) {
        fprintf(stderr, "[AdaCompose] FAIL 输出数 %zu != %d\n", top_blobs.size(), n_outputs_);
        return -1;
    }

    // 每个输出向量 dispatch 一次（同一 cmd，1 次 submit）
    for (int vi = 0; vi < n_outputs_; vi++) {
        int k, emb_stride, vec_offset;
        if (vi < nlayers_ * 12) {
            // ada 向量: emb[d*6+k], k 由 K_MAP 决定
            int ki = vi % 6;    // 0..5 对应 a_sc..m_g
            k = K_MAP[ki];
            emb_stride = 6;
        } else {
            // final: fin_sc/fin_sh 取 out 层(attn 组, l=0) 槽位 —— emb 6 槽布局下
            // fin_sc(scale) -> emb[d*6+1], fin_sh(shift) -> emb[d*6+0]。
            // [FIX 2026-09-10] 旧代码 emb_stride=3（仅当 3 槽布局才对；6 槽下 d*3 错位取到 MLP 组）。
            int fi = vi - nlayers_ * 12;
            k = (fi == 0) ? 1 : 0;   // fin_sc: k=1, fin_sh: k=0
            emb_stride = 6;
        }
        vec_offset = vi * dim_;

        ncnn::VkMat& out = top_blobs[vi];
        const bool bf16 = opt.use_bf16_storage && pipe_bf16_;
        const bool f16 = opt.use_fp16_storage && pipe_fp16_;
        out.create(dim_, 1, (bf16 || f16) ? 2u : 4u, opt.blob_vkallocator);
        // DIAG: 打印 vi 对应的 k 和 top_blobs[vi] 输出前几个值（验证 top_blob 顺序）
        if (getenv("SEEDVR_DIAG_ADAVI") && vi < 6) {
            ncnn::VkCompute dcmd(vkdev);
            ncnn::Mat dv;
            ncnn::Option do2 = opt; do2.use_packing_layout = false;
            dcmd.record_download(out, dv, do2);
            dcmd.submit_and_wait(); dcmd.reset();
            const unsigned short* hp16 = (const unsigned short*)dv.data;
            const float* hp32 = (const float*)dv.data;
            float v0=0,v1=0,v2=0;
            if (dv.elemsize==4){ v0=hp32[0]; v1=hp32[1]; v2=hp32[2]; }
            else { unsigned u0=(unsigned)hp16[0]<<16; unsigned u1=(unsigned)hp16[1]<<16; unsigned u2=(unsigned)hp16[2]<<16; memcpy(&v0,&u0,4); memcpy(&v1,&u1,4); memcpy(&v2,&u2,4); }
            fprintf(stderr, "[adavi] vi=%d k=%d vec_off=%d out[0:3]=%.4f %.4f %.4f\n", vi, k, vec_offset, v0, v1, v2);
        }

        std::vector<ncnn::VkMat> bindings(3);
        bindings[0] = emb;
        bindings[1] = vk_branch_;
        bindings[2] = out;

        std::vector<ncnn::vk_constant_type> constants(4);
        constants[0].i = k;
        constants[1].i = dim_;
        constants[2].i = vec_offset;
        constants[3].i = emb_stride;

        ncnn::VkMat dispatcher;
        // 低16位 shader 每线程写相邻两个元素，故 dispatcher 是 pair 数；fp32 是元素数。
        dispatcher.w = (bf16 || f16) ? ((dim_ + 1) / 2) : dim_;
        dispatcher.h = 1;
        dispatcher.c = 1;
        cmd.record_pipeline(bf16 ? pipe_bf16_ : f16 ? pipe_fp16_ : pipe_, bindings, constants, dispatcher);
    }
    return 0;
}

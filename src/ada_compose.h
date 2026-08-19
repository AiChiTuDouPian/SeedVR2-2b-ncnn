// ada_compose.h — AdaCompose 自定义 ncnn::Layer
//
// 把 ada 计算从 CPU + 384 次 upload 改为 GPU shader：
//   输入: emb (1×15360, 由 time_embedding 算出, 唯一需要 upload 的 Input)
//   权重: 所有层所有流的 branch 权重 (upload_model 一次性上传为 GPU buffer)
//   输出: nlayers*2*6 + 2(final) 个 VkMat, 每个 1×DIM=2560
//
// shader: out[d] = emb[d*6+k] + branch_buf[vec_offset+d]
//   k 由 ada 向量类型决定: 0=a_sc, 1=a_sh, 2=a_g, 3=m_sc, 4=m_sh, 5=m_g
//
// param 约定（与 AWA 层一致，0=spv_dir）:
//   0=(string) spv_dir  — 含 ada_compose.spv 的目录
//   1=(int) nlayers     — DiT 层数（4 或 32）
//   2=(int) mm_layers   — 双流层数（前 10 层 vid/txt 分支，后 22 层 all 分支）
//   3=(int) dim         — 向量维度 2560
//   4=(int) has_final   — 是否含 final ada（0=子图, 1=完整图末尾）
//
// bin 权重（load_model, 顺序）:
//   每层 i (0..nlayers-1):
//     流 0 (vid/all): attn_shift(DIM) attn_scale(DIM) attn_gate(DIM) mlp_shift(DIM) mlp_scale(DIM) mlp_gate(DIM)
//     流 1 (txt/all): 同上（后22层与流0相同权重，但图中仍需独立 blob）
//   若 has_final: vid_out_ada_shift(DIM) vid_out_ada_scale(DIM)
//
// 输出 blob 顺序（top_blobs 索引）:
//   for i in 0..nlayers-1:
//     for flow in 0..1:
//       for k in 0..5:  // a_sc, a_sh, a_g, m_sc, m_sh, m_g
//         top_blobs[i*12 + flow*6 + k]
//   若 has_final: top_blobs[nlayers*12 + 0] = fin_sc, [+1] = fin_sh
#pragma once

#include <ncnn/layer.h>
#include <ncnn/mat.h>
#include <string>
#include <vector>

namespace ncnn { class Pipeline; class VkMat; class VkTransfer; class VulkanDevice; }

class AdaComposeLayer : public ncnn::Layer
{
public:
    AdaComposeLayer() {
        one_blob_only = false;      // 1 输入(emb), 多输出(386+)
        support_inplace = false;
        support_vulkan = true;
        support_packing = false;    // flat float[] 线性布局
        // emb 由宿主强制按 fp32 上传；输出在层内转换为图的 bf16/fp16 存储。
        support_bf16_storage = true;
        support_fp16_storage = true;
    }
    virtual ~AdaComposeLayer();

    virtual int load_param(const ncnn::ParamDict& pd) override;
    virtual int load_model(const ncnn::ModelBin& mb) override;
    virtual int create_pipeline(const ncnn::Option& opt) override;
    virtual int destroy_pipeline(const ncnn::Option& opt) override;
    virtual int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override;
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                        std::vector<ncnn::VkMat>& top_blobs,
                        ncnn::VkCompute& cmd, const ncnn::Option& opt) const override;

    static ncnn::Layer* creator(void*) { return new AdaComposeLayer; }

private:
    int nlayers_ = 32;
    int mm_layers_ = 10;
    int dim_ = 2560;
    int has_final_ = 1;
    int n_outputs_ = 0;   // nlayers*12 + has_final*2

    // CPU 侧 branch 权重（连续存储，与输出顺序一致）
    // [i*12 + flow*6 + k][d]  →  branch_host_[(i*12 + flow*6 + k) * dim_ + d]
    std::vector<float> branch_host_;
    // GPU 侧
    ncnn::VkMat vk_branch_;
    ncnn::Pipeline* pipe_ = nullptr;       // fp32 输出
    ncnn::Pipeline* pipe_bf16_ = nullptr;  // 每线程打包一对 bf16 输出
    ncnn::Pipeline* pipe_fp16_ = nullptr;  // 每线程打包一对 fp16 输出

    static bool read_spv_file(const std::string& path, std::vector<uint32_t>& out);
    std::string spv_dir_ = "models/m5/";
};

// awa_layer.h — 阶段1：AWA 封装为自定义 ncnn::Layer（GPU 常驻，消除每层 2 upload + 2 download）
//
// 与 awa_vk.cpp（AwaVk 模块）的区别：
//   AwaVk 的 forward 接收/返回 CPU std::vector<float>，内部 upload + dispatch + download；
//   AwaLayer 的 forward 接收/返回 VkMat（已在 GPU），dispatch 后不 download，
//   上层（阶段3 整图）把 qkv 的 InnerProduct 输出直接接进来，AWA 输出直接接回 InnerProduct。
//
// 常量（窗口几何 vidx/cumf/freq + qk_norm 权重）在 create_pipeline/upload_model 阶段
// 一次性上传为 GPU buffer，不参与每层前向。
//
// param 约定（阶段2 整图 param 里 AWA 层使用）：
//   0=(string) spv 目录（含 awa.spv / awa_coalesce.spv / win_nonshifted.bin / win_shifted.bin）
//   1=(int)    窗口类型：0=nonshifted（偶数层） 1=shifted（奇数层）
//   2=(int)    Lv（视频 token 数）
//   3=(int)    TXT（文本 token 数）
// bin 权重（load_model，顺序）：nq_v, nk_v, nq_t, nk_t（各 128 个 fp32）
#pragma once

#include <ncnn/layer.h>
#include <ncnn/mat.h>
#include <string>
#include <vector>

namespace ncnn { class Pipeline; class VkMat; class VkTransfer; class VulkanDevice; }

class AwaLayer : public ncnn::Layer
{
public:
    AwaLayer() {
        one_blob_only = false;   // 双输入双输出
        support_inplace = false;
        support_vulkan = true;
        support_packing = false;   // flat float[] 线性布局（ncnn 自动把上游 pack4 转回 pack1）
        support_bf16_storage = true;   // 低精度模式：I/O 是 bf16 blob（层内自行转换）
        support_fp16_storage = true;   // fp16 模式：I/O 是 fp16 blob（层内自行转换，flat-float shader 不能直接读 half）
    }
    virtual ~AwaLayer();

    virtual int load_param(const ncnn::ParamDict& pd) override;
    virtual int load_model(const ncnn::ModelBin& mb) override;
    virtual int create_pipeline(const ncnn::Option& opt) override;
    virtual int destroy_pipeline(const ncnn::Option& opt) override;
    virtual int upload_model(ncnn::VkTransfer& cmd, const ncnn::Option& opt) override;
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                        std::vector<ncnn::VkMat>& top_blobs,
                        ncnn::VkCompute& cmd, const ncnn::Option& opt) const override;

    // ---- 供单测/直接构造（不走 param/bin 文件） ----
    void set_vulkan_device(const ncnn::VulkanDevice* d) { vkdev = d; }  // 手动创建时设置（Net 内自动赋值）
    void set_config(const std::string& spv_dir_, int win_type_, int Lv_, int TXT_);
    void set_qk_norm_host(const float* nqv, const float* nkv, const float* nqt, const float* nkt);
    // 注入窗口几何（测试用小合成窗口；不走磁盘 win bin）
    void set_window_geometry(const std::vector<float>& vidx, const std::vector<float>& cumf,
                             const std::vector<float>& vfreq, const std::vector<float>& tfreq,
                             int nwin, int Lv_, int TXT_);
    bool window_ready() const { return sumf_ > 0; }
    bool geometry_ready() const { return !vidx_.empty() && nwin_ > 0; }   // CPU 几何已就绪

    // ---- 测试用只读 ----
    int nwin() const { return nwin_; }
    int sumf() const { return sumf_; }
    int win_type() const { return win_type_; }
    const ncnn::VkMat& debug_toutw() const { return toutw_; }
    const ncnn::Pipeline* debug_awa_pipe() const { return awa_pipe_; }
    const ncnn::Pipeline* debug_coal_pipe() const { return coal_pipe_; }
    const ncnn::Pipeline* debug_init_pipe() const { return init_pipe_; }
    const ncnn::VkMat& debug_vidx() const { return vk_vidx; }
    const ncnn::VkMat& debug_cumf() const { return vk_cumf; }
    const ncnn::VkMat& debug_vfreq() const { return vk_vfreq; }
    const ncnn::VkMat& debug_tfreq() const { return vk_tfreq; }
    const ncnn::VkMat& debug_nqv() const { return vk_nqv; }
    const ncnn::VkMat& debug_nkv() const { return vk_nkv; }
    const ncnn::VkMat& debug_nqt() const { return vk_nqt; }
    const ncnn::VkMat& debug_nkt() const { return vk_nkt; }

    // 阶段3 整图注册用
    static ncnn::Layer* creator(void*) { return new AwaLayer; }

private:
    int build_window_geometry();   // 从 spv_dir 读 win_*.bin 构建 vidx/cumf/freq
    static bool read_spv_file(const std::string& path, std::vector<uint32_t>& out);

    std::string spv_dir_ = "models/m5/";
    int win_type_ = 0;    // 0=nonshifted, 1=shifted
    // 运行时从输入推断（整图里分辨率可变，param 里的 Lv 仅是占位）
    mutable int Lv_ = 0, TXT_ = 0;

    // CPU 侧常量
    std::vector<float> vidx_, cumf_, vfreq_, tfreq_;
    std::vector<float> nq_v_, nk_v_, nq_t_, nk_t_;
    int nwin_ = 0, sumf_ = 0;
    bool geom_injected_ = false;

    // GPU 侧（qk_norm 在 upload_model 上传；窗口几何在首次 forward 按需上传 -> mutable）
    mutable ncnn::VkMat vk_vidx, vk_cumf, vk_vfreq, vk_tfreq;
    ncnn::VkMat vk_nqv, vk_nkv, vk_nqt, vk_nkt;
    ncnn::Pipeline* awa_pipe_ = nullptr;
    ncnn::Pipeline* coal_pipe_ = nullptr;
    ncnn::Pipeline* init_pipe_ = nullptr;
    // 低精度（bf16/fp16）模式：AWA I/O 在层内转换（输入 低16->fp32 / 输出 fp32->低16）
    ncnn::Pipeline* b2f_pipe_ = nullptr;   // cast_bf16_f32.spv（bf16 -> fp32）
    ncnn::Pipeline* f2b_pipe_ = nullptr;   // cast_f32_bf16.spv（fp32 -> bf16）
    ncnn::Pipeline* h2f_pipe_ = nullptr;   // cast_f16_f32.spv（fp16 -> fp32）
    ncnn::Pipeline* f2h_pipe_ = nullptr;   // cast_f32_f16.spv（fp32 -> fp16）
    // 中间结果 nwin*TXT*DIM（awa -> coalesce 之间，须跨 submit 存活 -> 成员 mutable）
    mutable ncnn::VkMat toutw_;

    static const int HEADS = 20, HEAD_D = 128, DIM = 2560, QKV = HEADS * HEAD_D * 3;
    static const int ROPE_ROT = 126;
    static const float SCALE, EPS;
};

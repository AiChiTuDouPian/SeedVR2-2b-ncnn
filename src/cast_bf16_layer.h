// cast_bf16_layer.h — 自定义 bf16 <-> fp32 Cast 层（ncnn Cast_vulkan 不支持 bf16，TODO）
// 0=模式：0=bf16->fp32  1=fp32->bf16
#pragma once
#include <ncnn/layer.h>
#include <ncnn/pipeline.h>
#include <string>
#include <vector>

class Bf16CastLayer : public ncnn::Layer
{
public:
    Bf16CastLayer() { one_blob_only = true; support_inplace = false; support_vulkan = true; support_packing = false; }
    virtual int load_param(const ncnn::ParamDict& pd) override { mode_ = pd.get(0, 0); return 0; }
    virtual int create_pipeline(const ncnn::Option& opt) override;
    virtual int destroy_pipeline(const ncnn::Option& opt) override;
    virtual int upload_model(ncnn::VkTransfer&, const ncnn::Option&) override { return 0; }
    virtual int forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                        std::vector<ncnn::VkMat>& top_blobs,
                        ncnn::VkCompute& cmd, const ncnn::Option& opt) const override;

    int mode_ = 0;   // 0=bf16->fp32 1=fp32->bf16
    ncnn::VulkanDevice* vkdev = nullptr;
    ncnn::Pipeline* pipe_ = nullptr;
    std::string spv_dir_ = "models/m5/";

    static ncnn::Layer* creator(void*) { return new Bf16CastLayer; }
};

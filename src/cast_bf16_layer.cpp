// cast_bf16_layer.cpp — 自定义 bf16 <-> fp32 Cast 层实现
#include "cast_bf16_layer.h"
#include <ncnn/gpu.h>
#include <ncnn/command.h>
#include <ncnn/gpu.h>
#include <cstdio>
#include <fstream>

static bool read_spv_file(const std::string& path, std::vector<uint32_t>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[Bf16Cast] 无 spv %s\n", path.c_str()); return false; }
    f.seekg(0, std::ios::end);
    size_t n = (size_t)f.tellg();
    f.seekg(0);
    out.resize(n / 4);
    if (n) f.read((char*)out.data(), n);
    return !out.empty();
}

int Bf16CastLayer::create_pipeline(const ncnn::Option& /*opt*/)
{
    if (!vkdev) vkdev = ncnn::get_gpu_device(0);
    if (!vkdev) return -1;
    std::string spv = spv_dir_ + (mode_ == 0 ? "cast_bf16_f32.spv" : "cast_f32_bf16.spv");
    std::vector<uint32_t> s;
    if (!read_spv_file(spv, s)) return -1;
    pipe_ = new ncnn::Pipeline(vkdev);
    pipe_->set_local_size_xyz(128, 1, 1);
    if (pipe_->create(s.data(), s.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0)
    {
        fprintf(stderr, "[Bf16Cast] FAIL create pipeline\n");
        return -1;
    }
    return 0;
}

int Bf16CastLayer::destroy_pipeline(const ncnn::Option& /*opt*/)
{
    if (pipe_) { delete pipe_; pipe_ = nullptr; }
    return 0;
}

int Bf16CastLayer::forward(const std::vector<ncnn::VkMat>& bottom_blobs,
                           std::vector<ncnn::VkMat>& top_blobs,
                           ncnn::VkCompute& cmd, const ncnn::Option& opt) const
{
    if (bottom_blobs.size() != 1) { fprintf(stderr, "[Bf16Cast] FAIL 输入数=%zu\n", bottom_blobs.size()); return -1; }
    const ncnn::VkMat& b = bottom_blobs[0];
    ncnn::VkMat& t = top_blobs[0];
    // 保持 2D 形状（w/h 同输入），只改元素字节数；下游 InnerProduct 期望 2D
    size_t total = (size_t)b.w * b.h;
    if (mode_ == 0)
        t.create(b.w, b.h, 4u, opt.blob_vkallocator);            // bf16 -> fp32（4B/元素）
    else
        t.create(b.w, b.h, 2u, opt.blob_vkallocator);            // fp32 -> bf16（2B/元素）
    if (t.empty()) { fprintf(stderr, "[Bf16Cast] FAIL create w=%d h=%d mode=%d total=%zu\n", b.w, b.h, mode_, total); return -1; }

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0] = b;
    bindings[1] = t;
    std::vector<ncnn::vk_constant_type> constants(1);
    constants[0].i = (int)total;
    ncnn::VkMat dispatcher;
    dispatcher.w = (int)total;   // 元素数语义（ncnn 自动 ceil / local_size）
    dispatcher.h = 1;
    dispatcher.c = 1;
    cmd.record_pipeline(pipe_, bindings, constants, dispatcher);
    return 0;
}

// awa_vk.h — SeedVR2 自适应窗口注意力（Adaptive Window Attention）自定义 Vulkan 模块
// 把 dit_vk.cpp::awa_forward 中“窗口 partition + qk_norm + mmrope + varlen SDPA +
// unpartition + 文本 coalesce”的注意力重计算整体搬上 GPU（一个 compute shader）。
// 接口：输入 vid/txt 的 qkv(L,3*DIM) + 窗口几何 + RoPE freq + qk_norm 权重，
//       输出 vid_attn(L,DIM) 与 txt_attn(TXT,DIM)。
#pragma once
#include <string>
#include <vector>

struct Win;   // 前向声明；完整定义在 dit_vk.h

namespace ncnn { class VulkanDevice; class Pipeline; class VkAllocator; }

class AwaVk {
public:
    AwaVk();
    ~AwaVk();

    // vkdev: 已初始化的 ncnn Vulkan 设备；spv_path: awa.comp 编译产物 awa.spv 路径
    bool init(ncnn::VulkanDevice* vkdev, const std::string& spv_path);

    bool ready() const { return pipe != nullptr; }

    // 显式释放 pipeline（销毁 GPU 设备前必须调用，否则悬空引用 -> SIGSEGV）
    void release() {
        if (pipe) { delete pipe; pipe = nullptr; }
        if (init_pipe) { delete init_pipe; init_pipe = nullptr; }
    }

    // 复用 DitVk 缓存的 blob/staging allocator（避免每次 acquire 从池里取新实例导致显存泄漏）。
    // 必须在 init 之后、forward 之前调用。
    void set_allocators(ncnn::VkAllocator* blob, ncnn::VkAllocator* staging) {
        blob_alloc = blob; staging_alloc = staging;
    }

    // vqkv: Lv*3*DIM, tqkv: TXT*3*DIM (fp32, 行序 q|k|v 各 DIM)
    // nq_*/nk_*: 各 128（per-head-dim qk_norm 权重，vid/txt 各一套）
    // 输出 vid_out(Lv*DIM) / txt_out(TXT*DIM)，已含文本跨窗口平均
    bool forward(const std::vector<float>& vqkv, const std::vector<float>& tqkv,
                 const Win& win,
                 const std::vector<float>& nq_v, const std::vector<float>& nk_v,
                 const std::vector<float>& nq_t, const std::vector<float>& nk_t,
                 int Lv, int TXT,
                 std::vector<float>& vid_out, std::vector<float>& txt_out);

private:
    ncnn::VulkanDevice* vkdev = nullptr;
    ncnn::Pipeline*     pipe  = nullptr;
    ncnn::Pipeline*     init_pipe = nullptr;   // awa_init.spv：vattn 未覆盖 token 清零
    ncnn::VkAllocator*  blob_alloc    = nullptr;   // 复用 DitVk 缓存实例
    ncnn::VkAllocator*  staging_alloc = nullptr;
};

// dit_vk.h — SeedVR2 NaDiT 的 Vulkan 加速推理引擎（接口）
// 所有 GEMM（qkv / proj_out / mlp / top 模块）走 ncnn Vulkan Net 在 GPU 上计算；
// RMSNorm / ada 调制 / qk_norm / 3D-RoPE / 窗口注意力 仍在 CPU（轻量）。
#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <list>
#include <cstdint>
#include "awa_vk.h"

namespace ncnn { class Net; class VulkanDevice; }

// ---------- 窗口几何（与 m5_export_windows.py 同格式） ----------
struct Win {
    int t = 0, h = 0, w = 0, nwin = 0;
    int txt_len = 8;            // 文本 token 数（真实模型 = 58，M5 测试 = 8）
    std::vector<int> st, en, sh, eh, sw, ew;
    std::vector<float> vid_freq, txt_freq;
};

class DitVk {
public:
    // CPU 侧分支权重（norm / ada）
    struct RawBranch {
        std::vector<float> nq, nk, attn_shift, attn_scale, attn_gate,
                           mlp_shift, mlp_scale, mlp_gate;
    };

    DitVk();
    ~DitVk();

    // model_dir: 含 *.param/*.bin 的目录（如 models/m5/）
    // fp16_arith: true=GPU 用 fp16 计算（快，略损精度）；false=GPU fp32 计算（慢，贴近 ref）
    // precision: GEMM 存储精度 0=fp32(默认,逐位对齐) 1=fp16 2=bf16
    bool init(const std::string& model_dir, bool fp16_arith = false, int precision = 0);

    // 核心前向（输入已是 patchify 后的 token）
    //   vid_patch : (Lv, 132) 展开，33ch patchified 输入
    //   txt       : (TXT, 5120) 展开
    //   timestep  : 标量（SeedVR2 单步蒸馏固定 1000.0）
    //   out_sr    : (Lv, 64) 展开（= vid_out.proj 输出）
    bool forward(const std::vector<float>& vid_patch, int Lv,
                 const std::vector<float>& txt, int TXT,
                 float timestep, std::vector<float>& out_sr);

    // 端到端前向（输入完整网格，内部 patchify / unpatchify）
    //   vid_grid  : (T, H, W, 33) 展开（noise(16)+cond(16)+mask(1) 已拼好）
    //   sr_latent : (T, H, W, 16) 展开
    bool forward_grid(const std::vector<float>& vid_grid, int T, int H, int W,
                      const std::vector<float>& txt, int TXT, float timestep,
                      std::vector<float>& sr_latent);

    // 设置窗口（由外部按实际 grid 生成，格式同 load_win）
    void set_windows(const Win& ns, const Win& sh) { win_ns = ns; win_sh = sh; }

    // 自定义 Vulkan AWA 模块是否就绪（spv 加载成功）
    bool awa_ready() const { return awa.ready(); }

    // 每层末调用：blob/staging allocator 的 free-list 只增不减，clear() 把已释放的
    // Vulkan 显存真正还给驱动，避免 32 层累计资源耗尽。仅在当层所有 Net 已析构、
    // AwaVk 的 VkMat 已出作用域后调用（无悬空引用）。
    void reclaim_vram();

    // 块间重置：销毁并重建 Vulkan 设备实例（全新 VkDevice），彻底清空 ncnn 在设备级
    // 累积却只在 device 析构时才释放的资源（命令池/描述符池/pipeline）。CPU 侧权重与
    // fcache 字节保留在内存，不重读 40GB 模型；下一次 get_net 从 fcache 重新上传到新设备。
    // 用于分块推理：每 K 层调一次，绕开深层累积导致的非确定性崩溃。
    void reset_vulkan_device();

    // 网格 <-> token 展开（供分块推理在引擎外部做 patchify/unpatchify）
    static std::vector<float> patchify_grid(const std::vector<float>& vid_grid, int T, int H, int W);
    static std::vector<float> unpatchify_latent(const std::vector<float>& sr_latent, int T, int H, int W);

    // 分块前向：在 [l0,l1) 层区间上跑 DiT。
    //   do_init=true 时先 vid_in_proj/txt_in/time_embedding（首块）；
    //   do_final=true 时跑末尾 norm + vid_out_proj 并写出 sr_out（末块）；
    //   否则仅更新 latent，经 vid_out_latent/txt_out_latent 传出供下一块续算。
    // vid_patch: (Lv,132)；txt: (TXT,5120)；返回 vid/txt 残差相加后的 latent 或最终 sr。
    bool forward_latent(const std::vector<float>& vid_patch, int Lv,
                        const std::vector<float>& txt, int TXT, float timestep,
                        int l0, int l1, bool do_init, bool do_final,
                        std::vector<float>& vid_out_latent, std::vector<float>& txt_out_latent,
                        std::vector<float>& sr_out);

    // 单层 AWA（debug 用，原 private）：GPU(AwaVk) 或 CPU 参考，取决于 awa.ready()
    void awa_forward(int i, const std::vector<float>& vid, const std::vector<float>& txt,
                     const Win& win, const RawBranch& vR, const RawBranch& tR,
                     std::vector<float>& vid_out, std::vector<float>& txt_out);
    // 供 debug 脚本生成 vqkv/tqkv（与 awa_forward 内部一致的 qkv 投影）
    std::vector<float> lin(const std::string& base, const float* x, int Ln,
                           int in_dim, int out_dim);

private:
    bool cache_file(const std::string& base);                 // 读 param+bin 进内存
    // 单次 Linear（Vulkan）：通过 get_net 取/建按 base 缓存的 ncnn::Net（LRU 逐出，见 net_cache），
    // 避免每层 new/delete 数百个 Net 造成 Vulkan 资源 churn。权重字节缓存在 fcache。
    // 核心 32 层前向（in/out 均为 (N, DIM) 展开）
    bool core_forward(std::vector<float>& vid, int Lv,
                      std::vector<float>& txt, int TXT, float timestep,
                      std::vector<float>& out_sr);
    std::vector<float> time_embedding(float t);
    std::vector<float> swiglu(const std::string& b, const float* x, int Ln);

    std::string dir;
    ncnn::VulkanDevice* vkdev = nullptr;
    bool fp16_arith = false;
    int precision_ = 0;   // GEMM 存储精度 0=fp32 1=fp16 2=bf16

    // 全局只 acquire 一次的 blob/staging allocator，所有 ncnn::Net 与 AwaVk 复用同一实例，
    // 避免 acquire_blob_allocator() 池“取走不归还”导致每层新增一个 VkBlobAllocator 而显存泄漏。
    ncnn::VkAllocator* blob_alloc = nullptr;
    ncnn::VkAllocator* staging_alloc = nullptr;

    // 文件内存缓存：base -> (param文本, model字节)
    std::map<std::string, std::pair<std::string, std::vector<unsigned char>>> fcache;

    // ncnn::Net 按 base 持久缓存 + LRU 逐出：避免每层 new/delete 数百个 Net 造成的
    // Vulkan 描述符/管线资源 churn（32 层 ×7 ≈ 224 次创建销毁会在 ~layer20 耗尽某类资源）。
    // 上限 16 个 ≈ 2-3 层权重（fp16 ≈ 1-2GB），可驻留 16GB 显存；超出则淘汰最久未用者（其析构释放权重显存）。
    static const size_t NET_CACHE_CAP = 16;
    std::map<std::string, std::unique_ptr<ncnn::Net>> net_cache;
    std::list<std::string> net_lru;
    ncnn::Net* get_net(const std::string& base);   // 取/建缓存 Net（带 LRU 淘汰）
    void evict_nets();

    // CPU 侧权重（norm / ada）
    std::vector<float> Wvon, Wvoa_s, Wvoa_sc;
    std::vector<RawBranch> bvid_raw, btxt_raw;

    Win win_ns, win_sh;

    AwaVk awa;   // 自定义 Vulkan AWA 模块（窗口注意力 GPU 加速）
};

// ---------- 窗口文件加载（共享给测试/运行） ----------
// txt_len：文本 token 数，用于把 freq 缓冲正确切分为 vid_freq / txt_freq
//          （真实模型 = 58，M5 测试 = 8）。默认 8 兼容旧测试。
Win load_win(const char* path, int txt_len = 8);

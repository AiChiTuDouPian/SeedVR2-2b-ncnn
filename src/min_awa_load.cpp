// min_awa_load.cpp — 最小链路测试：Input + AWA 自定义层（Net 内加载 + 注入几何 + forward）
// 验证：整图里 AWA 层能正常 load_model / create_pipeline / forward（几何由宿主注入）
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/mat.h>
#include <ncnn/command.h>
#include "awa_layer.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

static const int DIM = 2560, QKV = 7680, ROPE_ROT = 126;
static unsigned int g_seed = 55;
static float frand() { g_seed = g_seed * 1664525u + 1013904223u; return ((g_seed >> 8) & 0xFFFFFF) / (float)0x1000000; }

int main(int argc, char** argv)
{
    std::string graph_dir = argc > 1 ? argv[1] : "models/m5_graph/";
    bool ok = false;
    {
    if (ncnn::create_gpu_instance() != 0) { printf("FAIL gpu\n"); return 1; }
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(0);
    ncnn::VkAllocator* blob = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging = vkdev->acquire_staging_allocator();

    ncnn::Net net;
    net.set_vulkan_device(vkdev);
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_storage = net.opt.use_fp16_packed = false;
    net.opt.use_bf16_storage = net.opt.use_bf16_packed = false;
    net.opt.blob_vkallocator = blob;
    net.opt.workspace_vkallocator = blob;
    net.opt.staging_vkallocator = staging;
    net.register_custom_layer("AWA", AwaLayer::creator);
    std::string pf = argc > 2 ? argv[2] : "min.param";
    if (net.load_param((graph_dir + pf).c_str()) != 0) { printf("FAIL load_param %s\n", pf.c_str()); return 1; }
    printf("[min] load_param OK (%s)\n", pf.c_str());
    if (argc > 2)
    {
        int cnt = 0;
        for (auto* l : net.mutable_layers()) { if (cnt < 40) printf("  layer[%d] = %s\n", cnt, l->name.c_str()); cnt++; }
        printf("  ... total %d layers\n", cnt);
        ncnn::destroy_gpu_instance();
        printf("[min] PARAM_ONLY_OK\n");
        return 0;
    }
    FILE* f = fopen((graph_dir + "min.bin").c_str(), "rb");
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> bin(n); fread(bin.data(), 1, n, f); fclose(f);
    int lr = net.load_model(bin.data());
    printf("[min] load_model ret=%d\n", lr);
    if (lr < 0) { printf("FAIL load_model\n"); return 1; }
    printf("[min] load_param + load_model OK\n");

    // ---- 注入窗口几何（合成小窗口 nwin=2, f_i=32, Lv=64, TXT=58）----
    AwaLayer* awa = nullptr;
    for (auto* l : net.mutable_layers())
        if (l->type == "AWA") { awa = (AwaLayer*)l; break; }
    if (!awa) { printf("FAIL 未找到 AWA 层\n"); return 1; }
    {
        const int nwin = 2, Lv = 64, TXT = 58;
        std::vector<float> vidx, cumf(nwin + 1, 0.f);
        int sumf = 0;
        // 8x8 网格，窗口 [0,4)x[0,4)x[0,2) 和 [0,4)x[0,4)x[2,4)
        for (int wi = 0; wi < nwin; wi++) {
            int st = 0, en = 4, sh = 0, eh = 4, sw = (wi == 0 ? 0 : 2), ew = (wi == 0 ? 2 : 4);
            int f_i = (en - st) * (eh - sh) * (ew - sw);
            cumf[wi + 1] = cumf[wi] + f_i; sumf += f_i;
            for (int lt = st; lt < en; lt++) for (int lh = sh; lh < eh; lh++) for (int lw = sw; lw < ew; lw++)
                vidx.push_back((float)(((lt * 8) + lh) * 8 + lw));
        }
        std::vector<float> vfreq((size_t)sumf * ROPE_ROT), tfreq((size_t)nwin * TXT * ROPE_ROT);
        for (auto& v : vfreq) v = frand() * 6.2831853f;
        for (auto& v : tfreq) v = frand() * 6.2831853f;
        awa->set_window_geometry(vidx, cumf, vfreq, tfreq, nwin, Lv, TXT);
        printf("[min] 几何注入 OK (sumf=%d)\n", sumf);
    }

    // ---- forward：in0=vqkv(Lv*QKV), in1=tqkv(TXT*QKV) ----
    const int Lv = 64, TXT = 58;
    std::vector<float> vqkv((size_t)Lv * QKV), tqkv((size_t)TXT * QKV);
    for (auto& v : vqkv) v = frand() * 2.f - 1.f;
    for (auto& v : tqkv) v = frand() * 2.f - 1.f;

    ncnn::VkCompute cmd(vkdev);
    ncnn::Extractor ex = net.create_extractor();
    auto up = [&](const std::vector<float>& src, const char* name, int w, int h) {
        ncnn::Mat m; m.create(w, h, (size_t)4u, 1);
        memcpy(m.data, src.data(), src.size() * sizeof(float));
        ncnn::VkMat vk; cmd.record_upload(m, vk, net.opt);
        ex.input(name, vk);
    };
    up(vqkv, "in0", QKV, Lv);
    up(tqkv, "in1", QKV, TXT);
    ncnn::VkMat vk_out0, vk_out1;
    ex.extract("out0", vk_out0, cmd);
    ex.extract("out1", vk_out1, cmd);
    ncnn::VkMat o0p, o1p;
    vkdev->convert_packing(vk_out0, o0p, 1, cmd, net.opt);
    vkdev->convert_packing(vk_out1, o1p, 1, cmd, net.opt);
    ncnn::Mat m0, m1;
    { ncnn::Option od = net.opt; od.use_packing_layout = false; cmd.record_download(o0p, m0, od); cmd.record_download(o1p, m1, od); }
    cmd.submit_and_wait();
    cmd.reset();
    printf("[min] forward OK: out0=(%d,%d,ep%d) out1=(%d,%d,ep%d)\n",
           m0.w, m0.h, m0.elempack, m1.w, m1.h, m1.elempack);
    printf("[min] out0[0..3] = %.4f %.4f %.4f %.4f\n", ((const float*)m0.data)[0], ((const float*)m0.data)[1],
           ((const float*)m0.data)[2], ((const float*)m0.data)[3]);

    ok = true;
    }   // 作用域结束：net/cmd 先析构，GPU 实例仍活
    ncnn::destroy_gpu_instance();
    printf("[min] %s\n", ok ? "ALL OK" : "FAIL");
    return ok ? 0 : 1;
}

// probe_vkmat_io.cpp — 阶段0 mini 图探针：验证 ncnn Vulkan 的 VkMat 连续执行链路
//
// 目标（阶段2/3 的地基）：
//   1. input(VkMat) / extract(VkMat, VkCompute&) 数值正确（vs CPU Mat 版 cos=1.0）
//   2. 同一 VkCompute 连续多次 extract（层间不 download/upload）数值正确
//   3. 加速比：VkMat 直通（无往返） vs 现有 lin() 式每次 upload+download
//
// 用法：seedvr2_probe_vkmat_io [model_dir] [base] [Ln] [in_dim] [out_dim] [iters]
//   默认：models/m5/ b0_vid_qkv 64 2560 7680 20
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/mat.h>
#include <ncnn/command.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>

static unsigned int g_seed = 777;
static float frand() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return ((g_seed >> 8) & 0xFFFFFF) / (float)0x1000000;
}

static double cos_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); i++) { dot += (double)a[i] * b[i]; na += (double)a[i] * a[i]; nb += (double)b[i] * b[i]; }
    return dot / sqrt(na * nb);
}

static std::string read_text(const std::string& p) { std::ifstream f(p, std::ios::binary); std::stringstream ss; ss << f.rdbuf(); return ss.str(); }
static std::vector<unsigned char> read_bin(const std::string& p) {
    std::ifstream f(p, std::ios::binary); f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg(); f.seekg(0);
    std::vector<unsigned char> b(n); f.read((char*)b.data(), n); return b;
}

int main(int argc, char** argv)
{
    std::string model_dir = argc > 1 ? argv[1] : "models/m5/";
    std::string base      = argc > 2 ? argv[2] : "b0_vid_out";
    int Ln = argc > 3 ? atoi(argv[3]) : 64;
    int in_dim = argc > 4 ? atoi(argv[4]) : 2560;
    int out_dim = argc > 5 ? atoi(argv[5]) : 7680;
    int iters = argc > 6 ? atoi(argv[6]) : 20;
    if (model_dir.empty() || model_dir.back() != '/') model_dir += '/';

    bool ok = false;
    {
    if (ncnn::create_gpu_instance() != 0) { fprintf(stderr, "FAIL create_gpu_instance\n"); return 1; }
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(0);
    ncnn::VkAllocator* blob_alloc = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_alloc = vkdev->acquire_staging_allocator();

    // ---- 加载单层 Net（qkv InnerProduct 等）----
    ncnn::Net net;
    net.set_vulkan_device(vkdev);
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_storage = net.opt.use_fp16_packed = false;
    net.opt.use_bf16_storage = net.opt.use_bf16_packed = false;
    net.opt.blob_vkallocator = blob_alloc;
    net.opt.workspace_vkallocator = blob_alloc;   // VkMat 版 extract 没有兜底，必须自己设（net.cpp 仅 CPU extract 兜底）
    net.opt.staging_vkallocator = staging_alloc;
    net.load_param_mem(read_text(model_dir + base + ".param").c_str());
    std::vector<unsigned char> bin = read_bin(model_dir + base + ".bin");
    net.load_model(bin.data());
    fprintf(stderr, "[probe] loaded %s (%d -> %d, Ln=%d)\n", base.c_str(), in_dim, out_dim, Ln);

    // ---- 随机输入 ----
    std::vector<float> x((size_t)Ln * in_dim);
    for (auto& v : x) v = frand() * 2.f - 1.f;

    // ---- CPU 参考（等价现有 lin()：input(Mat)->extract(Mat)）----
    fprintf(stderr, "[probe] step cpu-ref...\n"); fflush(stderr);
    std::vector<float> y_cpu;
    {
        ncnn::Mat in(in_dim, Ln, (void*)x.data());
        ncnn::Mat out;
        {
            ncnn::Extractor ex = net.create_extractor();
            ex.input("in0", in);
            ex.extract("out0", out);
        }
        if (out.w == out_dim && out.h == Ln && out.c == 1) {
            y_cpu.assign((const float*)out.data, (const float*)out.data + (size_t)Ln * out_dim);
        } else if (out.w == Ln && out.h == out_dim) {
            y_cpu.resize((size_t)Ln * out_dim);
            const float* od = out;
            for (int t = 0; t < Ln; t++) for (int o = 0; o < out_dim; o++) y_cpu[t * out_dim + o] = od[o * Ln + t];
        } else {
            fprintf(stderr, "FAIL CPU out dims w=%d h=%d c=%d\n", out.w, out.h, out.c);
            return 1;
        }
    }

    // ---- VkMat 版：upload 一次 -> input(VkMat) -> extract(VkMat, cmd) -> download 一次 ----
    fprintf(stderr, "[probe] step vkmat...\n"); fflush(stderr);
    ncnn::Mat in_mat(in_dim, Ln, (void*)x.data());
    ncnn::VkCompute cmd(vkdev);
    ncnn::VkMat vk_in;
    cmd.record_upload(in_mat, vk_in, net.opt);
    fprintf(stderr, "[probe] step extract...\n"); fflush(stderr);
    ncnn::VkMat vk_out;
    {
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", vk_in);
        ex.extract("out0", vk_out, cmd);
    }
    fprintf(stderr, "[probe] step download+submit...\n"); fflush(stderr);
    ncnn::Mat m_gpu;
    {
        fprintf(stderr, "  [dbg] vk_in: w=%d h=%d d=%d ep=%d | vk_out: w=%d h=%d d=%d ep=%d\n",
                vk_in.w, vk_in.h, vk_in.dims, vk_in.elempack, vk_out.w, vk_out.h, vk_out.dims, vk_out.elempack);
        // 输出是 pack4（InnerProduct 默认）；显式转 pack1 再下载，得到 (out_dim, Ln) 行序
        ncnn::VkMat vk_out_p1;
        vkdev->convert_packing(vk_out, vk_out_p1, 1, cmd, net.opt);
        fprintf(stderr, "  [dbg] vk_out_p1: w=%d h=%d d=%d ep=%d\n",
                vk_out_p1.w, vk_out_p1.h, vk_out_p1.dims, vk_out_p1.elempack);
        {
        ncnn::Option opt_dl = net.opt; opt_dl.use_packing_layout = false;   // download 用 pack1（否则 use_packing_layout 会重新 pack4）
        cmd.record_download(vk_out_p1, m_gpu, opt_dl);
    }
    }
    cmd.submit_and_wait();
    cmd.reset();
    fprintf(stderr, "[probe] step unpack...\n"); fflush(stderr);
    std::vector<float> y_gpu;
    {
        fprintf(stderr, "  [dbg] m_gpu: w=%d h=%d c=%d d=%d elempack=%d elemsize=%zu\n",
                m_gpu.w, m_gpu.h, m_gpu.c, m_gpu.dims, m_gpu.elempack, m_gpu.elemsize);
        const float* gp = (const float*)m_gpu.data;
        fprintf(stderr, "  [dbg] gpu[0..7] = %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f | cpu[0..7] = %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                gp[0], gp[1], gp[2], gp[3], gp[4], gp[5], gp[6], gp[7],
                y_cpu[0], y_cpu[1], y_cpu[2], y_cpu[3], y_cpu[4], y_cpu[5], y_cpu[6], y_cpu[7]);
        if (m_gpu.elempack == 1) {
            if (m_gpu.w == out_dim && m_gpu.h == Ln) y_gpu.assign((const float*)m_gpu.data, (const float*)m_gpu.data + (size_t)Ln * out_dim);
            else if (m_gpu.w == Ln && m_gpu.h == out_dim) {
                y_gpu.resize((size_t)Ln * out_dim);
                for (int t = 0; t < Ln; t++) for (int o = 0; o < out_dim; o++) y_gpu[t * out_dim + o] = ((const float*)m_gpu.data)[o * Ln + t];
            } else { fprintf(stderr, "FAIL gpu dims w=%d h=%d c=%d\n", m_gpu.w, m_gpu.h, m_gpu.c); return 1; }
        } else if (m_gpu.elempack == 4) {
            // pack4: (w=out_dim/4, h=Ln) -> 展开行序
            int w4 = m_gpu.w, h4 = m_gpu.h;
            if (w4 * 4 == out_dim && h4 == Ln) {
                y_gpu.resize((size_t)Ln * out_dim);
                const float* d = (const float*)m_gpu.data;
                for (int t = 0; t < Ln; t++) for (int w = 0; w < w4; w++) for (int k = 0; k < 4; k++)
                    y_gpu[t * out_dim + w * 4 + k] = d[(t * w4 + w) * 4 + k];
            } else { fprintf(stderr, "FAIL gpu pack4 dims w=%d h=%d\n", m_gpu.w, m_gpu.h); return 1; }
        } else { fprintf(stderr, "FAIL gpu elempack=%d\n", m_gpu.elempack); return 1; }
    }

    double c1 = cos_sim(y_cpu, y_gpu);
    fprintf(stderr, "\n========== 阶段0 mini 图探针 ==========\n");
    fprintf(stderr, "  [1] 单次 VkMat I/O vs CPU  cos = %.10f  (期望 1.0)\n", c1);

    // ---- 连续执行：同一 cmd 两次 extract（层间不 download/upload），与 CPU 双跑对比 ----
    if (in_dim != out_dim)
    {
        fprintf(stderr, "  [2][3] 跳过（in_dim=%d != out_dim=%d，非方阵层无法连续双跑）\n", in_dim, out_dim);
    }
    else
    {
        ncnn::VkCompute cmd2(vkdev);
        ncnn::VkMat vk_in2;
        cmd2.record_upload(in_mat, vk_in2, net.opt);
        ncnn::VkMat vk_a, vk_b;
        {
            ncnn::Extractor ex1 = net.create_extractor();
            ex1.input("in0", vk_in2);
            ex1.extract("out0", vk_a, cmd2);
            ncnn::Extractor ex2 = net.create_extractor();
            ex2.input("in0", vk_a);              // GPU -> GPU，零往返
            ex2.extract("out0", vk_b, cmd2);
            // ex1/ex2 保持存活到 submit 后（否则 blob buffer 可能被 allocator 复用）
            ncnn::VkMat vk_b_p1;
            vkdev->convert_packing(vk_b, vk_b_p1, 1, cmd2, net.opt);
            ncnn::Mat m2;
            { ncnn::Option opt_dl = net.opt; opt_dl.use_packing_layout = false; cmd2.record_download(vk_b_p1, m2, opt_dl); }
            cmd2.submit_and_wait();
            cmd2.reset();
            std::vector<float> y_b;
            if (m2.elempack == 1 && m2.w == out_dim && m2.h == Ln) {
                y_b.assign((const float*)m2.data, (const float*)m2.data + (size_t)Ln * out_dim);
            } else { fprintf(stderr, "FAIL 连续版下载 dims w=%d h=%d ep=%d\n", m2.w, m2.h, m2.elempack); return 1; }
            // CPU 双跑参考
            std::vector<float> y_cpu2;
            {
                ncnn::Mat in2(in_dim, Ln, (void*)x.data());
                ncnn::Mat mid, out2;
                {
                    ncnn::Extractor e1 = net.create_extractor();
                    e1.input("in0", in2);
                    e1.extract("out0", mid);
                }
                {
                    ncnn::Mat in2b(in_dim, Ln, (void*)mid.data);   // 假定 mid 行序 (Ln, in_dim)
                    ncnn::Extractor e2 = net.create_extractor();
                    e2.input("in0", in2b);
                    e2.extract("out0", out2);
                }
                if (out2.w == out_dim && out2.h == Ln) y_cpu2.assign((const float*)out2.data, (const float*)out2.data + (size_t)Ln * out_dim);
                else if (out2.w == Ln && out2.h == out_dim) {
                    y_cpu2.resize((size_t)Ln * out_dim);
                    for (int t = 0; t < Ln; t++) for (int o = 0; o < out_dim; o++) y_cpu2[t * out_dim + o] = ((const float*)out2.data)[o * Ln + t];
                } else { fprintf(stderr, "FAIL cpu2 dims\n"); return 1; }
            }
            double c2 = cos_sim(y_b, y_cpu2);
            fprintf(stderr, "  [2] 连续双跑(VkMat直通) vs CPU双跑  cos = %.10f  (期望 1.0)\n", c2);

            // ---- 加速比：VkMat 直通（同 cmd 双跑） vs CPU 式（每跑一次 upload+download）----
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                for (int it = 0; it < iters; it++) {
                    ncnn::VkCompute c3(vkdev);
                    ncnn::VkMat vi; c3.record_upload(in_mat, vi, net.opt);
                    ncnn::VkMat va, vb;
                    ncnn::Extractor e1 = net.create_extractor();
                    e1.input("in0", vi); e1.extract("out0", va, c3);
                    ncnn::Extractor e2 = net.create_extractor();
                    e2.input("in0", va); e2.extract("out0", vb, c3);
                    ncnn::VkMat vb_p1;
                    vkdev->convert_packing(vb, vb_p1, 1, c3, net.opt);
                    ncnn::Mat mm; { ncnn::Option opt_dl = net.opt; opt_dl.use_packing_layout = false; c3.record_download(vb_p1, mm, opt_dl); }
                    c3.submit_and_wait();
                }
                double t_vk = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count() / iters;

                auto t1 = std::chrono::high_resolution_clock::now();
                for (int it = 0; it < iters; it++) {
                    ncnn::Mat in2(in_dim, Ln, (void*)x.data());
                    ncnn::Mat mid, out2;
                    {
                        ncnn::Extractor e1 = net.create_extractor();
                        e1.input("in0", in2); e1.extract("out0", mid);
                    }
                    {
                        ncnn::Mat in2b(in_dim, Ln, (void*)mid.data);
                        ncnn::Extractor e2 = net.create_extractor();
                        e2.input("in0", in2b); e2.extract("out0", out2);
                    }
                }
                double t_cpu = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t1).count() / iters;
                fprintf(stderr, "  [3] 双跑耗时 VkMat直通=%.3fms  CPU往返式=%.3fms  加速=%.2fx\n",
                        t_vk * 1e3, t_cpu * 1e3, t_cpu / t_vk);
            }
        }
    }

    ok = (c1 > 1 - 1e-6);
    }   // 作用域结束：net/cmd 等先析构，GPU 实例仍活

    ncnn::destroy_gpu_instance();
    return ok ? 0 : 1;
}

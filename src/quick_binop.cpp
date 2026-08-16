// quick_binop.cpp — BinaryOp 广播最小测试：Input in0(2560,64) + Input in1(2560,1) -> ADD/MUL
#include <ncnn/net.h>
#include <ncnn/gpu.h>
#include <ncnn/mat.h>
#include <ncnn/command.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

int main()
{
    bool ok = false;
    {
    ncnn::create_gpu_instance();
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
    if (net.load_param("models/m5_graph/t8_add_same.param") != 0) { printf("FAIL param\n"); return 1; }
    printf("[binop] load_param OK\n");
    {   // 无权重层：load_model 空 bin（pipelinize 创建各层 vulkan pipeline）
        unsigned char empty[1] = {0};
        if (net.load_model(empty) < 0) { printf("FAIL load_model\n"); return 1; }
        printf("[binop] load_model OK\n");
    }

    ncnn::VkCompute cmd(vkdev);
    ncnn::Extractor ex = net.create_extractor();
    // in0 (2560, 64), in1 (2560, 1) 广播
    int DIM = 2560, L = 64;
    ncnn::Mat m0; m0.create(DIM, L, (size_t)4u, 1);
    ncnn::Mat m1; m1.create(DIM, L, (size_t)4u, 1);
    for (int i = 0; i < DIM * L; i++) ((float*)m0.data)[i] = 1.0f;
    for (int i = 0; i < DIM * L; i++) ((float*)m1.data)[i] = 2.0f;
    ncnn::VkMat v0, v1;
    cmd.record_upload(m0, v0, net.opt);
    cmd.record_upload(m1, v1, net.opt);
    ex.input("in0", v0);
    ex.input("in1", v1);
    ncnn::VkMat vo;
    ex.extract("o0", vo, cmd);
    ncnn::VkMat vop;
    vkdev->convert_packing(vo, vop, 1, cmd, net.opt);
    ncnn::Mat mo;
    { ncnn::Option od = net.opt; od.use_packing_layout = false; cmd.record_download(vop, mo, od); }
    cmd.submit_and_wait();
    printf("[binop] out: w=%d h=%d ep=%d, [0]=%.2f (期望 3.0)\n", mo.w, mo.h, mo.elempack, ((const float*)mo.data)[0]);
    fflush(stdout);

    ok = true;
    }
    ncnn::destroy_gpu_instance();
    return ok ? 0 : 1;
}

// test_awa_min.cpp — 隔离测试：用与 AwaVk::forward 完全相同的 plumbing
// （12 绑定 + 10 push constant + dispatcher + record_pipeline + download）
// 跑一个最小 shader（awa_min.spv）。若仍 device-lost -> plumbing 有问题；
// 若成功 -> AWA 着色器本体有问题，需二分定位。
#include <ncnn/gpu.h>
#include <ncnn/mat.h>
#include <ncnn/pipeline.h>
#include <ncnn/command.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>

static std::vector<uint32_t> read_spv(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "[min] FAIL 读 %s\n", path.c_str()); exit(1); }
    f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint32_t> d(n / 4); f.read((char*)d.data(), n); return d;
}

int main(int argc, char** argv) {
    std::string spv = (argc > 1) ? argv[1] : "models/m5/awa_min.spv";
    if (ncnn::create_gpu_instance() != 0) { fprintf(stderr, "[min] FAIL gpu\n"); return 1; }
    ncnn::VulkanDevice* vkdev = ncnn::get_gpu_device(0);

    auto spv_data = read_spv(spv);
    ncnn::Pipeline pipe(vkdev);
    pipe.set_local_size_xyz(8, 8, 1);
    if (pipe.create(spv_data.data(), spv_data.size() * 4, std::vector<ncnn::vk_specialization_type>()) != 0) {
        fprintf(stderr, "[min] FAIL create %s\n", spv.c_str()); return 1;
    }
    fprintf(stderr, "[min] create ok (%zu dwords)\n", spv_data.size());

    const int DIM=2560, HEAD_D=128, HEADS=20, ROPE_ROT=126, nwin=2, TXT=4, Lv=12, sumf=12;

    ncnn::Option opt;
    opt.use_vulkan_compute = true;
    opt.use_packing_layout = false;
    opt.blob_vkallocator    = vkdev->acquire_blob_allocator();
    opt.staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::VkCompute cmd(vkdev);
    auto upload = [&](const std::vector<float>& src, ncnn::VkMat& dst) {
        ncnn::Mat m; m.create((int)src.size());
        memcpy(m.data, src.data(), src.size() * sizeof(float));
        cmd.record_upload(m, dst, opt);
    };
    std::vector<float> vqkv(Lv*3*DIM, 1.0f), tqkv(TXT*3*DIM, 2.0f);
    std::vector<float> vidx(sumf, 0.f), cumf(nwin+1, 0.f), vfreq(sumf*ROPE_ROT, 0.1f), tfreq(nwin*TXT*ROPE_ROT, 0.2f);
    std::vector<float> nqv(HEAD_D,1.f), nkv(HEAD_D,1.f), nqt(HEAD_D,1.f), nkt(HEAD_D,1.f);
    ncnn::VkMat vk_vqkv,vk_tqkv,vk_vidx,vk_cumf,vk_vfreq,vk_tfreq,vk_nqv,vk_nkv,vk_nqt,vk_nkt;
    upload(vqkv,vk_vqkv); upload(tqkv,vk_tqkv); upload(vidx,vk_vidx); upload(cumf,vk_cumf);
    upload(vfreq,vk_vfreq); upload(tfreq,vk_tfreq); upload(nqv,vk_nqv); upload(nkv,vk_nkv);
    upload(nqt,vk_nqt); upload(nkt,vk_nkt);

    ncnn::VkMat vk_vattn, vk_toutw;
    vk_vattn.create(Lv*DIM, (size_t)4, vkdev->acquire_blob_allocator());
    vk_toutw.create(nwin*TXT*DIM, (size_t)4, vkdev->acquire_blob_allocator());
    fprintf(stderr, "[min] upload/alloc done vattn.w=%d toutw.w=%d\n", vk_vattn.w, vk_toutw.w);

    std::vector<ncnn::VkMat> bindings(12);
    bindings[0]=vk_vqkv; bindings[1]=vk_tqkv; bindings[2]=vk_vidx; bindings[3]=vk_cumf;
    bindings[4]=vk_vfreq; bindings[5]=vk_tfreq; bindings[6]=vk_nqv; bindings[7]=vk_nkv;
    bindings[8]=vk_nqt; bindings[9]=vk_nkt; bindings[10]=vk_vattn; bindings[11]=vk_toutw;

    std::vector<ncnn::vk_constant_type> constants(10);
    constants[0].i=DIM; constants[1].i=HEAD_D; constants[2].i=HEADS; constants[3].i=ROPE_ROT;
    constants[4].i=nwin; constants[5].i=TXT; constants[6].i=Lv; constants[7].i=sumf;
    constants[8].f=1.0f/sqrtf((float)HEAD_D); constants[9].f=1e-5f;

    ncnn::VkMat dispatcher;
    dispatcher.w = Lv; dispatcher.h = HEADS; dispatcher.c = nwin;

    cmd.record_pipeline(&pipe, bindings, constants, dispatcher);
    ncnn::Mat m_vattn, m_toutw;
    cmd.record_download(vk_vattn, m_vattn, opt);
    cmd.record_download(vk_toutw, m_toutw, opt);
    fprintf(stderr, "[min] submit...\n");
    cmd.submit_and_wait();
    cmd.reset();
    fprintf(stderr, "[min] submit done  m_vattn.w=%d m_toutw.w=%d\n", m_vattn.w, m_toutw.w);
    if (m_vattn.w != Lv*DIM) { fprintf(stderr, "[min] FAIL size\n"); return 1; }
    const float* d = (const float*)m_vattn.data;
    fprintf(stderr, "[min] vattn[0..15] =");
    for (int i = 0; i < 16; i++) fprintf(stderr, " %.4f", d[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "[min] PASS ✅ (无 device-lost, 尺寸正确)\n");
    ncnn::destroy_gpu_instance();
    return 0;
}

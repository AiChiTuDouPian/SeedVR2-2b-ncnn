// test_copy.cpp — 诊断 ncnn Vulkan 上传/绑定/回读链路：
// 上传一个已知的 float 向量（0..N-1 的递增值），用 copy shader 原样拷回，下载比对。
#include <ncnn/gpu.h>
#include <ncnn/mat.h>
#include <ncnn/pipeline.h>
#include <ncnn/command.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>

static std::vector<uint32_t> read_spv(const std::string& path){
    FILE* f=fopen(path.c_str(),"rb"); if(!f){fprintf(stderr,"[copy] FAIL open %s\n",path.c_str());exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    std::vector<uint32_t> d(n/4); fread(d.data(),1,n,f); fclose(f); return d;
}

int main(int argc,char**argv){
    if(ncnn::create_gpu_instance()!=0){fprintf(stderr,"[copy] FAIL gpu\n");return 1;}
    ncnn::VulkanDevice* vkdev=ncnn::get_gpu_device(0);
    std::string spv = (argc>1)? argv[1] : "models/m5/awa_copy.spv";
    std::vector<uint32_t> spvdata=read_spv(spv);
    ncnn::Pipeline* pipe=new ncnn::Pipeline(vkdev);
    pipe->set_local_size_xyz(1,1,1);
    if(pipe->create(spvdata.data(), spvdata.size()*4, std::vector<ncnn::vk_specialization_type>())!=0){
        fprintf(stderr,"[copy] FAIL create %s\n",spv.c_str());return 1;}
    fprintf(stderr,"[copy] init ok %s\n",spv.c_str());

    int N = (argc>2)? atoi(argv[2]) : 256;
    std::vector<float> host(N);
    for(int i=0;i<N;i++) host[i]=(float)i;   // 已知递增值

    ncnn::Option opt; opt.use_vulkan_compute=true; opt.use_packing_layout=false;
    // 关键：离散 GPU 默认 use_fp16_packed=true，record_upload 会把 fp32 数据转成 fp16 存入 buffer；
    // 自定义 shader 用 flat float[] 读取时必须保持 fp32，否则读到的是 fp16 字节(被当 fp32 解读=乱码)。
    opt.use_fp16_storage=false; opt.use_fp16_packed=false;
    opt.use_bf16_storage=false; opt.use_bf16_packed=false;
    opt.blob_vkallocator=vkdev->acquire_blob_allocator();
    opt.staging_vkallocator=vkdev->acquire_staging_allocator();

    ncnn::Mat m; m.create(N); memcpy(m.data, host.data(), N*sizeof(float));
    ncnn::VkCompute cmd(vkdev);
    ncnn::VkMat vk_in, vk_out;
    cmd.record_upload(m, vk_in, opt);
    vk_out.create(N, (size_t)4, vkdev->acquire_blob_allocator());
    fprintf(stderr,"[copy] in.w=%d out.w=%d\n", vk_in.w, vk_out.w);

    std::vector<ncnn::VkMat> bindings(2);
    bindings[0]=vk_in; bindings[1]=vk_out;
    std::vector<ncnn::vk_constant_type> constants(1);
    constants[0].i=N;
    ncnn::VkMat dispatcher; dispatcher.w=N; dispatcher.h=1; dispatcher.c=1;
    cmd.record_pipeline(pipe, bindings, constants, dispatcher);

    ncnn::Mat m_out;
    cmd.record_download(vk_out, m_out, opt);
    cmd.submit_and_wait(); cmd.reset();
    fprintf(stderr,"[copy] download m_out.w=%d\n", m_out.w);

    const float* o=(const float*)m_out.data;
    int bad=0; float maxd=0;
    for(int i=0;i<N;i++){ float d=fabs(o[i]-host[i]); if(d>1e-3)bad++; maxd=std::max(maxd,d);}
    fprintf(stderr,"[copy] bad=%d/%d maxdiff=%.6f\n", bad, N, maxd);
    fprintf(stderr,"[copy] host[0..7]=%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
        host[0],host[1],host[2],host[3],host[4],host[5],host[6],host[7]);
    fprintf(stderr,"[copy] out [0..7]=%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
        o[0],o[1],o[2],o[3],o[4],o[5],o[6],o[7]);
    delete pipe; ncnn::destroy_gpu_instance();
    return bad? 1:0;
}

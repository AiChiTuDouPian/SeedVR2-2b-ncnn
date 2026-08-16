// dit_vk_test.cpp — 验证 Vulkan DiT 引擎数值正确性
// 用 M5 导出的 x_patch/x_txt + win_ns/win_sh 跑 DitVk.forward，
// 与 ref_vid_out.bin 对拍 cos≥0.99。
#include "dit_vk.h"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <cstdint>

static std::vector<float> load_raw(const char* path, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary); if (!f) { fprintf(stderr,"[FAIL] %s\n",path); exit(1); }
    int64_t ndim; f.read((char*)&ndim, 8); shape.resize(ndim);
    for (auto& d : shape) f.read((char*)&d, 8);
    int64_t total = 1; for (auto d : shape) total *= d;
    std::vector<float> d(total); f.read((char*)d.data(), total*4); return d;
}
static std::vector<float> load_raw_f(const char* path) { std::vector<int64_t> s; return load_raw(path, s); }
static float cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot=0,na=0,nb=0; size_t n=a.size()<b.size()?a.size():b.size();
    for (size_t i=0;i<n;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    return (float)(dot/(sqrt(na)*sqrt(nb)+1e-12));
}
static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m=0; size_t n=a.size()<b.size()?a.size():b.size();
    for (size_t i=0;i<n;i++) m=std::max(m,std::fabs(a[i]-b[i])); return m;
}

int main(int argc, char** argv) {
    bool fp16_arith = (argc > 1 && std::string(argv[1])=="fp16");
    const char* M5 = "models/m5/";
    DitVk eng;
    if (!eng.init(M5, fp16_arith)) return 1;

    // 输入（已是 patchify 后的 token）
    std::vector<int64_t> sh;
    auto x_patch = load_raw((std::string(M5)+"x_patch.bin").c_str(), sh);
    auto x_txt   = load_raw((std::string(M5)+"x_txt.bin").c_str(), sh);
    const int VID_IN_CH = 33*1*2*2, TXT_IN_CH = 5120;
    int Lv = (int)x_patch.size()/VID_IN_CH, TXT = (int)x_txt.size()/TXT_IN_CH;
    fprintf(stderr, "[info] Lv=%d TXT=%d fp16_arith=%d\n", Lv, TXT, (int)fp16_arith);

    std::vector<float> out_sr;
    if (!eng.forward(x_patch, Lv, x_txt, TXT, 500.0f, out_sr)) return 1;
    fprintf(stderr, "[test] forward 完成, out_sr 元素=%zu，写盘 out_vk.bin\n", out_sr.size());
    { // 落盘，避免退出清理崩溃丢结果
        std::ofstream fo("models/m5/out_vk.bin", std::ios::binary);
        int64_t n = (int64_t)out_sr.size(); fo.write((char*)&n, 8); fo.write((char*)out_sr.data(), n*4);
    }

    auto ref = load_raw_f((std::string(M5)+"ref_vid_out.bin").c_str());
    float c = cosine(out_sr, ref), dd = max_abs_diff(out_sr, ref);
    fprintf(stderr, "[FINAL vid_out] cos=%.6f max|diff|=%.5f\n", c, dd);
    if (c < 0.99f || dd > 0.5f) { fprintf(stderr, "\n[FAIL] forward cos 不足\n"); return 1; }

    // ---- forward_grid 往返校验（验证 patchify/unpatchify 索引） ----
    // 由 x_patch(3200,132) 反推 latent 网格 (T=2, H=80, W=80, 33)
    int Tg=2, Hg=80, Wg=80; int H2=Hg/2, W2=Wg/2;
    std::vector<float> grid((size_t)Tg*Hg*Wg*33, 0.f);
    for (int t=0;t<Tg;t++) for(int hh=0;hh<H2;hh++) for(int ww=0;ww<W2;ww++)
        for(int ph=0;ph<2;ph++) for(int pw=0;pw<2;pw++) for(int cc=0;cc<33;cc++){
            int li=((t*H2+hh)*W2+ww)*132+((ph*2+pw)*33+cc);
            int gi=((t*Hg+(hh*2+ph))*Wg+(ww*2+pw))*33+cc;
            grid[gi]=x_patch[li];
        }
    std::vector<float> sr_latent;
    if (!eng.forward_grid(grid, Tg, Hg, Wg, x_txt, TXT, 500.0f, sr_latent)) return 1;
    // 再 patchify sr_latent -> (3200,64)
    std::vector<float> sr_patch((size_t)Tg*H2*W2*64, 0.f);
    for (int t=0;t<Tg;t++) for(int hh=0;hh<H2;hh++) for(int ww=0;ww<W2;ww++)
        for(int ph=0;ph<2;ph++) for(int pw=0;pw<2;pw++) for(int cc=0;cc<16;cc++){
            int li=((t*H2+hh)*W2+ww)*64+((ph*2+pw)*16+cc);
            int gi=((t*Hg+(hh*2+ph))*Wg+(ww*2+pw))*16+cc;
            sr_patch[li]=sr_latent[gi];
        }
    float c2 = cosine(sr_patch, ref), dd2 = max_abs_diff(sr_patch, ref);
    fprintf(stderr, "[FINAL grid-path] cos=%.6f max|diff|=%.5f\n", c2, dd2);
    if (c2 >= 0.99f && dd2 <= 0.5f) { fprintf(stderr, "\n[PASS] Vulkan DiT 引擎验证通过（forward + forward_grid 双路径）\n"); return 0; }
    fprintf(stderr, "\n[FAIL] forward_grid 路径 cos 不足\n"); return 1;
}

// seedvr2_pipeline.cpp — 端到端 DiT 管线实现
#include "seedvr2_pipeline.h"
#include "dit_vk.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cstdint>
#include <string>

static const int NUM_LAYERS = 32;   // 与 dit_vk.cpp 保持一致；实际层数可被 SEEDVR_MAX_LAYERS 覆盖

static void write_raw(const std::string& path, const std::vector<int64_t>& shape, const std::vector<float>& data) {
    std::ofstream f(path, std::ios::binary);
    int64_t ndim = (int64_t)shape.size(); f.write((char*)&ndim, 8);
    for (auto d : shape) f.write((char*)&d, 8);
    f.write((char*)data.data(), data.size()*4);
}
static std::vector<float> read_raw(const std::string& path, std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary); if (!f) { fprintf(stderr,"[FAIL] %s\n",path.c_str()); exit(1); }
    int64_t ndim; f.read((char*)&ndim, 8); shape.resize(ndim);
    for (auto& d : shape) f.read((char*)&d, 8);
    int64_t total = 1; for (auto d : shape) total *= d;
    std::vector<float> d(total); f.read((char*)d.data(), total*4); return d;
}

namespace seedvr2 {

int run_dit(const std::string& workdir, const std::string& modeldir, bool fp16_arith) {
    std::string wd = workdir; if (wd.back()!='/') wd += '/';
    std::string md = modeldir; if (md.back()!='/') md += '/';

    int T=0, H=0, W=0, TXT=0; float ts=1000.f;
    {
        std::ifstream pf(wd + "params.txt");
        if (!(pf >> T >> H >> W >> TXT >> ts)) { fprintf(stderr, "[FAIL] 读 params.txt\n"); return 1; }
    }
    fprintf(stderr, "[pipeline] T=%d H=%d W=%d TXT=%d ts=%.1f fp16=%d\n", T, H, W, TXT, ts, (int)fp16_arith);

    std::vector<int64_t> sh;
    auto vid_grid = read_raw(wd + "vid_grid.bin", sh);
    auto txt       = read_raw(wd + "txt.bin", sh);

    DitVk eng;
    if (!eng.init(md, fp16_arith)) return 1;

    // 真实模型 txt_len=TXT（=58）。workdir 的窗口按实际 grid + TXT 生成，覆盖 init 默认加载。
    Win wns = load_win((wd + "win_ns.bin").c_str(), TXT);
    Win wsh = load_win((wd + "win_sh.bin").c_str(), TXT);
    eng.set_windows(wns, wsh);

    int Lv = T * (H / 2) * (W / 2);
    std::vector<float> sr;
    {
        int K = 0;
        if (const char* ck = getenv("SEEDVR_CHUNK")) { K = atoi(ck); }
        if (K <= 0) {
            // 单块全量（旧行为）
            if (!eng.forward_grid(vid_grid, T, H, W, txt, TXT, ts, sr)) return 1;
        } else {
            // 分块推理：每块间 reset_vulkan_device() 拿到全新 VkDevice，绕开深层资源累积。
            fprintf(stderr, "[pipeline] 分块推理 K=%d (总 %d 层)\n", K, NUM_LAYERS); fflush(stderr);
            std::vector<float> cur_vid = DitVk::patchify_grid(vid_grid, T, H, W);
            std::vector<float> cur_txt = txt;
            int done = 0;
            bool ok = true;
            while (done < NUM_LAYERS) {
                int l0 = done;
                int l1 = (done + K < NUM_LAYERS) ? done + K : NUM_LAYERS;
                bool do_init  = (l0 == 0);
                bool is_final = (l1 >= NUM_LAYERS);
                std::vector<float> vout, tout, sro;
                if (!eng.forward_latent(cur_vid, Lv, cur_txt, TXT, ts, l0, l1, do_init, is_final, vout, tout, sro)) {
                    ok = false; break;
                }
                if (!sro.empty()) {
                    sr = DitVk::unpatchify_latent(sro, T, H, W);   // 末块：do_final 已写 sr_out
                } else {
                    cur_vid = vout; cur_txt = tout;          // 块间用残差 latent 续传
                }
                done = l1;
                if (done < NUM_LAYERS) eng.reset_vulkan_device();
            }
            if (!ok) return 1;
            if (sr.empty()) { fprintf(stderr, "[FAIL] 分块推理未产出 sr\n"); return 1; }
        }
    }

    std::vector<int64_t> oshape = {(int64_t)T, (int64_t)H, (int64_t)W, 16};
    write_raw(wd + "sr_latent.bin", oshape, sr);
    fprintf(stderr, "[pipeline] 写出 sr_latent.bin (%zu 元素)\n", sr.size());
    return 0;
}

} // namespace seedvr2

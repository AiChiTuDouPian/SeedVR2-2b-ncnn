// verify_awa_vk.cpp — 自定义 Vulkan AWA 模块数值对拍
// 同一个模型/窗口/输入，分别用：
//   GPU 路径（modeldir 含 awa.spv -> AwaVk 就绪，注意力在 shader 内完成）
//   CPU 路径（modeldir 不含 awa.spv -> 走 dit_vk.cpp 的 CPU awa_forward）
// 跑 forward_grid，比较两路 sr_latent 的 cos，要求 ≈1（验证 GLSL 数学等价 CPU）。
#include "dit_vk.h"
#include <cstdio>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <string>
#ifdef _WIN32
#include <windows.h>
#include <stdio.h>
#include <signal.h>
static void my_sig(int s) {
    fprintf(stderr, "[CRASH] signal=%d", s);
    if (s == SIGSEGV) fprintf(stderr, " SIGSEGV(access violation)");
    else if (s == SIGABRT) fprintf(stderr, " SIGABRT(abort/assert)");
    else if (s == SIGFPE) fprintf(stderr, " SIGFPE");
    fprintf(stderr, "\n");
    fflush(stderr);
    exit(2);
}
static LONG WINAPI my_filter(EXCEPTION_POINTERS* ep) {
    DWORD code = ep->ExceptionRecord->ExceptionCode;
    fprintf(stderr, "[CRASH] ExceptionCode=0x%08lX\n", code);
    if (code == 0xC0000005) fprintf(stderr, "[CRASH] ACCESS VIOLATION\n");
    else if (code == 0xC0000374) fprintf(stderr, "[CRASH] HEAP CORRUPTION\n");
    else if (code == 0xC00000FD) fprintf(stderr, "[CRASH] STACK OVERFLOW\n");
    else if (code == 0xC0000094) fprintf(stderr, "[CRASH] DIVIDE BY ZERO\n");
    fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

static std::vector<float> read_raw(const std::string& path, std::vector<int64_t>& sh) {
    std::ifstream f(path, std::ios::binary); if (!f) { fprintf(stderr,"[FAIL] %s\n",path.c_str()); exit(1); }
    int64_t ndim; f.read((char*)&ndim, 8); sh.resize(ndim);
    for (auto& d : sh) f.read((char*)&d, 8);
    int64_t total = 1; for (auto d : sh) total *= d;
    std::vector<float> d(total); f.read((char*)d.data(), total*4); return d;
}

static double cos_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot=0, na=0, nb=0; for (size_t i=0;i<a.size();i++){ dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
    return dot / (sqrt(na)*sqrt(nb) + 1e-12);
}
static float maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m=0; for (size_t i=0;i<a.size();i++) m = std::max(m, std::fabs(a[i]-b[i])); return m;
}

static int run(const std::string& wd, const std::string& md, std::vector<float>& sr_latent, bool fp16=false, int chunk=0) {
    int T=0,H=0,W=0,TXT=0; float ts=1000.f;
    { std::ifstream pf(wd+"params.txt"); if(!(pf>>T>>H>>W>>TXT>>ts)){fprintf(stderr,"[FAIL] params\n");return 1;} }
    std::vector<int64_t> sh;
    auto vid_grid = read_raw(wd+"vid_grid.bin", sh);
    auto txt = read_raw(wd+"txt.bin", sh);
    DitVk eng;
    if (!eng.init(md, fp16)) return 1;
    fprintf(stderr, "  [verify] awa.ready=%d chunk=%d\n", (int)eng.awa_ready(), chunk);
    Win wns = load_win((wd+"win_ns.bin").c_str(), TXT);
    Win wsh = load_win((wd+"win_sh.bin").c_str(), TXT);
    eng.set_windows(wns, wsh);
    int H2=H/2,W2=W/2; int Lv=T*H2*W2;
    std::vector<float> vid_patch = DitVk::patchify_grid(vid_grid, T,H,W);
    std::vector<float> sr;            // 最终 (Lv,64)
    std::vector<float> vid_lat = vid_patch, txt_lat = txt;
    int NL = 32; if (const char* ml=getenv("SEEDVR_MAX_LAYERS")) { NL = atoi(ml); if (NL<1) NL=32; }
    if (chunk <= 0) {
        std::vector<float> dv, dt;
        if (!eng.forward_latent(vid_lat, Lv, txt_lat, TXT, ts, 0, NL, true, true, dv, dt, sr)) return 1;
    } else {
        for (int c0 = 0; c0 < NL; c0 += chunk) {
            if (c0 > 0) eng.reset_vulkan_device();   // 块间重置 Vulkan 设备，绕开深层累积崩溃
            int l1 = (c0+chunk < NL) ? (c0+chunk) : NL;
            bool do_init  = (c0 == 0);
            bool do_final = (l1 >= NL);
            std::vector<float> vout, tout;
            fprintf(stderr, "  [verify] 块 [%d,%d) do_init=%d do_final=%d\n", c0, l1, (int)do_init, (int)do_final);
            if (!eng.forward_latent(vid_lat, Lv, txt_lat, TXT, ts, c0, l1, do_init, do_final, vout, tout, sr)) return 1;
            if (do_final) break;
            vid_lat = vout; txt_lat = tout;
        }
    }
    sr_latent = DitVk::unpatchify_latent(sr, T, H, W);
    return 0;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetUnhandledExceptionFilter(my_filter);
    signal(SIGSEGV, my_sig);
    signal(SIGABRT, my_sig);
    signal(SIGFPE, my_sig);
#endif
    if (argc < 4) { fprintf(stderr, "用法: %s <workdir> <gpu_modeldir> <cpu_modeldir>\n", argv[0]); return 1; }
    std::string wd = argv[1]; if (wd.back()!='/') wd+='/';
    std::string gmd = argv[2]; if (gmd.back()!='/') gmd+='/';
    std::string cmd = argv[3]; if (cmd.back()!='/') cmd+='/';

    std::vector<float> sr_gpu, sr_cpu;
    bool fp16 = (argc >= 5 && std::string(argv[4]) == "fp16");
    int chunk = 0; if (const char* ck = getenv("SEEDVR_CHUNK")) { chunk = atoi(ck); }
    fprintf(stderr, "[verify] fp16_arith=%d  SEEDVR_CHUNK=%d\n", (int)fp16, chunk);
    fprintf(stderr, "[verify] === GPU 路径 (AwaVk) ===\n");
    if (run(wd, gmd, sr_gpu, fp16, chunk)) return 1;
    fprintf(stderr, "[verify] === CPU 路径 (参考) ===\n");
    if (run(wd, cmd, sr_cpu, false, 0)) return 1;

    if (sr_gpu.size() != sr_cpu.size()) { fprintf(stderr, "[FAIL] 尺寸不一致\n"); return 1; }
    double c = cos_sim(sr_gpu, sr_cpu);
    float md_ = maxdiff(sr_gpu, sr_cpu);
    fprintf(stderr, "[verify] FINAL cos(GPU AWA, CPU ref) = %.6f  max|diff| = %.5f  (n=%zu)\n",
            c, md_, sr_gpu.size());
    if (c < 0.999) { fprintf(stderr, "[FAIL] cos 过低\n"); return 1; }
    fprintf(stderr, "[verify] PASS ✅ 自定义 Vulkan AWA 与 CPU 参考一致\n");
    return 0;
}

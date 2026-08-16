// verify_window.cpp — 验证 C++ 窗口生成（对比 Python awa_window + mmrope）
// 用法: seedvr2_verify_window <t> <h> <w> <txt_len> <out_ns.bin> <out_sh.bin>
#include "awa_window.h"
#include <cstdio>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>

static void write_win(const std::string& path, const Win& win) {
    std::ofstream f(path, std::ios::binary);
    int64_t hdr[10] = {win.t, win.h, win.w, 4, 3, 3, 0, 0, 0, win.nwin};
    for (int i = 0; i < 10; i++) f.write((char*)&hdr[i], 8);
    std::vector<int64_t> sl;
    for (int i = 0; i < win.nwin; i++) {
        sl.push_back(win.st[i]); sl.push_back(win.en[i]);
        sl.push_back(win.sh[i]); sl.push_back(win.eh[i]);
        sl.push_back(win.sw[i]); sl.push_back(win.ew[i]);
    }
    f.write((char*)sl.data(), sl.size() * 8);
    f.write((char*)win.vid_freq.data(), win.vid_freq.size() * 4);
    f.write((char*)win.txt_freq.data(), win.txt_freq.size() * 4);
    fprintf(stderr, "[verify_window] 写出 %s: nwin=%d vid_freq=%zu txt_freq=%zu\n",
            path.c_str(), win.nwin, win.vid_freq.size(), win.txt_freq.size());
}

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "用法: %s <t> <h> <w> <txt_len> <out_ns> <out_sh>\n", argv[0]); return 1; }
    int t = atoi(argv[1]), h = atoi(argv[2]), w = atoi(argv[3]), txt_len = atoi(argv[4]);
    int num_windows[3] = {4, 3, 3};
    Win ns = awa::make_win(t, h, w, num_windows, txt_len, false);
    Win sh = awa::make_win(t, h, w, num_windows, txt_len, true);
    write_win(argv[5], ns);
    write_win(argv[6], sh);
    return 0;
}

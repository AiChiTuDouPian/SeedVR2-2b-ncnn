// awa_window.h — AWA 窗口划分 + mmrope 频率生成的 C++ 移植（替代 Python awa_window.py + mmrope.py）
#pragma once
#include <vector>
#include "dit_vk.h"   // Win 结构

namespace awa {

// 生成非 shifted 窗口的 slice 列表（每窗口 [st_t,en_t,st_h,en_h,st_w,en_w]）
void make_windows(int t, int h, int w, const int num_windows[3],
                  std::vector<int>& st_t, std::vector<int>& en_t,
                  std::vector<int>& st_h, std::vector<int>& en_h,
                  std::vector<int>& st_w, std::vector<int>& en_w);

// 生成 shifted 窗口
void make_shifted_windows(int t, int h, int w, const int num_windows[3],
                          std::vector<int>& st_t, std::vector<int>& en_t,
                          std::vector<int>& st_h, std::vector<int>& en_h,
                          std::vector<int>& st_w, std::vector<int>& en_w);

// 生成 rope 频率（vid_freq: sum(f_i)*ROPE_ROT, txt_freq: nwin*txt_len*ROPE_ROT）
void build_freqs(const std::vector<int>& wt, const std::vector<int>& wh,
                 const std::vector<int>& ww, int txt_len,
                 std::vector<float>& vid_freq, std::vector<float>& txt_freq);

// 完整生成一个 Win（窗口 + freq）
Win make_win(int t, int h, int w, const int num_windows[3], int txt_len, bool shifted);

} // namespace awa

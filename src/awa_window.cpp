// awa_window.cpp — AWA 窗口划分 + mmrope 频率生成（从 Python awa_window.py/mmrope.py 忠实移植）
#include "awa_window.h"
#include <cmath>
#include <algorithm>
#include <cstdio>

namespace awa {

static const int ROPE_DIM = 42;      // = rope_dim(128) // 3
static const int ROT_DIM = 42;       // 2*ceil(ROPE_DIM/2)
static const int FREQ_LAST = 126;    // 3 * ROT_DIM
static const double THETA = 10000.0;

// 生成窗口 slice（非 shifted）
void make_windows(int t, int h, int w, const int num_windows[3],
                  std::vector<int>& st_t, std::vector<int>& en_t,
                  std::vector<int>& st_h, std::vector<int>& en_h,
                  std::vector<int>& st_w, std::vector<int>& en_w) {
    st_t.clear(); en_t.clear(); st_h.clear(); en_h.clear(); st_w.clear(); en_w.clear();
    int rnt = num_windows[0], rnh = num_windows[1], rnw = num_windows[2];
    double scale = std::sqrt((45.0 * 80.0) / ((double)h * w));
    int resized_h = (int)std::round(h * scale);
    int resized_w = (int)std::round(w * scale);
    int wh = (resized_h + rnh - 1) / rnh;   // ceil
    int ww = (resized_w + rnw - 1) / rnw;
    int wt = (std::min(t, 30) + rnt - 1) / rnt;
    int nt = (t + wt - 1) / wt;
    int nh = (h + wh - 1) / wh;
    int nw = (w + ww - 1) / ww;
    for (int iw = 0; iw < nw; iw++)
        if (std::min((iw + 1) * ww, w) > iw * ww)
            for (int ih = 0; ih < nh; ih++)
                if (std::min((ih + 1) * wh, h) > ih * wh)
                    for (int it = 0; it < nt; it++)
                        if (std::min((it + 1) * wt, t) > it * wt) {
                            st_t.push_back(it * wt); en_t.push_back(std::min((it + 1) * wt, t));
                            st_h.push_back(ih * wh); en_h.push_back(std::min((ih + 1) * wh, h));
                            st_w.push_back(iw * ww); en_w.push_back(std::min((iw + 1) * ww, w));
                        }
}

void make_shifted_windows(int t, int h, int w, const int num_windows[3],
                          std::vector<int>& st_t, std::vector<int>& en_t,
                          std::vector<int>& st_h, std::vector<int>& en_h,
                          std::vector<int>& st_w, std::vector<int>& en_w) {
    st_t.clear(); en_t.clear(); st_h.clear(); en_h.clear(); st_w.clear(); en_w.clear();
    int rnt = num_windows[0], rnh = num_windows[1], rnw = num_windows[2];
    double scale = std::sqrt((45.0 * 80.0) / ((double)h * w));
    int resized_h = (int)std::round(h * scale);
    int resized_w = (int)std::round(w * scale);
    int wh = (resized_h + rnh - 1) / rnh;
    int ww = (resized_w + rnw - 1) / rnw;
    int wt = (std::min(t, 30) + rnt - 1) / rnt;

    double st = (wt < t) ? 0.5 : 0.0;
    double sh = (wh < h) ? 0.5 : 0.0;
    double sw = (ww < w) ? 0.5 : 0.0;
    int nt = (int)std::ceil((t - st) / wt);
    int nh = (int)std::ceil((h - sh) / wh);
    int nw = (int)std::ceil((w - sw) / ww);
    nt = (st > 0) ? nt + 1 : 1;
    nh = (sh > 0) ? nh + 1 : 1;
    nw = (sw > 0) ? nw + 1 : 1;

    for (int iw = 0; iw < nw; iw++)
        if (std::min((int)((iw - sw + 1) * ww), w) > std::max((int)((iw - sw) * ww), 0))
            for (int ih = 0; ih < nh; ih++)
                if (std::min((int)((ih - sh + 1) * wh), h) > std::max((int)((ih - sh) * wh), 0))
                    for (int it = 0; it < nt; it++)
                        if (std::min((int)((it - st + 1) * wt), t) > std::max((int)((it - st) * wt), 0)) {
                            st_t.push_back(std::max((int)((it - st) * wt), 0));
                            en_t.push_back(std::min((int)((it - st + 1) * wt), t));
                            st_h.push_back(std::max((int)((ih - sh) * wh), 0));
                            en_h.push_back(std::min((int)((ih - sh + 1) * wh), h));
                            st_w.push_back(std::max((int)((iw - sw) * ww), 0));
                            en_w.push_back(std::min((int)((iw - sw + 1) * ww), w));
                        }
}

// 生成 rope 频率（3D mmrope，与 dit_vk.cpp 的 ROPE_ROT=126 一致）
void build_freqs(const std::vector<int>& wt, const std::vector<int>& wh,
                 const std::vector<int>& ww, int txt_len,
                 std::vector<float>& vid_freq, std::vector<float>& txt_freq) {
    int nwin = (int)wt.size();
    int l = txt_len;
    int max_t = 0, max_h = 0, max_w = 0;
    for (int i = 0; i < nwin; i++) {
        max_t = std::max(max_t, l + wt[i]);
        max_h = std::max(max_h, wh[i]);
        max_w = std::max(max_w, ww[i]);
    }

    // base freq: 1/(theta^(arange(0,ROPE_DIM,2)/ROPE_DIM))，21 个值
    double base[21];
    for (int j = 0; j < 21; j++) base[j] = 1.0 / std::pow(THETA, (double)(j * 2) / ROPE_DIM);

    // vid_freq: 每窗口 (wt*wh*ww, 126)，时序 freq 从偏移 l 开始
    int total = 0;
    for (int i = 0; i < nwin; i++) total += wt[i] * wh[i] * ww[i];
    vid_freq.resize((size_t)total * FREQ_LAST);
    size_t off = 0;
    for (int i = 0; i < nwin; i++) {
        for (int it = 0; it < wt[i]; it++) {
            int tt = l + it;
            for (int ih = 0; ih < wh[i]; ih++) {
                for (int iw = 0; iw < ww[i]; iw++) {
                    for (int k = 0; k < FREQ_LAST; k++) {
                        double v;
                        if (k < ROT_DIM)            v = (double)tt * base[k / 2];
                        else if (k < 2 * ROT_DIM)   v = (double)ih * base[(k - ROT_DIM) / 2];
                        else                        v = (double)iw * base[(k - 2 * ROT_DIM) / 2];
                        vid_freq[off++] = (float)v;
                    }
                }
            }
        }
    }

    // txt_freq: 每窗口 (txt_len, 126)，tile(1,3) 重复
    txt_freq.resize((size_t)nwin * l * FREQ_LAST);
    off = 0;
    for (int i = 0; i < nwin; i++) {
        for (int it = 0; it < l; it++) {
            for (int k = 0; k < FREQ_LAST; k++) {
                int kk = k % ROT_DIM;
                double v = (double)it * base[kk / 2];
                txt_freq[off++] = (float)v;
            }
        }
    }
}

Win make_win(int t, int h, int w, const int num_windows[3], int txt_len, bool shifted) {
    Win win;
    win.t = t; win.h = h; win.w = w; win.txt_len = txt_len;
    if (shifted)
        make_shifted_windows(t, h, w, num_windows, win.st, win.en, win.sh, win.eh, win.sw, win.ew);
    else
        make_windows(t, h, w, num_windows, win.st, win.en, win.sh, win.eh, win.sw, win.ew);
    win.nwin = (int)win.st.size();

    std::vector<int> wt, wh, ww;
    for (int i = 0; i < win.nwin; i++) {
        wt.push_back(win.en[i] - win.st[i]);
        wh.push_back(win.eh[i] - win.sh[i]);
        ww.push_back(win.ew[i] - win.sw[i]);
    }
    build_freqs(wt, wh, ww, txt_len, win.vid_freq, win.txt_freq);
    return win;
}

} // namespace awa

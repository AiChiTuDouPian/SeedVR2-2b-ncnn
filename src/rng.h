// rng.h — 标准正态随机数生成，逐位对齐 PyTorch CPU 的 torch.randn
//
// PyTorch 2.x CPU 的 torch.randn 链路（已从 torch 2.13 源码逆向验证，逐位一致）：
//   1. 引擎：at::mt19937（= 标准 std::mt19937，同样的 seed 初始化 + tempering，输出一致）
//   2. uniform：u = (uint32 & 0xFFFFFF) * (1.0f / 16777216.0f)   // float32，2^-24
//   3. Box-Muller（AVX2 cephes 近似，16 元素一批，见 avx_mathfun.h）：
//        u1 = 1 - data[j] ; u2 = data[j+8]
//        radius = sqrt(-2.0f * cephes_log(u1))     // cephes log256_ps 的标量等价
//        theta  = float32( float32(2*pi) * u2 )    // 先 round 2pi 到 float，再乘
//        data[j]   = radius * cephes_cos(theta)
//        data[j+8] = radius * cephes_sin(theta)
//   4. 末尾处理（size % 16 != 0）：重新生成最后 16 个 uniform，覆盖 out[size-16:]
//
// 注意：CUDA 的 torch.randn 用 curand Philox4x32-10（算法不同），本实现对齐的是 CPU 路径。
// 验证脚本：tools/verify_rng_pytorch.py（Python 侧） + src/verify_rng.cpp（C++ 侧）。
#pragma once
#include <cstdint>
#include <cstring>
#include <random>
#include <cmath>

namespace rng {

// ==================== float 位操作辅助 ====================
inline uint32_t f2b(float x) {
    uint32_t b;
    std::memcpy(&b, &x, 4);
    return b;
}
inline float b2f(uint32_t b) {
    float x;
    std::memcpy(&x, &b, 4);
    return x;
}

// ==================== cephes log（float32，等价 log256_ps 标量）====================
inline float cephes_log(float x) {
    static const float SQRTHF = 0.707106781186547524f;
    static const float P[9] = {
        7.0376836292E-2f, -1.1514610310E-1f, 1.1676998740E-1f, -1.2420140846E-1f,
        1.4249322787E-1f, -1.6668057665E-1f, 2.0000714765E-1f, -2.4999993993E-1f,
        3.3333331174E-1f};
    static const float Q1 = -2.12194440e-4f;
    static const float Q2 = 0.693359375f;

    x = (x > b2f(0x00800000u)) ? x : b2f(0x00800000u);  // max(x, min_norm_pos)
    uint32_t xb = f2b(x);
    int imm0 = (int)(xb >> 23);              // 指数（含 bias 127）
    xb = (xb & 0x807fffffu) | 0x3f000000u;   // 清零指数，尾数归一到 [0.5,1)
    x = b2f(xb);
    imm0 -= 0x7f;
    float e = (float)imm0;
    e = e + 1.0f;
    bool mask = x < SQRTHF;
    float tmp = mask ? x : 0.0f;
    x = x - 1.0f;
    e = mask ? (e - 1.0f) : e;
    x = x + tmp;
    float z = x * x;
    float y = P[0];
    y = y * x + P[1];
    y = y * x + P[2];
    y = y * x + P[3];
    y = y * x + P[4];
    y = y * x + P[5];
    y = y * x + P[6];
    y = y * x + P[7];
    y = y * x + P[8];
    y = y * x;
    y = y * z;
    y = y + e * Q1;
    y = y - z * 0.5f;
    x = x + y;
    x = x + e * Q2;
    return x;
}

// ==================== cephes sincos（float32，等价 sincos256_ps 标量）====================
inline void cephes_sincos(float x, float& s, float& c) {
    static const float DP1 = -0.78515625f;
    static const float DP2 = -2.4187564849853515625e-4f;
    static const float DP3 = -3.77489497744594108e-8f;
    static const float SIN_P[3] = {-1.9515295891E-4f, 8.3321608736E-3f, -1.6666654611E-1f};
    static const float COS_P[3] = {2.443315711809948E-005f, -1.388731625493765E-003f, 4.166664568298827E-002f};
    static const float FOPI = 1.27323954473516f;

    uint32_t sign_bit_sin = f2b(x) & 0x80000000u;
    x = b2f(f2b(x) & 0x7fffffffu);   // 取绝对值
    float y = x * FOPI;
    int imm2 = (int)y;               // cvttps_epi32：向零截断（y>=0 等价 floor）
    imm2 = (imm2 + 1) & (~1);
    y = (float)imm2;
    int imm4 = imm2;
    uint32_t swap_sign_bit_sin = (uint32_t)((imm2 & 4) << 29);
    uint32_t poly_mask = ((imm2 & 2) == 0) ? 0xffffffffu : 0u;
    // magic pass：扩展精度模约减
    x = x + y * DP1;
    x = x + y * DP2;
    x = x + y * DP3;
    uint32_t sign_bit_cos = (uint32_t)(((~(imm4 - 2)) & 4) << 29);
    sign_bit_sin = sign_bit_sin ^ swap_sign_bit_sin;

    float z = x * x;
    // cos 多项式
    float yc = COS_P[0];
    yc = yc * z + COS_P[1];
    yc = yc * z + COS_P[2];
    yc = yc * z;
    yc = yc * z;
    yc = yc - z * 0.5f;
    yc = yc + 1.0f;
    // sin 多项式
    float ys = SIN_P[0];
    ys = ys * z + SIN_P[1];
    ys = ys * z + SIN_P[2];
    ys = ys * z;
    ys = ys * x;
    ys = ys + x;
    // 选择正确多项式 + 符号
    float ysin2 = b2f(f2b(ys) & poly_mask);
    float ysin1 = b2f(f2b(yc) & (~poly_mask));
    float y2b = ys - ysin2;
    float yb = yc - ysin1;
    float xmm1 = ysin1 + ysin2;
    float xmm2 = yb + y2b;
    s = b2f(f2b(xmm1) ^ sign_bit_sin);
    c = b2f(f2b(xmm2) ^ sign_bit_cos);
}

// ==================== Randn：逐位对齐 PyTorch CPU torch.randn ====================
class Randn {
public:
    explicit Randn(uint32_t seed) : gen_(seed) {}

    // 生成 n 个标准正态到 dst（精确复现 PyTorch normal_fill 的批量逻辑）
    void fill(float* dst, size_t n) {
        // 1. 生成 n 个 uniform 到 dst
        for (size_t i = 0; i < n; i++) dst[i] = uniform();
        // 2. 每 16 个做 Box-Muller（只处理完整块）
        for (size_t i = 0; i + 15 < n; i += 16) {
            box_muller_16(dst + i);
        }
        // 3. 末尾：重新生成最后 16 个 uniform，覆盖 dst[n-16:]
        if (n % 16 != 0) {
            float buf[16];
            for (int j = 0; j < 16; j++) buf[j] = uniform();
            box_muller_16(buf);
            std::memcpy(dst + n - 16, buf, 16 * sizeof(float));
        }
    }

private:
    float uniform() {
        uint32_t v = gen_();
        return (float)(v & 0xFFFFFFu) * (1.0f / 16777216.0f);
    }

    void box_muller_16(float* data) {
        static const float TWO_PI_F32 = 6.2831854820251465f;  // float32(2*pi)
        for (int j = 0; j < 8; j++) {
            float u1 = 1.0f - data[j];
            float u2 = data[j + 8];
            float radius = std::sqrt(-2.0f * cephes_log(u1));
            float theta = TWO_PI_F32 * u2;
            float s, c;
            cephes_sincos(theta, s, c);
            data[j] = radius * c;
            data[j + 8] = radius * s;
        }
    }

    std::mt19937 gen_;
};

// ==================== PhiloxRandn：对齐 PyTorch CUDA torch.randn ====================
//
// 复刻 torch CUDA normal_kernel（ATen/native/cuda/DistributionTemplates.h）在 float32 下的完整链路：
//   - curand_init(seed, subsequence=t, offset=0)：ctr 起点 = t<<64（skipahead_sequence），key=(seed,0)
//   - kernel 布局：block=256；grid=min(ceil(N/256), 36*(1536/256)=216)（RTX 5060 Ti: 36 SM）
//     thread t 第 r 次 curand_normal4 -> ctr = t<<64 + r（128-bit 小端，每调用 +1）
//     元素写位置 i = t + (4r+ii)*S, S=block*grid, ii=0..3
//   - _curand_box_muller（curand_normal.h）：u = x*2^-32 + 2^-33；v = y*(2π*2^-32) + (2π*2^-32)/2
//     s = sqrt(-2*log u)；输出 = sin(v)*s, cos(v)*s（GPU 用 __sincosf 快速近似，CPU 用标准 sinf/cosf，
//     与 GPU 最大偏差 ~2.3e-6（~19% 元素逐位一致），噪声间 cos≈0.999999999+，图像影响可忽略）
//   - N=16*H8*W8 等调用必须与官方 torch.randn 的 C-major 线性序一致（noise[c*H8W8 + h*W8 + w]）
//   - 局限：grid 上限依赖 GPU SM 数，本常量按 RTX 5060 Ti(36SM) 固化；不同 GPU 上官方序列布局不同
class PhiloxRandn {
public:
    explicit PhiloxRandn(uint32_t seed) : seed_(seed) {}

    // 生成 n 个标准正态到 dst（模拟 torch CUDA grid-stride normal4 kernel）
    void fill(float* dst, size_t n) {
        const uint64_t block = 256;
        uint64_t grid = (n + block - 1) / block;
        const uint64_t cap = 36 * (1536 / 256);   // RTX 5060 Ti: multiProcessorCount * (maxThreadsPerSM/256)
        if (grid > cap) grid = cap;
        const uint64_t S = block * grid;
        const uint64_t rounds = (n + 4 * S - 1) / (4 * S);
        for (uint64_t t = 0; t < S; t++) {
            uint32_t ctr = 0;  // 低 64 位 = r（r < 2^32 即够，rounds 很小）
            for (uint64_t r = 0; r < rounds; r++) {
                uint32_t out4[4];
                philox4((uint32_t)t, ctr, out4);
                float f[4];
                box_muller4(out4, f);
                uint64_t base = t + r * 4 * S;
                for (int ii = 0; ii < 4; ii++) {
                    uint64_t i = base + (uint64_t)ii * S;
                    if (i < n) dst[i] = f[ii];
                }
                ctr++;
            }
        }
    }

private:
    static inline uint32_t lo32(uint64_t v) { return (uint32_t)(v & 0xffffffffu); }
    static inline uint32_t hi32(uint64_t v) { return (uint32_t)(v >> 32); }

    // philox4x32-10：ctr = (r_lo, r_hi, t_lo, t_hi)（数值 t<<64 + r），key = (seed, 0)
    void philox4(uint32_t t, uint64_t r, uint32_t out[4]) const {
        uint32_t x = lo32(r), y = hi32(r), z = t, w = 0;
        uint32_t k0 = seed_, k1 = 0;
        for (int round = 0; round < 10; round++) {
            // M0*x, M1*z 的 64 位积拆高低
            uint64_t p0 = (uint64_t)PHILOX_M0 * (uint64_t)x;
            uint64_t p1 = (uint64_t)PHILOX_M1 * (uint64_t)z;
            uint32_t lo0 = lo32(p0), hi0 = hi32(p0);
            uint32_t lo1 = lo32(p1), hi1 = hi32(p1);
            uint32_t nx = hi1 ^ y ^ k0;
            uint32_t ny = lo1;
            uint32_t nz = hi0 ^ w ^ k1;
            uint32_t nw = lo0;
            x = nx; y = ny; z = nz; w = nw;
            k0 += PHILOX_W0; k1 += PHILOX_W1;
        }
        out[0] = x; out[1] = y; out[2] = z; out[3] = w;
    }

    static void box_muller4(const uint32_t u4[4], float f[4]) {
        // 与 curand 常量逐位一致（float32 位型）
        static const float K_2PI = b2f(0x30c90fdb);  // 2π*2^-32
        static const float H_2PI = b2f(0x30490fdb);  // (2π*2^-32)/2
        static const float INV   = b2f(0x2f800000);  // 2^-32
        static const float HINV  = b2f(0x2f000000);  // 2^-33
        for (int j = 0; j < 2; j++) {
            float u = (float)u4[2 * j] * INV + HINV;
            float v = (float)u4[2 * j + 1] * K_2PI + H_2PI;
            float s = std::sqrt(-2.0f * std::log(u));
            f[2 * j] = std::sin(v) * s;
            f[2 * j + 1] = std::cos(v) * s;
        }
    }

    static const uint32_t PHILOX_W0 = 0x9E3779B9u;
    static const uint32_t PHILOX_W1 = 0xBB67AE85u;
    static const uint32_t PHILOX_M0 = 0xD2511F53u;
    static const uint32_t PHILOX_M1 = 0xCD9E8D57u;

    uint32_t seed_;
};

} // namespace rng

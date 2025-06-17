
// The implementation is largely based on the implementation of Faiss.
// https://github.com/facebookresearch/faiss/wiki/Fast-accumulation-of-PQ-and-AQ-codes-(FastScan)
#pragma once
#include <immintrin.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <assert.h>
#include <heap.h>
#include <stdint.h>
#define lowbit(x) (x & (-x))
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

#if defined(_MSC_VER)
#include <nmmintrin.h>
#define POPCOUNT32(x) _popcnt_u32(x)
#else
// GCC/Clang
#define POPCOUNT32(x) __builtin_popcount(x)
#endif

using namespace std;

static inline uint32_t lane_sequence_from_mask(uint32_t mask_keep)
{
    uint32_t nibble_mask = _pdep_u32(mask_keep, 0x11111111u); // 0→0001, 1→0010...

    // 2) 再用 PEXT 把 lane 号 nibble 摘出来
    uint32_t seq = _pext_u32(0x76543210u, nibble_mask); // BMI2
    return seq;
}

template <std::size_t Align>
static inline std::size_t round_up_bytes(std::size_t bytes) noexcept
{
    return (bytes + Align - 1) & ~(Align - 1);
}

// inline void batch_floor_normalize(
//     const float *dist,
//     int32_t *output, // 输出为 int 数组
//     const float &low,
//     const float &delta // 除数
// )
// {
//     const __m256 v_delta = _mm256_set1_ps(delta);
//     const __m256 v_low = _mm256_set1_ps(low);
//     const __m256 v_zero = _mm256_setzero_ps();

//     for (int i = 0; i + 8 <= 32; i += 8)
//     {
//         // 1. 加载输入
//         __m256 v = _mm256_loadu_ps(&dist[i]);

//         // 2. 归一化：(v - low) / delta
//         v = _mm256_sub_ps(v, v_low);
//         v = _mm256_div_ps(v, v_delta);
//         v = _mm256_max_ps(v, v_zero);
//         // 3. 向下取整
//         v = _mm256_floor_ps(v);

//         // 4. 转换为 int 并存储
//         _mm256_storeu_si256((__m256i *)&output[i], _mm256_cvttps_epi32(v));
//     }
// }

// ==============================================================
// look up the tables for a packed batch of 32 quantization codes
// ==============================================================
template <uint32_t B>
inline void accumulate_one_block(const uint8_t *codes, const uint8_t *LUT, uint16_t *result)
// 32 * 4bit + 16 * 8bit
// 该函数用于处理一块数据（即一个批次）。它通过使用 AVX2 SIMD（单指令多数据）指令 来高效地操作数据
{
    // 似乎这个地方没有像PQ的fastscan一样，有一个early termination的操作，而是通过查出所有结果来计算bound
    __m256i low_mask = _mm256_set1_epi8(0xf);
    // _mm256_set1_epi8 是 AVX2（Advanced Vector Extensions 2） 指令集中的一个函数，用于创建一个 256 位的向量。
    // 具体来说，_mm256_set1_epi8 创建一个具有 32 个 8 位整数（即一个 256 位的整数向量）的向量，其中每个整数的值都设置为你传入的参数。
    // 该函数返回一个 256 位的向量（类型为 __m256i），其中包含 32 个 a 值。每个值都被存储为一个 8 位的整数。
    // 0xf 是一个 16 进制数（十六进制数），表示的值是 15，其二进制表示为 1111。
    __m256i accu[4];
    // accu[4]：声明了一个长度为 4 的 accu 数组，每个元素都是一个 __m256i 类型的向量。
    // 由于每个 __m256i 向量有 256 位，因此整个数组的大小为 4 * 256 = 1024 位。
    for (int i = 0; i < 4; i++)
    {
        accu[i] = _mm256_setzero_si256(); // 它的作用是将 accu[i] 中的每个元素都设置为零
    }

    constexpr uint32_t M = B / 4;

    for (int m = 0; m < M; m += 2)
    {
        __m256i c = _mm256_load_si256((__m256i const *)codes);
        __m256i lo = _mm256_and_si256(c, low_mask);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        __m256i lut = _mm256_load_si256((__m256i const *)LUT);

        __m256i res_lo = _mm256_shuffle_epi8(lut, lo);
        __m256i res_hi = _mm256_shuffle_epi8(lut, hi);

        accu[0] = _mm256_add_epi16(accu[0], res_lo);
        accu[1] = _mm256_add_epi16(accu[1], _mm256_srli_epi16(res_lo, 8));

        accu[2] = _mm256_add_epi16(accu[2], res_hi);
        accu[3] = _mm256_add_epi16(accu[3], _mm256_srli_epi16(res_hi, 8));

        codes += 32;
        LUT += 32;
    }

    accu[0] = _mm256_sub_epi16(accu[0], _mm256_slli_epi16(accu[1], 8));
    __m256i dis0 = _mm256_add_epi16(_mm256_permute2f128_si256(accu[0], accu[1], 0x21), _mm256_blend_epi32(accu[0], accu[1], 0xF0));
    _mm256_store_si256((__m256i *)(result + 0), dis0);

    accu[2] = _mm256_sub_epi16(accu[2], _mm256_slli_epi16(accu[3], 8));
    __m256i dis1 = _mm256_add_epi16(_mm256_permute2f128_si256(accu[2], accu[3], 0x21), _mm256_blend_epi32(accu[2], accu[3], 0xF0));
    _mm256_store_si256((__m256i *)(result + 16), dis1);
}

// ==============================================================
// look up the tables for all the packed batches
// ==============================================================
template <uint32_t B>
inline void accumulate(uint32_t nblk, const uint8_t *codes, const uint8_t *LUT, uint16_t *result)
{
    // nblk 表示本次调用要处理多少个“32-条样本”的批次（block）
    for (int i = 0; i < nblk; i++)
    {
        accumulate_one_block<B>(codes, LUT, result);
        codes += 32 * B / 8;
        result += 32;
    }
}

// ==============================================================
// prepare the look-up-table from the quantized query vector
// ==============================================================
template <uint32_t B>
inline void pack_LUT(uint8_t *byte_query, uint8_t *LUT)
{
    constexpr uint32_t M = B / 4;
    // 共有多少个 4‑维小组
    // LUT 把查询向量 每 4 维的 4‑bit 量化值 预先变成了一个 16 槽的小查表

    constexpr uint32_t pos[16] = {
        3 /*0000*/,
        3 /*0001*/,
        2 /*0010*/,
        3 /*0011*/,
        1 /*0100*/,
        3 /*0101*/,
        2 /*0110*/,
        3 /*0111*/,
        0 /*1000*/,
        3 /*1001*/,
        2 /*1010*/,
        3 /*1011*/,
        1 /*1100*/,
        3 /*1101*/,
        2 /*1110*/,
        3 /*1111*/,
    };
    /*
    pos 里存的是 对每个 4‑bit 掩码，最低位 1 出现在哪个位置。
    这是在用经典技巧：
    prefix[mask]= prefix[mask−lowbit(mask)]+q[lowbitPos]
    这里具像化为
    LUT[j] = LUT[j - lowbit(j)] + byte_query[pos[j]];

    其中lowbit(mask)：取掩码中最低位的 1
    pos[mask]：这 1 在 4 bit 中是第几维（0–3）
        mask   3210
        ------ ----
        0001 → 0001 lowbit=1  pos=0
        0010 → 0010 lowbit=2  pos=1
        ...
        0101 → 0001 lowbit=1  pos=0

        递推公式的直观解释
        使得 16 种前缀和只需 15 次简单相加，而不是每槽都扫 4 维。
    */
    for (int i = 0; i < M; i++)
    {
        LUT[0] = 0;
        for (int j = 1; j < 16; j++)
        {
            LUT[j] = LUT[j - lowbit(j)] + byte_query[pos[j]];
            // LUT[j - lowbit(j)] 这一定是“去掉最低位 1”后的那个子集的和；它在前面循环里已经算好，所以可直接复用。
            // 然后再把被擦掉那一维（由 pos[j] 指示）补进来，就完成了 j 的和值。
        }
        LUT += 16;
        byte_query += 4;
    }
}

template <typename T, class TA>
inline void get_matrix_column(T *src, size_t m, size_t n, int64_t i, int64_t j, TA &dest)
{
    for (int64_t k = 0; k < dest.size(); k++)
    {
        if (k + i >= 0 && k + i < m)
        {
            dest[k] = src[(k + i) * n + j];
        }
        else
        {
            dest[k] = 0;
        }
    }
}

// ==============================================================
// pack 32 quantization codes in a batch from the quantization
// codes represented by a sequence of uint8_t variables
// ==============================================================
template <uint32_t B>
void pack_codes(const uint8_t *codes, uint32_t ncode, uint8_t *blocks)
{

    uint32_t ncode_pad = (ncode + 31) / 32 * 32;
    constexpr uint32_t M = B / 4;
    const uint8_t bbs = 32;
    memset(blocks, 0, ncode_pad * M / 2);

    const uint8_t perm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
    uint8_t *codes2 = blocks;
    for (int blk = 0; blk < ncode_pad; blk += bbs)
    {
        // enumerate m
        for (int m = 0; m < M; m += 2)
        {
            std::array<uint8_t, 32> c, c0, c1;
            get_matrix_column(codes, ncode, M / 2, blk, m / 2, c);
            for (int j = 0; j < 32; j++)
            {
                c0[j] = c[j] & 15;
                c1[j] = c[j] >> 4;
            }
            for (int j = 0; j < 16; j++)
            {
                uint8_t d0, d1;
                d0 = c0[perm0[j]] | (c0[perm0[j] + 16] << 4);
                d1 = c1[perm0[j]] | (c1[perm0[j] + 16] << 4);
                codes2[j] = d0;
                codes2[j + 16] = d1;
            }
            codes2 += 32;
        }
    }
}

// ==============================================================
// pack 32 quantization codes in a batch from the quantization
// codes represented by a sequence of uint64_t variables
// ==============================================================
template <uint32_t B>
void pack_codes(const uint64_t *binary_code, uint32_t ncode, uint8_t *blocks)
{
    uint32_t ncode_pad = (ncode + 31) / 32 * 32;
    memset(blocks, 0, ncode_pad * sizeof(uint8_t));

    uint8_t *binary_code_8bit = new uint8_t[ncode_pad * B / 8];
    memcpy(binary_code_8bit, binary_code, ncode * B / 64 * sizeof(uint64_t));

    for (int i = 0; i < ncode; i++)
        for (int j = 0; j < B / 64; j++)
            for (int k = 0; k < 4; k++)
                swap(binary_code_8bit[i * B / 8 + 8 * j + k], binary_code_8bit[i * B / 8 + 8 * j + 8 - k - 1]);

    for (int i = 0; i < ncode * B / 8; i++)
    {
        uint8_t v = binary_code_8bit[i];
        uint8_t x = (v >> 4);
        uint8_t y = (v & 15);
        binary_code_8bit[i] = (y << 4 | x);
    }
    pack_codes<B>(binary_code_8bit, ncode, blocks);
    delete[] binary_code_8bit;
}

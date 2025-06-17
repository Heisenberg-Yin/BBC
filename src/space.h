
#pragma once
#include <queue>
#include <vector>
#include <algorithm>
#include <iostream>
#include <map>
#include <immintrin.h>
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#include "matrix.h"
#include "utils.h"
#include <random>

template <uint32_t D, uint32_t B>
class Space
{
public:
    // ================================================================================================
    // ********************
    //   Binary Operation
    // ********************
    inline static uint32_t popcount(u_int64_t *d);
    inline static uint32_t ip_bin_bin(uint64_t *q, uint64_t *d);
    inline static uint32_t ip_byte_bin(uint64_t *q, uint64_t *d);
    inline static void transpose_bin(uint8_t *q, uint64_t *tq);

    // ================================================================================================
    inline static void range(const float *q, const float *c, float &vl, float &vr);
    inline static void quantize(uint8_t *result, const float *q, const float *c, const float *u, float max_entry, float width, uint32_t &sum_q);
    inline static uint32_t sum(uint8_t *d);
    Space() {};
    ~Space() {};
};

// ==============================================================
// inner product between binary strings
// ==============================================================
template <uint32_t D, uint32_t B>
inline uint32_t Space<D, B>::ip_bin_bin(uint64_t *q, uint64_t *d)
{
    uint64_t ret = 0;
    for (int i = 0; i < B / 64; i++)
    {
        ret += __builtin_popcountll((*d) & (*q));
        q++;
        d++;
    }
    return ret;
}

// ==============================================================
// popcount (a.k.a, bitcount)
// ==============================================================
template <uint32_t D, uint32_t B>
inline uint32_t Space<D, B>::popcount(u_int64_t *d)
{
    uint64_t ret = 0;
    for (int i = 0; i < B / 64; i++)
    {
        ret += __builtin_popcountll((*d));
        d++;
    }
    return ret;
}

// ==============================================================
// inner product between a decomposed byte string q
// and a binary string d
// ==============================================================
template <uint32_t D, uint32_t B>
uint32_t Space<D, B>::ip_byte_bin(uint64_t *q, uint64_t *d)
{
    uint64_t ret = 0;
    for (int i = 0; i < B_QUERY; i++)
    {
        ret += (ip_bin_bin(q, d) << i);
        q += (B / 64);
    }
    return ret;
}

// ==============================================================
// decompose the quantized query vector into B_q binary vector
// ==============================================================

// transpose_bin() 把行方向的 4‑bit 量化码转置成列方向的 bit‑plane，并打包成 64‑bit 字，以便之后用 XOR + POPCNT 对整条查询与 64 条数据维并行比较——这是 RabitQ / FastScan 在 SCAN 模式下实现极高吞吐的关键步骤。
template <uint32_t D, uint32_t B>
void Space<D, B>::transpose_bin(uint8_t *q, uint64_t *tq)
{
    for (int i = 0; i < B; i += 32)
    {
        // 每次处理 32 维 每次搬运 32 个字节（32 维），因为 32 byte = 256 bit，正好填满一条 AVX2 寄存器
        // 把一个普通指针（如 uint8_t*）直接解释成 “指向 __m256i 的指针”，而 不做任何数据拷贝或转换。 这只是告诉编译器：“以后你把这块内存当成 256‑bit 向量来看”
        // __m256i	AVX2/AVX‑512 提供的 256‑bit 整数向量类型（32 byte）
        // _mm256_load_si256(const __m256i*)	对齐加载：把内存中 32 byte 读进 YMM 寄存器
        __m256i v = _mm256_load_si256(reinterpret_cast<__m256i *>(q));
        v = _mm256_slli_epi32(v, (8 - B_QUERY));
        /*
        按 32‑bit lane 做逻辑左移 (Shift‑Left Logical Immediate)
        先前用 _mm256_load_si256 读进来的 256 bit 向量，里面有 32 个字节；每个字节只用低 B_QUERY 位（例如 B_QUERY = 4 时，是一个 0‑15 的量化码）。
        立即数移位位数。若 B_QUERY = 4 就等于 4；若 B_QUERY = 5 就等于 3；总之就是 “把最高那一位推到第 7 位”。
        */
        for (int j = 0; j < B_QUERY; j++)
        {
            // 1️⃣  把 32 个字节的“最高位”抽成 32‑bit 掩码
            uint32_t v1 = _mm256_movemask_epi8(v);
            //      movemask 把字节 bit7 采样，结果 bit0←字节0、bit31←字节31
            // 2️⃣  把位顺序反转（想让“维 0”落在 uint64_t 的最高位）
            v1 = reverseBits(v1);
            //      reverseBits(b0b1...b31) -> b31...b1b0

            // 3️⃣  写入目标数组 tq
            //     ┌ plane 索引： (B_QUERY - j - 1)
            //     │               j=0 → plane3，j=3 → plane0
            //     ├ block 索引：  i/64
            //     │               一次处理 32 维，64 维为一块，所以同一块会被写两次
            //     └ = plane_idx * (B/64) + block_idx
            // (B_QUERY ‑ j ‑ 1) 把 循环变量 j（0 → B_QUERY‑1）翻转成 “bit‑plane 编号”，
            // 使得写进 tq[] 时的 plane 顺序从高位到低位。
            // plane 3 (权重 8) 存在 tq[0…]   ← 数组最前面
            // plane 2 (权重 4) 存在 tq[…]
            // plane 1 (权重 2) 存在 tq[…]
            // plane 0 (权重 1) 存在 tq[…]
            tq[(B_QUERY - j - 1) * (B / 64) + i / 64] |= ((uint64_t)v1 << ((i / 32 % 2 == 0) ? 32 : 0));
            //        偏移 32 or 0：  i=0 → 高 32 位；i=32 → 低 32 位
            //        用 |= 合并，保证同一个 64‑bit 槽先后写高/低两半
            // 4️⃣  v <<= 1  ：把下一个 bit‑plane 推到字节 MSB，为下一轮 movemask 做准备
            v = _mm256_add_epi32(v, v); // 对每个 32‑bit lane 乘 2 -> 等价于逻辑左移 1
        }
        q += 32; // 5️⃣  外层循环步进：指针跳到下一组 32 维（接下来 i += 32）
    }
}

// ==============================================================
// compute the min and max value of the entries of q
// ==============================================================
template <uint32_t D, uint32_t B>
void Space<D, B>::range(const float *q, const float *c, float &vl, float &vr)
{
    vl = +1e20;
    vr = -1e20;
    for (int i = 0; i < B; i++)
    {
        float tmp = (*q) - (*c);
        if (tmp < vl)
            vl = tmp;
        if (tmp > vr)
            vr = tmp;
        q++;
        c++;
    }
}

// ==============================================================
// quantize the query vector with uniform scalar quantization
// ==============================================================
template <uint32_t D, uint32_t B>
void Space<D, B>::quantize(uint8_t *result, const float *q, const float *c, const float *u, float vl, float width, uint32_t &sum_q)
{
    float one_over_width = 1.0 / width;
    uint8_t *ptr_res = result; // 0-255 1byte
    uint32_t sum = 0;
    for (int i = 0; i < B; i++)
    {
        (*ptr_res) = (uint8_t)((((*q) - (*c)) - vl) * one_over_width + (*u));
        sum += (*ptr_res);
        q++;
        c++;
        ptr_res++;
        u++;
    }
    sum_q = sum;
}

// The implementation is based on https://github.com/nmslib/hnswlib/blob/master/hnswlib/space_ip.h
template <uint32_t L>
inline float sqr_dist(const float *d, const float *q)
{
    float PORTABLE_ALIGN32 TmpRes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    constexpr uint32_t num_blk16 = L >> 4;
    constexpr uint32_t l = L & 0b1111;

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);
    for (int i = 0; i < num_blk16; i++)
    {
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    for (int i = 0; i < l / 8; i++)
    {
        v1 = _mm256_loadu_ps(d);
        v2 = _mm256_loadu_ps(q);
        d += 8;
        q += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    _mm256_store_ps(TmpRes, sum);

    float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    for (int i = 0; i < l % 8; i++)
    {
        float tmp = (*q) - (*d);
        ret += tmp * tmp;
        d++;
        q++;
    }
    return ret;
}
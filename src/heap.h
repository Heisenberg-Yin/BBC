#pragma once

#include <variant>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>
#include <memory.h>
#include <fast_scan.h>

struct PairBucket
{
    float *val = nullptr;    // 距离值
    uint32_t *idx = nullptr; // 对应ID
    uint32_t sz = 0;
    uint32_t cap = 0;

    void reserve(uint32_t new_cap)
    {
        if (new_cap <= cap)
            return; // 够用，直接返回

        // 1) 选择对齐粒度：64 B 适配 AVX-512，亦兼容 32 B
#if defined(__AVX512F__)
        constexpr uint8_t kAlign = 64;
#elif defined(__AVX2__)
        constexpr uint8_t kAlign = 32;
#endif

        // 2) 分配对齐内存（长度需为对齐粒度的整数倍）
        float *new_val = static_cast<float *>(
            std::aligned_alloc(kAlign, round_up_bytes<kAlign>(new_cap * sizeof(float))));
        uint32_t *new_idx = static_cast<uint32_t *>(
            std::aligned_alloc(kAlign, round_up_bytes<kAlign>(new_cap * sizeof(uint32_t))));

        // 3) SIMD 拷贝旧数据
#if defined(__AVX512F__)
        constexpr uint8_t kStep = 16; // 16 × 4 B = 64 B
        uint32_t i = 0;
        for (; i + kStep <= sz; i += kStep)
        {
            // val (float)
            __m512 v = _mm512_load_ps(val + i); // 已对齐，可用 aligned load
            _mm512_store_ps(new_val + i, v);    // 对齐 store

            // idx (uint32_t)
            __m512i vi = _mm512_load_si512(reinterpret_cast<const __m512i *>(idx + i));
            _mm512_store_si512(reinterpret_cast<__m512i *>(new_idx + i), vi);
        }
#elif defined(__AVX2__)
        constexpr uint8_t kStep = 8; // 8 × 4 B = 32 B
        uint32_t i = 0;
        for (; i + kStep <= sz; i += kStep)
        {
            __m256 v = _mm256_load_ps(val + i);
            _mm256_store_ps(new_val + i, v);

            __m256i vi = _mm256_load_si256(reinterpret_cast<const __m256i *>(idx + i));
            _mm256_store_si256(reinterpret_cast<__m256i *>(new_idx + i), vi);
        }
#endif
        const uint32_t tail = sz - i;
        if (tail)
        {
            std::memcpy(new_val + i, val + i, tail * sizeof(float));
            std::memcpy(new_idx + i, idx + i, tail * sizeof(uint32_t));
        }

        std::free(val);
        std::free(idx);

        // 4) 回写指针和容量
        val = new_val;
        idx = new_idx;
        cap = new_cap;
    }

    /* -------- 插入 -------- */
    inline void emplace(float v, uint32_t id_)
    {
        if (sz == cap)
            reserve(cap * 2);

        val[sz] = v;
        idx[sz] = id_;
        ++sz;
    }

    inline void emplace_batch(const float *src_val,
                              const uint32_t *src_idx,
                              uint8_t n)
    {
        if (sz + n > cap)
            reserve(std::max(cap * 2, sz + n));

        float *dst_val = val + sz;
        uint32_t *dst_idx = idx + sz;

#if defined(__AVX512F__)
        constexpr uint8_t kStep = 16;
        uint8_t i = 0;
        for (; i + kStep <= n; i += kStep)
        {
            // 浮点数据
            __m512 v = _mm512_loadu_ps(src_val + i);
            _mm512_store_ps(dst_val + i, v); // dst 已对齐，可用 store_ps

            // 整数索引
            __m512i vi = _mm512_loadu_si512(reinterpret_cast<const void *>(src_idx + i));
            _mm512_store_si512(reinterpret_cast<void *>(dst_idx + i), vi);
        }
#elif defined(__AVX2__) // ------- AVX2 (256-bit) -------
        constexpr uint8_t kStep = 8; // 8 × 32-bit = 256 bit
        uint8_t i = 0;
        for (; i + kStep <= n; i += kStep)
        {
            __m256 v = _mm256_loadu_ps(src_val + i);
            _mm256_store_ps(dst_val + i, v);

            __m256i vi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src_idx + i));
            _mm256_store_si256(reinterpret_cast<__m256i *>(dst_idx + i), vi);
        }
#else                   // ------- 标量回退 -------
        uint8_t i = 0;
#endif

        /* 处理剩余 < kStep 的元素（或无 SIMD 时所有元素） */
        for (uint8_t j = i; j < n; ++j)
        {
            dst_val[j] = src_val[j];
            dst_idx[j] = src_idx[j];
        }

        sz += n;
    }

    /* -------- 清空（复用内存） -------- */
    inline void clear() { sz = 0; }

    /* -------- 常用只读接口 -------- */
    uint32_t size() const { return sz; }
    bool empty() const { return sz == 0; }

    const float *val_data() const { return val; }
    const uint32_t *idx_data() const { return idx; }

    /* -------- 释放全部内存 -------- */
    void free_memory()
    {
        std::free(val);
        std::free(idx);
        val = nullptr;
        idx = nullptr;
        sz = cap = 0;
    }

    /* -------- 构/析/移动 -------- */
    PairBucket() = default;
    ~PairBucket() { free_memory(); }
};

class TopKBufferSoA
{
public:
    /* ---------- 常量与别名 ---------- */
#if defined(__AVX512F__)
    static constexpr uint8_t kAlign = 64;
#elif defined(__AVX2__)
    static constexpr uint8_t kAlign = 32;
#endif
    static constexpr uint8_t BlockSize = 64; // 顺序写缓冲条数

    /* ---------- 写缓冲块 ---------- */
    struct alignas(kAlign) BlockBuf
    {
        float dist[BlockSize];
        uint32_t id[BlockSize];
        uint8_t pos = 0; // 已写入元素计数
    };

    /* ---------- 构造 / 析构 ---------- */
    TopKBufferSoA(uint32_t k,
                  uint32_t physical_bucket_num,
                  uint32_t logical_bucket_num)
        : k_(k),
          logical_bucket_num_(logical_bucket_num),
          logical_threshold_bucket_id_(logical_bucket_num),
          physical_bucket_num_(physical_bucket_num),
          physical_threshold_bucket_id_(physical_bucket_num),
          bucket_list(physical_bucket_num),
          bucket_buffer_(physical_bucket_num)
    {
        /* LUT：logical→physical 桶映射，按 kAlign 对齐 */
        const uint32_t lut_bytes =
            round_up_bytes<kAlign>(logical_bucket_num_);
        code_lut_ = static_cast<uint8_t *>(
            std::aligned_alloc(kAlign, lut_bytes));

        /* 预留每个桶的 PairBuffer 容量（2 × K） */
        for (auto &B : bucket_list)
            B.reserve(k_ * 2);
    }

    /* 禁止拷贝，仅允许移动（必要时可自行实现） */
    TopKBufferSoA(const TopKBufferSoA &) = delete;
    TopKBufferSoA &operator=(const TopKBufferSoA &) = delete;

    ~TopKBufferSoA() { std::free(code_lut_); }

    uint8_t *get_code_lut() { return code_lut_; }

    /* 设置上下界（可在搜索开头调用） */
    void set_bounds(float lo, float up, float d)
    {
        lower_ = lo;
        upper_ = up;
        delta_ = d;
        logical_threshold_bucket_id_ = logical_bucket_num_;
        physical_threshold_bucket_id_ = physical_bucket_num_;
    }

    /* —— PUSH：写入一条 upper-bound 候选 —— */
    inline void push_upper(uint32_t b_logical, float lb, uint32_t id)
    {
        const uint32_t b = code_lut_[b_logical]; // physical bucket ID
        BlockBuf &blk = bucket_buffer_[b];

        /* 顺序写入 L1 缓冲块 */
        blk.dist[blk.pos] = lb;
        blk.id[blk.pos] = id;
        ++blk.pos;

        // #if defined(__GNUC__) || defined(__clang__)
        //         /* 预取下一行，减轻回写停顿（可选） */
        //         if (blk.pos + 16 < BlockSize)
        //             _mm_prefetch(reinterpret_cast<const char *>(blk.dist + blk.pos + 16),
        //                          _MM_HINT_T0);
        // #endif
        /* 块写满：批量写入 PairBucket */
        if (blk.pos == BlockSize)
        {
            bucket_list[b].emplace_batch(blk.dist, blk.id, BlockSize);
            blk.pos = 0;
        }
    }

    /* —— 全量 flush：将残余写回 PairBucket —— */
    void flush_all_upper()
    {
        for (uint32_t b = 0; b < physical_threshold_bucket_id_; ++b)
        {
            BlockBuf &blk = bucket_buffer_[b];
            if (blk.pos)
            {
                bucket_list[b].emplace_batch(blk.dist, blk.id, blk.pos);
                blk.pos = 0;
            }
        }
    }

    /* —— 清空全部数据 —— */
    void reset()
    {
        for (auto &B : bucket_list)
            B.clear();
        for (auto &blk : bucket_buffer_)
            blk.pos = 0;
        logical_threshold_bucket_id_ = logical_bucket_num_;
        physical_threshold_bucket_id_ = physical_bucket_num_;
    }

    /* —— 阈值更新：找到逻辑/物理阈值桶 —— */
    void update_th_code()
    {
        uint32_t acc = 0, j = 0; // 已累积元素/逻辑桶游标
        for (uint32_t i = 0; i < physical_bucket_num_; ++i)
        {
            acc += static_cast<uint32_t>(bucket_list[i].size());
            while (j < logical_bucket_num_ && code_lut_[j] == i)
                ++j;

            if (acc >= k_)
            {
                logical_threshold_bucket_id_ = j;      // 逻辑阈值桶 (pseudo)
                physical_threshold_bucket_id_ = i + 1; // 物理阈值桶 (exclusive)
                return;
            }
        }
        /* 如果不足 k，则阈值指向最右端 */
        logical_threshold_bucket_id_ = logical_bucket_num_;
        physical_threshold_bucket_id_ = physical_bucket_num_;
    }

    /* ---------- getters ---------- */
    uint32_t get_logical_bucket_num() const { return logical_bucket_num_; }
    uint32_t get_physical_bucket_num() const { return physical_bucket_num_; }
    uint32_t get_logical_threshold_bucket_id() const { return logical_threshold_bucket_id_; }
    uint32_t get_physical_threshold_bucket_id() const { return physical_threshold_bucket_id_; }
    float get_delta() const { return delta_; }
    float get_lower() const { return lower_; }
    float get_upper() const { return upper_; }
    auto &get_upper_buffer() { return bucket_list; } // 供外部遍历

private:
    /* ---------- 成员变量 ---------- */
    uint32_t k_;                            // top-k
    uint32_t logical_bucket_num_;           // 逻辑桶总数
    uint32_t logical_threshold_bucket_id_;  // 逻辑阈值桶 (pseudo)
    uint32_t physical_bucket_num_;          // 物理桶总数
    uint32_t physical_threshold_bucket_id_; // 物理阈值桶 (exclusive)

    float lower_{}, upper_{}, delta_{}; // 搜索上下界 & 步长

    alignas(kAlign) uint8_t *code_lut_;   // logical→physical LUT
    std::vector<PairBucket> bucket_list;  // 每个物理桶的 PairBucket
    std::vector<BlockBuf> bucket_buffer_; // 写缓冲块
};

#pragma once

#include <variant>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>
#include <memory.h>

struct PairBucket
{
    float *val = nullptr;    // 距离值
    uint32_t *idx = nullptr; // 对应ID
    std::size_t sz = 0;
    std::size_t cap = 0;

    void reserve(std::size_t new_cap)
    {
        if (new_cap <= cap)
            return; // 够用，直接返回

        float *new_val = static_cast<float *>(std::aligned_alloc(32, new_cap * sizeof(float)));
        uint32_t *new_idx = static_cast<uint32_t *>(std::aligned_alloc(32, new_cap * sizeof(uint32_t)));

        std::memcpy(new_val, val, sz * sizeof(float));
        std::memcpy(new_idx, idx, sz * sizeof(uint32_t));

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

        /* SIMD 拷贝 8 条/批 */
        uint8_t i = 0;
        for (; i + 8 <= n; i += 8)
        {
            _mm256_store_ps(dst_val + i, _mm256_loadu_ps(src_val + i));
            _mm256_store_si256(reinterpret_cast<__m256i *>(dst_idx + i),
                               _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src_idx + i)));
        }
        // 处理余数
        for (; i < n; ++i)
        {
            dst_val[i] = src_val[i];
            dst_idx[i] = src_idx[i];
        }

        sz += n;
    }

    /* -------- 清空（复用内存） -------- */
    inline void clear() { sz = 0; }

    /* -------- 常用只读接口 -------- */
    std::size_t size() const { return sz; }
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
    TopKBufferSoA(uint32_t k,                   // top-k
                  uint32_t physical_bucket_num, // 物理桶数
                  uint32_t logical_bucket_num)  // 桶数
        : k_(k), physical_bucket_num_(physical_bucket_num), logical_bucket_num_(logical_bucket_num)
    {
        bucketed_upper_buffer_.resize(physical_bucket_num_);
        // bucketed_lower_buffer_.resize(physical_bucket_num_);
        // bucketed_exact_buffer_.resize(physical_bucket_num_);
        tmp_upper_.resize(physical_bucket_num_); // ← add
        code_lut_ = new PORTABLE_ALIGN64 uint8_t[logical_bucket_num_];
        for (uint32_t i = 0; i < physical_bucket_num_; ++i)
        {
            bucketed_upper_buffer_[i].reserve(k_ * 2);
            // bucketed_lower_buffer_[i].reserve(k_ * 2);
            // bucketed_exact_buffer_[i].reserve(k_ * 2);
        }
        logical_threshold_bucket_id_ = logical_bucket_num_;   // 最右桶
        physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
    }

    uint8_t *get_code_lut()
    {
        return code_lut_;
    }

    void set_bounds(float lowest, float upper, float delta) // ← 多一个阈值数组
    {
        lower_ = lowest;
        upper_ = upper;
        delta_ = delta;
        logical_threshold_bucket_id_ = logical_bucket_num_;   // 最右桶
        physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
    }

    /* ---------- push API ---------- */
    // inline void push_upper(uint32_t b, KeyType ub, KeyType lb, DataType lb_code, DataType id)
    // {
    //     bucketed_upper_buffer_[b].emplace(ub, lb, lb_code, id);
    // }

    // // 假设 KeyType 对应 float, DataType 对应 uint32_t/int32_t
    // inline void push_upper_batch(uint32_t b,
    //                              const KeyType *ub_arr,
    //                              const KeyType *lb_arr,
    //                              const DataType *lb_code_arr,
    //                              const DataType *id_arr,
    //                              std::size_t n)
    // {
    //     // 直接批量插入 n 条
    //     bucketed_upper_buffer_[b].emplace_batch(
    //         ub_arr,
    //         lb_arr,
    //         reinterpret_cast<const uint32_t *>(lb_code_arr),
    //         reinterpret_cast<const uint32_t *>(id_arr),
    //         n);
    // }

    // inline void push_upper(uint32_t b, KeyType lb, DataType id)
    // {
    //     bucketed_upper_buffer_[code_lut_[b]].emplace(lb, id);
    // }

    inline void push_upper(uint32_t b_logical, float lb, uint32_t id)
    {
        const uint32_t b = code_lut_[b_logical]; // physical bucket id
        BlockBuf32 &blk = tmp_upper_[b];

        // ① 写入缓冲（顺序写，L1 命中）
        blk.dist[blk.pos] = lb;
        blk.id[blk.pos] = id;
        ++blk.pos;

        if (blk.pos == BlockSize)
        {
            PairBucket &bucket = bucketed_upper_buffer_[b];
            bucket.emplace_batch(blk.dist, blk.id, BlockSize); // 批量写入
            blk.pos = 0;                                       // 清空块
        }
    }

    // inline void push_lower(uint32_t b, KeyType dist, DataType id)
    // {
    //     bucketed_lower_buffer_[code_lut_[b]].emplace(dist, id);
    // }

    // inline void push_exact(uint32_t b, KeyType dist, DataType id)
    // {
    //     bucketed_exact_buffer_[code_lut_[b]].emplace(dist, id);
    // }

    void reset()
    {
        for (auto &B : bucketed_upper_buffer_)
            B.clear();
        // for (auto &B : bucketed_lower_buffer_)
        //     B.clear();
        // for (auto &B : bucketed_exact_buffer_)
        //     B.clear();

        logical_threshold_bucket_id_ = logical_bucket_num_;
        physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
    }

    void update_th_code()
    {
        uint32_t acc = 0;
        uint32_t j = 0;
        for (uint32_t i = 0; i < physical_bucket_num_; ++i)
        {
            acc += static_cast<uint32_t>(bucketed_upper_buffer_[i].size());
            while (j < logical_bucket_num_ && code_lut_[j] == i)
            {
                ++j;
            }
            if (acc >= k_)
            {                                          // 一旦达到 k_
                logical_threshold_bucket_id_ = j;      // 当前 pseudo id 即阈值
                physical_threshold_bucket_id_ = i + 1; // 当前物理桶 id
                return;
            }
        }
        logical_threshold_bucket_id_ = logical_bucket_num_;
        physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
    }

    void update_th_code_with_exact()
    {
        uint32_t acc = 0;
        uint32_t j = 0;
        for (uint32_t i = 0; i < physical_bucket_num_; ++i)
        {
            acc += static_cast<uint32_t>(bucketed_upper_buffer_[i].size() + bucketed_exact_buffer_[i].size());
            while (j < logical_bucket_num_ && code_lut_[j] == i)
            {
                ++j;
            }
            if (acc >= k_)
            {                                          // 一旦达到 k_
                logical_threshold_bucket_id_ = j;      // 当前 pseudo id 即阈值
                physical_threshold_bucket_id_ = i + 1; // 当前物理桶 id
                return;
            }
        }
        logical_threshold_bucket_id_ = logical_bucket_num_;
        physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
    }

    void flush_all_upper()
    {
        for (uint32_t b = 0; b < physical_threshold_bucket_id_; ++b)
        {
            BlockBuf32 &blk = tmp_upper_[b];
            if (blk.pos == 0)
                continue;

            PairBucket &bucket = bucketed_upper_buffer_[b];
            bucket.emplace_batch(blk.dist, blk.id, blk.pos); // 批量写入
            blk.pos = 0;
        }
    }

    uint32_t get_logical_bucket_num() const { return logical_bucket_num_; }
    uint32_t get_physical_bucket_num() const { return physical_bucket_num_; }
    uint32_t get_logical_threshold_bucket_id() const { return logical_threshold_bucket_id_; }
    uint32_t get_physical_threshold_bucket_id() const { return physical_threshold_bucket_id_; }

    float get_delta() const { return delta_; }
    float get_lower() const { return lower_; }
    float get_upper() const { return upper_; }

    auto &get_upper_buffer() { return bucketed_upper_buffer_; }
    // auto &get_lower_buffer() { return bucketed_lower_buffer_; }
    // auto &get_exact_buffer() { return bucketed_exact_buffer_; }

private:
    uint32_t k_;
    uint32_t logical_bucket_num_;
    uint32_t logical_threshold_bucket_id_;
    uint32_t physical_bucket_num_;
    uint32_t physical_threshold_bucket_id_;

    float lower_{}, upper_{}, delta_{};
    PORTABLE_ALIGN64 uint8_t *code_lut_;
    std::vector<PairBucket> bucketed_upper_buffer_;
    // std::vector<PairBucket> bucketed_lower_buffer_;
    // std::vector<PairBucket> bucketed_exact_buffer_;
    constexpr static uint8_t BlockSize = 64; // 每个块的大小
    struct BlockBuf32
    {
        alignas(32) float dist[BlockSize];
        alignas(32) uint32_t id[BlockSize];
        uint8_t pos = 0; // 已写条数
    };
    std::vector<BlockBuf32> tmp_upper_; // size = physical_bucket_num_

    /* 临时批量缓冲：只在 early_stage_ 使用 */
    std::vector<float> stage_dist_;
};

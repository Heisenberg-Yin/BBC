#pragma once
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <matrix.hpp>
#include <utils.hpp>
#include <queue>

typedef std::pair<float, uint32_t> Result;
typedef std::priority_queue<Result> ResultHeap;

class IVF_PQFastScan
{
public:
    // 核心参数
    uint32_t N; // 向量总数
    uint32_t D; // 原始维度
    uint32_t C; // 簇数
    uint32_t M;
    uint32_t sub_dim;
    uint32_t centroid_per_sub;
    uint32_t pq_code_size; // 每条PQ码长度（字节）
    uint32_t nprobe;       // 查询时使用的簇数

    // 成员变量：全部换为裸指针
    uint32_t *start = nullptr; // 每簇起始
    uint32_t *len = nullptr;   // 每簇长度
    uint32_t *id = nullptr;    // 向量全局ID（重排后）
    uint32_t *packed_start = nullptr;
    float min_val = 0;
    float max_val = 0;
    float scale = 0;
    float factor = 0;
    float *centroid = nullptr;
    float *data = nullptr;
    float *residual = nullptr;
    float *pq_cookbook = nullptr;
    float *LUT_float = nullptr;
    uint8_t *LUT_uint8 = nullptr;
    uint8_t *pq_codes; // 所有PQ编码，64字节对齐
    uint8_t *packed_code = nullptr;
    IVF_PQFastScan();
    IVF_PQFastScan(unsigned n,
                   unsigned M,
                   unsigned ksub,
                   unsigned code_size,
                   const Matrix<float> &centroid_mat,
                   const float *base_data,
                   const uint8_t *pq_code_mat,
                   const unsigned *assign,
                   const float *cookbook);

    ~IVF_PQFastScan();

    void compute_pq_LUT(float *query);
    void fast_scan(ResultHeap &KNNs, float &dist_to_centroid, float &distK, uint32_t k,
        uint8_t *packed_code, uint32_t len, float *query, float *data, uint32_t start_id);

    ResultHeap search(float *query, float *rd_query, uint32_t num_cand, uint32_t k, uint32_t nprobe, float distK = std::numeric_limits<float>::max());

    void rotate_centroid(const Matrix<float> &opq);
    // 存盘/加载
    void save(char *filename);
    void load(char *filename);
};

IVF_PQFastScan::IVF_PQFastScan()
    : N(0), D(0), C(0), M(0), sub_dim(0), centroid_per_sub(0), pq_code_size(0), nprobe(0),
      start(nullptr), len(nullptr), id(nullptr),
      centroid(nullptr), data(nullptr), residual(nullptr),
      pq_cookbook(nullptr), pq_codes(nullptr)
{
}
void IVF_PQFastScan::rotate_centroid(const Matrix<float> &opq)
{
    // 1. 将裸指针数据包装为 Matrix<float>
    Matrix<float> cent(C, D);
    memcpy(cent.data, centroid, sizeof(float) * C * D);

    // 2. 右乘 OPQ
    Matrix<float> new_cent = mul(cent, opq); // (C, D) * (D, D') -> (C, D')

    // 3. 检查维度兼容
    assert(new_cent.n == C);
    assert(new_cent.d == D);

    // 4. 写回类成员 centroid
    memcpy(centroid, new_cent.data, sizeof(float) * C * D);
}

IVF_PQFastScan::IVF_PQFastScan(
    unsigned n,
    unsigned m,
    unsigned ksub,
    unsigned code_size,
    const Matrix<float> &centroid_mat, // C × D
    const float *base_data,
    const uint8_t *pq_code_mat, // N × pq_code_size
    const unsigned *assign,     // N
    const float *cookbook)
{
    N = n;
    M = m;
    centroid_per_sub = ksub;
    pq_code_size = code_size;
    D = centroid_mat.d;
    C = centroid_mat.n;
    sub_dim = D / M;

    std::cerr << "M: " << M << "ksub: " << ksub << "dsub: " << sub_dim << std::endl;

    // 1. 分配长度指针
    start = new uint32_t[C];
    len = new uint32_t[C];
    id = new uint32_t[N];

    // 2. 统计每簇数量
    memset(len, 0, C * sizeof(uint32_t));
    for (uint32_t i = 0; i < N; ++i)
        len[assign[i]]++;

    // 3. 计算每簇起始位置
    int sum = 0;
    for (uint32_t i = 0; i < C; ++i)
    {
        start[i] = sum;
        sum += len[i];
    }

    for (int i = 0; i < N; i++)
    {
        id[start[assign[i]]] = i;
        start[assign[i]]++;
    }

    for (int i = 0; i < C; i++)
    {
        start[i] -= len[i];
    }

    centroid = new float[C * D];
    data = new float[1ull * N * D];
    pq_codes = new uint8_t[1ull * N * pq_code_size]; // 临时重排buffer
    pq_cookbook = new float[M * centroid_per_sub * sub_dim];
    std::memcpy(centroid, centroid_mat.data, C * D * sizeof(float));
    std::memcpy(pq_cookbook, cookbook, M * centroid_per_sub * sub_dim * sizeof(float));

    uint8_t *code_ptr = pq_codes;
    float *data_ptr = data;

    for (int i = 0; i < N; i++)
    {
        int x = id[i];
        std::memcpy(data_ptr, base_data + 1ull * x * D, D * sizeof(float));
        std::memcpy(code_ptr, pq_code_mat + 1ull * x * pq_code_size, pq_code_size * sizeof(uint8_t));
        data_ptr += D;
        code_ptr += pq_code_size;
    }
};

void IVF_PQFastScan::compute_pq_LUT(
     float* query
) {
    for (int sq = 0; sq < M; ++sq) {
         float* q_sub = query + sq * sub_dim;
        __m128 qv = _mm_loadu_ps(q_sub);           // 4 floats

        for (int c = 0; c < centroid_per_sub; ++c) {
             float* cent = pq_cookbook + (sq * centroid_per_sub + c) * sub_dim;
            __m128 cv   = _mm_loadu_ps(cent);
            __m128 diff = _mm_sub_ps(qv, cv);
            __m128 mul  = _mm_mul_ps(diff, diff);
            float dist2 = hsum4(mul);
            LUT_float[sq * centroid_per_sub + c] = dist2;
        }
    }
    // 2. 找min/max
    int total = M * centroid_per_sub;
    min_val = LUT_float[0];
    max_val = LUT_float[0];
    for (int i = 0; i < total; ++i) {
        if (LUT_float[i] < min_val) min_val = LUT_float[i];
        if (LUT_float[i] > max_val) max_val = LUT_float[i];
    }
    scale = (max_val > min_val) ? (255.f / (max_val - min_val)) : 1.f;  // 防止除0
    for (int i = 0; i < total; ++i) {
        float norm = (LUT_float[i] - min_val) * scale;
        if (norm < 0) norm = 0;
        if (norm > 255) norm = 255;
        LUT_uint8[i] = static_cast<uint8_t>(norm); // 四舍五入
    }
    factor = 1.0f / scale;
}

void IVF_PQFastScan::fast_scan(ResultHeap &KNNs, float &dist_to_centroid,  float &distK, uint32_t k,
                          uint8_t *packed_code, uint32_t len, float *query, float *data, uint32_t start_id)
{

    constexpr uint32_t SIZE = 32;
    uint32_t it = len / SIZE;
    uint32_t remain = len - it * SIZE;
    uint32_t nblk_remain = (remain + 31) / 32;
    uint32_t local_id = start_id;
    while (it--)
    {
        uint16_t PORTABLE_ALIGN32 result[SIZE];
        accumulate((SIZE / 32), packed_code, LUT_uint8, result, pq_code_size);
        packed_code += SIZE * pq_code_size;

        for (int i = 0; i < SIZE; i++)
        {
            float dist = result[i]*factor + min_val * M;
            if (dist < distK)
            {
                KNNs.emplace(dist, local_id);
                if (KNNs.size() > k)
                    KNNs.pop();
                if (KNNs.size() == k)
                    distK = KNNs.top().first;
            }
            local_id = local_id + 1;
            data += D;  
        }
    }

    {
        uint16_t PORTABLE_ALIGN32 result[SIZE];
        accumulate(nblk_remain, packed_code, LUT_uint8, result, pq_code_size);

        for (int i = 0; i < remain; i++)
        {
            float dist = result[i]*factor+ min_val * M;
            if (dist < distK)
            {
                KNNs.emplace(dist, local_id);
                if (KNNs.size() > k)
                    KNNs.pop();
                if (KNNs.size() == k)
                    distK = KNNs.top().first;
            }
            local_id = local_id + 1;
            data += D;
        }
    }
}

ResultHeap IVF_PQFastScan::search(float *query, float *rd_query, uint32_t num_cand, uint32_t k, uint32_t nprobe, float distK) 
{
    ResultHeap KNNs;
    Result centroid_dist[C];
    float *ptr_c = centroid;
    float *ptr_residual = residual;
    for (int i = 0; i < C; i++)
    {
        centroid_dist[i].first = compute_sub(rd_query, ptr_c, ptr_residual, D);
        centroid_dist[i].second = i;
        ptr_c += D;
        ptr_residual += D;
    }
    std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

    // ===========================================================================================================
    // Scan the first nprobe clusters.
    Result *ptr_centroid_dist = (&centroid_dist[0]);
    ptr_residual = residual;

    for (int pb = 0; pb < nprobe; pb++)
    {
        uint32_t c = ptr_centroid_dist->second;
        float dist_to_centroid = ptr_centroid_dist->first;
        float *ptr_data = data + 1ull * start[c] * D;
        ptr_residual = residual + 1ull * c * D;
        uint8_t *ptr_packed_code = packed_code + packed_start[c];
        compute_pq_LUT(ptr_residual);
        fast_scan(KNNs, dist_to_centroid, distK, num_cand, ptr_packed_code, len[c], query, ptr_data, start[c]);    
        ptr_centroid_dist++;
    }
    ResultHeap result;
    distK = std::numeric_limits<float>::max();
    uint32_t CL = (D * sizeof(float) + 63) / 64; // 每条向量多少行 64B

    while (!KNNs.empty())
    {   
        uint32_t idx = KNNs.top().second;
        KNNs.pop();
        if (!KNNs.empty())
        {
            uint32_t future_idx = KNNs.top().second;
            const char *addr = reinterpret_cast<const char *>(
                data + static_cast<size_t>(future_idx) * D);        
            mem_prefetch_l1(addr, CL);
        }
        float dist = compute_l2_distance(query, data + 1ull * idx * D, D);
        if (dist < distK)   
        {
            result.emplace(dist, *(id + idx));
            if (result.size() > k)
                result.pop();
            if (result.size() == k)
                distK = result.top().first;
        }
    }

    return result;
}


void IVF_PQFastScan::load(char *filename)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open())
    {
        throw std::runtime_error("Cannot open file");
    }
    input.read((char *)&N, sizeof(uint32_t));
    input.read((char *)&D, sizeof(uint32_t));
    input.read((char *)&C, sizeof(uint32_t));
    input.read((char *)&M, sizeof(uint32_t));
    input.read((char *)&centroid_per_sub, sizeof(uint32_t));
    input.read((char *)&pq_code_size, sizeof(uint32_t));
    sub_dim = D / M;

    std::cerr << "M: " << M << " ksub: " << centroid_per_sub << " dsub: " << sub_dim << std::endl;

    start = new uint32_t[C];
    len = new uint32_t[C];
    id = new uint32_t[N];

    input.read((char *)start, C * sizeof(uint32_t));
    input.read((char *)len, C * sizeof(uint32_t));
    input.read((char *)id, N * sizeof(uint32_t));

    centroid = new float[C * D];
    residual = new float[C * D];
    pq_cookbook = new float[M * centroid_per_sub * sub_dim];
    data = new float[1ull * N * D];

    input.read((char *)centroid, C * D * sizeof(float));
    input.read((char *)pq_cookbook, M * centroid_per_sub * sub_dim * sizeof(float));
    input.read((char *)data, 1ull * N * D * sizeof(float));
    
    pq_codes = static_cast<uint8_t *>(aligned_alloc(64, 1ull * N * pq_code_size * sizeof(uint8_t)));

    input.read((char *)pq_codes, 1ull * N * pq_code_size * sizeof(uint8_t));
    input.close();
    LUT_float = (float*)aligned_alloc(32, sizeof(float) * M * centroid_per_sub);
    LUT_uint8 = (uint8_t*)aligned_alloc(32, sizeof(uint8_t) * M * centroid_per_sub);
    packed_start = new uint32_t[C];
    uint32_t cur = 0;
    for (int i = 0; i < C; i++)
    {
        packed_start[i] = cur;
        cur += (len[i] + 31) / 32 * 32 * pq_code_size;
    }
    packed_code = static_cast<uint8_t *>(aligned_alloc(32, cur * sizeof(uint8_t)));
    for (int i = 0; i < C; i++)
    {
        pack_codes(pq_codes + 1ull * start[i] * pq_code_size, len[i], packed_code + packed_start[i], pq_code_size*8);
    }
    std::cerr << "loaded!" << std::endl;
}

void IVF_PQFastScan::save(char *filename)
{
    std::ofstream output(filename, std::ios::binary);

    output.write((char *)&N, sizeof(uint32_t));
    output.write((char *)&D, sizeof(uint32_t));
    output.write((char *)&C, sizeof(uint32_t));
    output.write((char *)&M, sizeof(uint32_t));
    output.write((char *)&centroid_per_sub, sizeof(uint32_t));
    output.write((char *)&pq_code_size, sizeof(uint32_t));

    output.write((char *)start, C * sizeof(uint32_t));
    output.write((char *)len, C * sizeof(uint32_t));
    output.write((char *)id, N * sizeof(uint32_t));

    output.write((char *)centroid, C * D * sizeof(float));
    output.write((char *)pq_cookbook, M * centroid_per_sub * sub_dim * sizeof(float));

    output.write((char *)data, 1ull * N * D * sizeof(float));
    output.write((char *)pq_codes, 1ull * N * pq_code_size * sizeof(uint8_t));

    output.close();
    std::cerr << "Saved!" << std::endl;
}

IVF_PQFastScan::~IVF_PQFastScan()
{
    if (start)
        delete[] start;
    if (len)
        delete[] len;
    if (id)
        delete[] id;
    if (centroid)
        delete[] centroid;
    if (data)
        delete[] data;
    if (residual)
        delete[] residual;
    if (pq_cookbook)
        delete[] pq_cookbook;
    if (pq_codes)
        std::free(pq_codes);
};
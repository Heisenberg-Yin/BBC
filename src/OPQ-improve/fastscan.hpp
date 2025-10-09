#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <matrix.hpp>
#include <queue>
#include <utils.hpp>
#include <vector>
uint32_t rerank_count;
typedef std::pair<float, uint32_t> Result;
typedef std::priority_queue<Result> ResultHeap;

class IVF_PQFastScan {
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
  float *centroid = nullptr;
  float *data = nullptr;
  float *residual = nullptr;
  float *pq_cookbook = nullptr;
  std::vector<ProbeInfo> probe_infos;
  uint8_t *pq_codes; // 所有PQ编码，64字节对齐
  uint8_t *packed_code = nullptr;
  IVF_PQFastScan();
  IVF_PQFastScan(unsigned n, unsigned M, unsigned ksub, unsigned code_size,
                 const Matrix<float> &centroid_mat, const float *base_data,
                 const uint8_t *pq_code_mat, const unsigned *assign,
                 const float *cookbook);

  ~IVF_PQFastScan();
  std::vector<std::pair<float, uint32_t>>
  improved_search(float *query, float *rd_query, uint32_t num_cand, uint32_t k,
                  uint32_t nprobe, TopKBufferSoA &KNNs);

  void compute_pq_LUT(float *query, float *LUT_float, uint8_t *LUT_uint8,
                      float &min_val, float &max_val, float &scale,
                      float &factor);

  void rerank(float *query, float *data, uint32_t D, uint32_t num_cand,
              uint32_t k, TopKBufferSoA &KNNs,
              std::vector<std::pair<float, uint32_t>> &result_ids_dist);

  void fast_scan(TopKBufferSoA &KNNs, uint8_t *LUT_uint8, float &factor,
                 float &min_val, uint32_t k, uint8_t *packed_code, uint32_t len,
                 float *query, float *data, uint32_t start_id);

  void build_codebook_from_samples(std::vector<ProbeInfo> &probe_infos,
                                   uint32_t nprobe, uint32_t k,
                                   TopKBufferSoA &KNNs);

  void sample_fast_scan(float *sample_dist, float &samllest_dist,
                        float &largest_dist, uint8_t *LUT_uint8,
                        uint8_t *packed_code, float &factor, float &min_val,
                        uint32_t len);

  void rotate_centroid(const Matrix<float> &opq);
  // 存盘/加载
  void save(char *filename);
  void load(char *filename);
};

IVF_PQFastScan::IVF_PQFastScan()
    : N(0), D(0), C(0), M(0), sub_dim(0), centroid_per_sub(0), pq_code_size(0),
      nprobe(0), start(nullptr), len(nullptr), id(nullptr), centroid(nullptr),
      data(nullptr), residual(nullptr), pq_cookbook(nullptr),
      pq_codes(nullptr) {}
void IVF_PQFastScan::rotate_centroid(const Matrix<float> &opq) {
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

IVF_PQFastScan::IVF_PQFastScan(unsigned n, unsigned m, unsigned ksub,
                               unsigned code_size,
                               const Matrix<float> &centroid_mat, // C × D
                               const float *base_data,
                               const uint8_t *pq_code_mat, // N × pq_code_size
                               const unsigned *assign,     // N
                               const float *cookbook) {
  N = n;
  M = m;
  centroid_per_sub = ksub;
  pq_code_size = code_size;
  D = centroid_mat.d;
  C = centroid_mat.n;
  sub_dim = D / M;

  std::cerr << "M: " << M << "ksub: " << ksub << "dsub: " << sub_dim
            << std::endl;

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
  for (uint32_t i = 0; i < C; ++i) {
    start[i] = sum;
    sum += len[i];
  }

  for (int i = 0; i < N; i++) {
    id[start[assign[i]]] = i;
    start[assign[i]]++;
  }

  for (int i = 0; i < C; i++) {
    start[i] -= len[i];
  }

  centroid = new float[C * D];
  data = new float[1ull * N * D];
  pq_codes = new uint8_t[1ull * N * pq_code_size]; // 临时重排buffer
  pq_cookbook = new float[M * centroid_per_sub * sub_dim];
  std::memcpy(centroid, centroid_mat.data, C * D * sizeof(float));
  std::memcpy(pq_cookbook, cookbook,
              M * centroid_per_sub * sub_dim * sizeof(float));

  uint8_t *code_ptr = pq_codes;
  float *data_ptr = data;

  for (int i = 0; i < N; i++) {
    int x = id[i];
    std::memcpy(data_ptr, base_data + 1ull * x * D, D * sizeof(float));
    std::memcpy(code_ptr, pq_code_mat + 1ull * x * pq_code_size,
                pq_code_size * sizeof(uint8_t));
    data_ptr += D;
    code_ptr += pq_code_size;
  }
};

void IVF_PQFastScan::compute_pq_LUT(float *query, float *LUT_float,
                                    uint8_t *LUT_uint8, float &min_val,
                                    float &max_val, float &scale,
                                    float &factor) {
  for (int sq = 0; sq < M; ++sq) {
    float *q_sub = query + sq * sub_dim;
    __m128 qv = _mm_loadu_ps(q_sub); // 4 floats

    for (int c = 0; c < centroid_per_sub; ++c) {
      float *cent = pq_cookbook + (sq * centroid_per_sub + c) * sub_dim;
      __m128 cv = _mm_loadu_ps(cent);
      __m128 diff = _mm_sub_ps(qv, cv);
      __m128 mul = _mm_mul_ps(diff, diff);
      float dist2 = hsum4(mul);
      LUT_float[sq * centroid_per_sub + c] = dist2;
    }
  }
  // 2. 找min/max
  int total = M * centroid_per_sub;
  min_val = std::numeric_limits<float>::infinity();
  max_val = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < total; ++i) {
    if (LUT_float[i] < min_val)
      min_val = LUT_float[i];
    if (LUT_float[i] > max_val)
      max_val = LUT_float[i];
  }
  scale = (max_val > min_val) ? (255.f / (max_val - min_val)) : 1.f; // 防止除0
  for (int i = 0; i < total; ++i) {
    float norm = (LUT_float[i] - min_val) * scale;
    if (norm < 0)
      norm = 0;
    if (norm > 255)
      norm = 255;
    LUT_uint8[i] = static_cast<uint8_t>(norm); // 四舍五入
  }
  factor = 1.0f / scale;
}

void IVF_PQFastScan::build_codebook_from_samples(
    std::vector<ProbeInfo> &probe_infos, uint32_t nprobe, uint32_t k,
    TopKBufferSoA &KNNs) {
  float sample_ratio = 0.01;
  uint32_t num_clusters_to_sample =
      std::max(1, static_cast<int>(nprobe * sample_ratio));
  uint32_t total_sample_size = 0;
  for (uint32_t i = 0; i < num_clusters_to_sample; ++i) {
    auto &info = probe_infos[i];
    uint32_t cluster_id = info.cluster_id;
    total_sample_size += len[cluster_id];
  }
  float PORTABLE_ALIGN32 sampled_dist[total_sample_size + 32];
  float global_min_dist = std::numeric_limits<float>::infinity();
  float global_max_dist = 0.0f;
  uint32_t offset = 0;
  for (int pb = 0; pb < num_clusters_to_sample; pb++) {
    ProbeInfo &info = probe_infos[pb];
    uint32_t c = info.cluster_id;
    sample_fast_scan(sampled_dist + offset, global_min_dist, global_max_dist,
                     info.LUT_uint8, packed_code + packed_start[c], info.factor,
                     info.min_val, len[c]);
    offset += int(len[c] / 32) * 32;
  }
  if (offset > k) {
    std::nth_element(sampled_dist, sampled_dist + k, sampled_dist + offset);
    global_max_dist = sampled_dist[k - 1];
    offset = k;
  }
  global_max_dist = global_max_dist + 1e-5;

  int32_t PSEUDO_BUCKETS = KNNs.get_logical_bucket_num();
  int32_t PHYSICLA_BUCKETS = KNNs.get_physical_bucket_num();
  float delta = (global_max_dist - global_min_dist) / PSEUDO_BUCKETS;
  float inv_delta = 1.0f / delta;
  uint32_t tmp_count[PSEUDO_BUCKETS] = {0};
  int32_t PORTABLE_ALIGN32 tmp_code[PHYSICLA_BUCKETS];

  const __m256 v_lower = _mm256_set1_ps(global_min_dist);
  const __m256 v_inv_delta = _mm256_set1_ps(inv_delta);
  const float *p = &sampled_dist[0];
  uint32_t vec32 = offset / 32;
  for (uint32_t b = 0; b < vec32; ++b) {
    for (int v = 0; v < 4; ++v) {
      __m256 dist = _mm256_loadu_ps(p + v * 8);
      __m256i uid = _mm256_cvttps_epi32(
          _mm256_mul_ps(_mm256_sub_ps(dist, v_lower), v_inv_delta));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp_code + v * 8), uid);
    }
    p += 32;

    for (int t = 0; t < 32; ++t) {
      ++tmp_count[std::min(tmp_code[t], PSEUDO_BUCKETS - 1)];
    }
  }

  uint32_t NUM_BUCKETS = KNNs.get_physical_bucket_num();
  uint32_t target = (offset + NUM_BUCKETS - 1) / NUM_BUCKETS;
  uint32_t cur_bucket = 0;
  uint32_t acc = 0;
  uint8_t *code_lut_ = KNNs.get_code_lut();
  for (uint32_t i = 0; i < PSEUDO_BUCKETS; ++i) {
    code_lut_[i] = static_cast<uint8_t>(cur_bucket);
    acc += tmp_count[i];
    if (acc >= target && cur_bucket + 1 < NUM_BUCKETS) {
      ++cur_bucket;
      acc = 0;
    }
  }
  KNNs.set_bounds(global_min_dist, global_max_dist, delta);
  acc = 0;
  cur_bucket = 0;
  int i = 0;
  while (acc < sample_ratio * k) {
    acc += tmp_count[i];
    if (acc >= sample_ratio * k) {
      break;
    }
    i++;
  }
  int physical_predict_bucket = code_lut_[i];

  while (code_lut_[i] == physical_predict_bucket) {
    i++;
  }
  int logical_predict_bucket = i;
  physical_predict_bucket = code_lut_[i];
  KNNs.set_predict_logical_bucket_id(logical_predict_bucket);
}

void IVF_PQFastScan::sample_fast_scan(float *sample_dist, float &samllest_dist,
                                      float &largest_dist, uint8_t *LUT_uint8,
                                      uint8_t *packed_code, float &factor,
                                      float &min_val, uint32_t len) {
  constexpr uint32_t SIZE = 32;
  uint32_t it = len / SIZE;
  float PORTABLE_ALIGN32 dist[SIZE];
  const __m256 v_factor = _mm256_set1_ps(factor);
  const __m256 v_add_bias = _mm256_set1_ps(min_val * M);
  __m256 v_min = _mm256_set1_ps(samllest_dist);
  __m256 v_max = _mm256_set1_ps(largest_dist);
  // uint32_t out_idx = 0;
  float *ptr_dist = sample_dist;
  while (it--) {
    uint16_t PORTABLE_ALIGN32 result[SIZE];
    accumulate((SIZE / 32), packed_code, LUT_uint8, result, pq_code_size);
    packed_code += SIZE * pq_code_size;

    for (uint32_t i = 0; i < SIZE; i += 8) {
      __m128i v_u16 =
          _mm_load_si128(reinterpret_cast<const __m128i *>(result + i));
      __m256i v_u32 = _mm256_cvtepu16_epi32(v_u16);
      __m256 v_f32 = _mm256_cvtepi32_ps(v_u32);
      __m256 v_dist = _mm256_fmadd_ps(v_f32, v_factor, v_add_bias);
      _mm256_storeu_ps(ptr_dist, v_dist);
      v_min = _mm256_min_ps(v_min, v_dist);
      v_max = _mm256_max_ps(v_max, v_dist);
      ptr_dist += 8;
    }
  }

  {
    // 先合并高低 128-bit
    __m128 lo_min = _mm256_castps256_ps128(v_min);
    __m128 hi_min = _mm256_extractf128_ps(v_min, 1);
    __m128 min128 = _mm_min_ps(lo_min, hi_min); // 4 lane

    __m128 lo_max = _mm256_castps256_ps128(v_max);
    __m128 hi_max = _mm256_extractf128_ps(v_max, 1);
    __m128 max128 = _mm_max_ps(lo_max, hi_max);

    // 再水平归约 4→1
    min128 = _mm_min_ps(min128, _mm_movehl_ps(min128, min128));
    min128 = _mm_min_ps(min128, _mm_shuffle_ps(min128, min128, 1));
    max128 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, 1));

    float local_min = _mm_cvtss_f32(min128);
    float local_max = _mm_cvtss_f32(max128);

    samllest_dist = std::min(samllest_dist, local_min);
    largest_dist = std::max(largest_dist, local_max);
  }
}

void IVF_PQFastScan::fast_scan(TopKBufferSoA &KNNs, uint8_t *LUT_uint8,
                               float &factor, float &min_val, uint32_t k,
                               uint8_t *packed_code, uint32_t len, float *query,
                               float *data, uint32_t start_id) {

  constexpr uint32_t SIZE = 32;
  uint32_t it = len / SIZE;
  uint32_t remain = len - it * SIZE;
  uint32_t nblk_remain = (remain + 31) / 32;
  uint32_t local_id = start_id;
  float lower = KNNs.get_lower();
  float inv_delta = 1.0 / KNNs.get_delta();
  int32_t lg_th_code = KNNs.get_logical_threshold_bucket_id();
  int lg_bucket_num = KNNs.get_logical_bucket_num();

  const __m256 v_sub_lower = _mm256_set1_ps(-lower);
  const __m256 v_inv_delta = _mm256_set1_ps(inv_delta);
  const __m256i v_th_epi = _mm256_set1_epi32(lg_th_code);
  const __m256 v_factor = _mm256_set1_ps(factor);
  const __m256 v_add_bias = _mm256_set1_ps(min_val * M);

  float PORTABLE_ALIGN32 dist[SIZE];
  int32_t PORTABLE_ALIGN32 code[SIZE];
  int32_t lg_predict_code = KNNs.get_predict_logical_bucket_id();
  while (it--) {
    uint16_t PORTABLE_ALIGN32 result[SIZE];
    accumulate((SIZE / 32), packed_code, LUT_uint8, result, pq_code_size);
    packed_code += SIZE * pq_code_size;
    float *ptr_dist = &dist[0];
    int32_t *ptr_code = &code[0];
    for (int i = 0; i < SIZE; i += 8) {
      __m128i v_u16 =
          _mm_load_si128(reinterpret_cast<const __m128i *>(result + i));
      __m256i v_u32 = _mm256_cvtepu16_epi32(v_u16);
      __m256 v_f32 = _mm256_cvtepi32_ps(v_u32);
      __m256 v_dist = _mm256_fmadd_ps(v_f32, v_factor, v_add_bias);
      v_dist = _mm256_add_ps(v_dist, v_sub_lower);
      __m256 v_code_f = _mm256_mul_ps(v_dist, v_inv_delta);
      __m256i v_code_i = _mm256_cvttps_epi32(v_code_f);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr_code), v_code_i);
      _mm256_storeu_ps(ptr_dist, v_dist);
      ptr_code += 8;
      ptr_dist += 8;
    }
    ptr_code = &code[0];
    ptr_dist = &dist[0];

    for (int i = 0; i < SIZE; i += 8) {

      __m256i v_code = _mm256_load_si256( // 一次读 32B 对齐；若未对齐换 loadu
          reinterpret_cast<const __m256i *>(ptr_code));
      __m256i v_gt = _mm256_cmpgt_epi32(v_th_epi, v_code); // v_th > code ?
      uint32_t mask_keep = _mm256_movemask_ps(_mm256_castsi256_ps(v_gt));
      while (mask_keep) {
        int lane = _tzcnt_u32(mask_keep);
        mask_keep &= mask_keep - 1;
        if (ptr_code[lane] < lg_predict_code) {
          float dist = compute_l2_distance(data + lane * D, query, D);
          rerank_count += 1;
          KNNs.push_exact(std::clamp(ptr_code[lane], 0, 255), dist,
                          local_id + lane);
        } else {
          KNNs.push(std::clamp(ptr_code[lane], 0, 255), ptr_dist[lane],
                    local_id + lane);
        }
      }

      ptr_dist += 8;
      ptr_code += 8;
      local_id += 8;
      data += 8 * D;
    }

    // for (int i = 0; i < SIZE; i++)
    // {
    //     if(code[i] < lg_th_code){
    //         if(code[i] < lg_predict_code){
    //             float tmp_dist = compute_l2_distance(data, query, D);
    //             KNNs.push(std::max(code[i], 0), tmp_dist, pack(local_id));
    //         }
    //         else{
    //             KNNs.push(std::max(code[i], 0), dist[i], local_id);
    //         }
    //     }
    //     local_id = local_id + 1;
    //     data += D;
    // }
  }

  {
    uint16_t PORTABLE_ALIGN32 result[SIZE];
    accumulate(nblk_remain, packed_code, LUT_uint8, result, pq_code_size);

    for (int i = 0; i < remain; i++) {
      float dist = result[i] * factor + min_val * M;
      int code = static_cast<int>((dist - lower) * inv_delta);
      if (code < lg_th_code) {
        if (code < lg_predict_code) {
          dist = compute_l2_distance(data, query, D);
          rerank_count += 1;
          KNNs.push_exact(std::clamp(code, 0, 255), dist, local_id);
        } else {
          KNNs.push(std::clamp(code, 0, 255), dist, local_id);
        }
      }
      local_id = local_id + 1;
      data += D;
    }
  }
}

void IVF_PQFastScan::rerank(
    float *query, float *data, uint32_t D, uint32_t num_cand, uint32_t k,
    TopKBufferSoA &KNNs,
    std::vector<std::pair<float, uint32_t>> &result_ids_dist) {
  uint32_t CL = (D * sizeof(float) + 63) / 64; // 每条向量多少行 64B
  unsigned accumulate_size = 0;
  for (uint32_t i = 0; i < KNNs.get_physical_threshold_bucket_id() - 1; ++i) {
    auto &bucket = KNNs.get_buffer()[i];
    int j = 0;
    int sz = bucket.size();
    accumulate_size += sz;
    const uint32_t *id_ptr = bucket.idx_data();
    const float *dist_ptr = bucket.val_data();
    const char *addr =
        reinterpret_cast<const char *>(data + 1ull * id_ptr[j] * D);
    memory::mem_prefetch_l1(addr, CL);
    while (j < sz) {
      if (j + 1 != sz) {
        addr = reinterpret_cast<const char *>(data + 1ull * id_ptr[j + 1] * D);
        memory::mem_prefetch_l1(addr, CL);
      }
      float dist = compute_l2_distance(query, data + 1ull * id_ptr[j] * D, D);
      result_ids_dist.emplace_back(dist, *(id + (id_ptr[j])));
      j++;
    }
    bucket = KNNs.get_exact_buffer()[i];
    j = 0;
    sz = bucket.size();
    id_ptr = bucket.idx_data();
    dist_ptr = bucket.val_data();
    while (j < sz) {
      result_ids_dist.emplace_back(dist_ptr[j], *(id + (id_ptr[j])));
      j++;
    }
  }

  auto &bucket = KNNs.get_buffer()[KNNs.get_physical_threshold_bucket_id() - 1];
  std::vector<std::pair<float, uint32_t>> tmp_result_ids_dist;
  int sz = bucket.size();
  tmp_result_ids_dist.reserve(sz);
  const uint32_t *id_ptr = bucket.idx_data();
  const float *dist_ptr = bucket.val_data();
  // 使用 nth_element 找到前 remaining_size 个最小距离的索引
  if (num_cand < sz + result_ids_dist.size()) {
    // 先做 [0..n) 的索引数组
    unsigned remaining_size = num_cand - result_ids_dist.size();

    std::vector<unsigned> idx(sz);
    std::iota(idx.begin(), idx.end(), 0);
    auto less_by_key = [&](unsigned a, unsigned b) {
      return dist_ptr[a] < dist_ptr[b];
    };
    std::nth_element(idx.begin(), idx.begin() + remaining_size, idx.end(),
                     less_by_key);
    const char *addr =
        reinterpret_cast<const char *>(data + 1ull * id_ptr[idx[0]] * D);
    memory::mem_prefetch_l1(addr, CL);
    for (unsigned i = 0; i < remaining_size; ++i) {
      if (i + 1 < remaining_size) {
        addr = reinterpret_cast<const char *>(data +
                                              1ull * id_ptr[idx[i + 1]] * D);
        memory::mem_prefetch_l1(addr, CL);
      }
      float dist =
          compute_l2_distance(query, data + 1ull * id_ptr[idx[i]] * D, D);
      result_ids_dist.emplace_back(dist, *(id + id_ptr[idx[i]]));
    }
  } else {
    const char *addr =
        reinterpret_cast<const char *>(data + 1ull * id_ptr[0] * D);
    memory::mem_prefetch_l1(addr, CL);
    for (unsigned i = 0; i < sz; ++i) {
      if (i + 1 < sz) {
        addr = reinterpret_cast<const char *>(data + 1ull * id_ptr[i + 1] * D);
        memory::mem_prefetch_l1(addr, CL);
      }
      float dist = compute_l2_distance(query, data + 1ull * id_ptr[i] * D, D);
      result_ids_dist.emplace_back(dist, *(id + id_ptr[i]));
    }
  }

  bucket = KNNs.get_exact_buffer()[KNNs.get_physical_threshold_bucket_id() - 1];
  id_ptr = bucket.idx_data();
  dist_ptr = bucket.val_data();
  sz = bucket.size();
  for (int i = 0; i < sz; i++) {
    result_ids_dist.emplace_back(dist_ptr[i], *(id + (id_ptr[i])));
  }

  if (result_ids_dist.size() >= num_cand) {
    std::nth_element(result_ids_dist.begin(), result_ids_dist.begin() + k,
                     result_ids_dist.end(),
                     [](const std::pair<float, uint32_t> &a,
                        const std::pair<float, uint32_t> &b) {
                       return a.first < b.first; // 从小到大
                     });
    result_ids_dist.resize(k);
  }
}

std::vector<std::pair<float, uint32_t>>
IVF_PQFastScan::improved_search(float *query, float *rd_query,
                                uint32_t num_cand, uint32_t k, uint32_t nprobe,
                                TopKBufferSoA &KNNs) {
  Result centroid_dist[C];
  float *ptr_c = centroid;
  float *ptr_residual = residual;
  for (int i = 0; i < C; i++) {
    centroid_dist[i].first = compute_sub(rd_query, ptr_c, ptr_residual, D);
    centroid_dist[i].second = i;
    ptr_c += D;
    ptr_residual += D;
  }

  std::partial_sort(centroid_dist, centroid_dist + nprobe, centroid_dist + C);

  // ===========================================================================================================
  // Scan the first nprobe clusters.
  Result *ptr_centroid_dist = (&centroid_dist[0]);
  for (int pb = 0; pb < nprobe; pb++) {
    uint32_t c = ptr_centroid_dist->second;
    ptr_residual = residual + 1ull * c * D;
    ProbeInfo &probe_info = probe_infos[pb];
    compute_pq_LUT(ptr_residual, probe_info.LUT_float, probe_info.LUT_uint8,
                   probe_info.min_val, probe_info.max_val, probe_info.scale,
                   probe_info.factor);
    probe_info.cluster_id = c;
    ptr_centroid_dist++;
  }

  build_codebook_from_samples(probe_infos, nprobe, num_cand, KNNs);
  float distK = std::numeric_limits<float>::max();
  for (int pb = 0; pb < nprobe; pb++) {
    ProbeInfo &info = probe_infos[pb];
    uint32_t c = info.cluster_id;
    float *ptr_data = data + 1ull * start[c] * D;
    uint8_t *ptr_packed_code = packed_code + packed_start[c];
    fast_scan(KNNs, info.LUT_uint8, info.factor, info.min_val, num_cand,
              ptr_packed_code, len[c], query, ptr_data, start[c]);
    KNNs.update_th_code(pb, nprobe);
  }

  std::vector<std::pair<float, uint32_t>> result_ids_dist;
  result_ids_dist.reserve(num_cand);

  rerank(query, data, D, num_cand, k, KNNs, result_ids_dist);

  return result_ids_dist;
}

void IVF_PQFastScan::load(char *filename) {
  std::ifstream input(filename, std::ios::binary);
  if (!input.is_open()) {
    throw std::runtime_error("Cannot open file");
  }
  input.read((char *)&N, sizeof(uint32_t));
  input.read((char *)&D, sizeof(uint32_t));
  input.read((char *)&C, sizeof(uint32_t));
  input.read((char *)&M, sizeof(uint32_t));
  input.read((char *)&centroid_per_sub, sizeof(uint32_t));
  input.read((char *)&pq_code_size, sizeof(uint32_t));
  sub_dim = D / M;

  std::cerr << "M: " << M << " ksub: " << centroid_per_sub
            << " dsub: " << sub_dim << std::endl;

  start = new uint32_t[C];
  len = new uint32_t[C];
  id = new uint32_t[N];

  input.read((char *)start, C * sizeof(uint32_t));
  input.read((char *)len, C * sizeof(uint32_t));
  input.read((char *)id, N * sizeof(uint32_t));

  centroid = new float[C * D];
  residual = new float[C * D];
  pq_cookbook = new float[M * centroid_per_sub * sub_dim];
  // data = new float[1ull * N * D];
  data = static_cast<float *>(aligned_alloc(64, 1ull * N * D * sizeof(float)));

  input.read((char *)centroid, C * D * sizeof(float));
  input.read((char *)pq_cookbook,
             M * centroid_per_sub * sub_dim * sizeof(float));
  input.read((char *)data, 1ull * N * D * sizeof(float));

  pq_codes = static_cast<uint8_t *>(
      aligned_alloc(64, 1ull * N * pq_code_size * sizeof(uint8_t)));

  input.read((char *)pq_codes, 1ull * N * pq_code_size * sizeof(uint8_t));
  input.close();

  probe_infos.resize(C);
  for (int i = 0; i < C; i++) {
    probe_infos[i].LUT_float =
        (float *)aligned_alloc(32, sizeof(float) * M * centroid_per_sub);
    probe_infos[i].LUT_uint8 =
        (uint8_t *)aligned_alloc(32, sizeof(uint8_t) * M * centroid_per_sub);
  }

  packed_start = new uint32_t[C];
  uint32_t cur = 0;
  for (int i = 0; i < C; i++) {
    packed_start[i] = cur;
    cur += (len[i] + 31) / 32 * 32 * pq_code_size;
  }
  packed_code =
      static_cast<uint8_t *>(aligned_alloc(32, cur * sizeof(uint8_t)));
  for (int i = 0; i < C; i++) {
    pack_codes(pq_codes + 1ull * start[i] * pq_code_size, len[i],
               packed_code + packed_start[i], pq_code_size * 8);
  }
  std::cerr << "loaded!" << std::endl;
}

void IVF_PQFastScan::save(char *filename) {
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
  output.write((char *)pq_cookbook,
               M * centroid_per_sub * sub_dim * sizeof(float));

  output.write((char *)data, 1ull * N * D * sizeof(float));
  output.write((char *)pq_codes, 1ull * N * pq_code_size * sizeof(uint8_t));

  output.close();
  std::cerr << "Saved!" << std::endl;
}

IVF_PQFastScan::~IVF_PQFastScan() {
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
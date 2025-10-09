// ==================================================================
// IVFRN() involves pre-processing steps (e.g., packing the
// quantization codes into a batch) in the index phase.

// search() is the main function of the query phase.
// ==================================================================
#pragma once
#define RANDOM_QUERY_QUANTIZATION
#include "fast_scan.h"
#include "matrix.h"
#include "memory.h"
#include "space.h"
#include "utils.h"
#include <algorithm>
#include <map>
#include <queue>
#include <vector>

uint32_t rerank_count;
template <uint32_t D, uint32_t B> class IVFRN {
private:
public:
  struct Factor {
    float sqr_x;
    float error;
    float factor_ppc;
    float factor_ip;
  };

  static constexpr float fac_norm = const_sqrt(1.0 * B);
  static constexpr float max_x1 = 1.9 / const_sqrt(1.0 * B - 1.0);
  static Space<D, B> space;

  uint32_t N; // the number of data vectors
  uint32_t C; // the number of clusters

  uint32_t *start; // the start point of a cluster
  uint32_t
      *packed_start; // the start point of a cluster (packed with batch of 32)
  uint32_t *len;     // the length of a cluster
  uint32_t *id;      // N of size_t the ids of the objects in a cluster

  float *data;
  float *dist_to_c;
  float *u;
  float *sqr_x;
  float *factor_ip;
  float *factor_ppc;
  float *error;
  uint64_t *binary_code; // (B / 64) * N of 64-bit uint64_t
  uint8_t *packed_code;  // packed code with the batch size of 32 vectors

  float *x0;       // N of floats in the Random Net algorithm
  float *centroid; // N * B floats (not N * D), note that the centroids should
                   // be randomized

  IVFRN();
  IVFRN(const Matrix<float> &X, const Matrix<float> &_centroids,
        const Matrix<float> &dist_to_centroid, const Matrix<float> &_x0,
        const Matrix<uint32_t> &cluster_id, const Matrix<uint64_t> &binary);
  ~IVFRN();

  std::vector<std::pair<float, uint32_t>>
  improved_search(float *query, float *rd_query, uint32_t k, uint32_t nprobe,
                  TopKBufferSoA &upper_KNNs) const;

  inline void rerank(float *query, float *data, uint32_t k,
                     TopKBufferSoA &upper_KNNs,
                     std::vector<std::pair<float, uint32_t>> &KNNs) const;

  inline void
  build_codebook_from_samples(std::vector<ProbeInfo<B>> &probe_infos,
                              uint32_t nprobe, uint32_t k,
                              TopKBufferSoA &upper_KNNs) const;

  static void bound_fast_scan(TopKBufferSoA &upper_KNNs, float *query,
                              float *data, const uint32_t k, const uint8_t *LUT,
                              const uint8_t *packed_code, uint32_t len,
                              // Factor *ptr_fac,
                              const float *sqr_x, const float *factor_ip,
                              const float *factor_ppc, const float *error,
                              const float sqr_y, const float vl,
                              const float width, const float sumq,
                              const uint32_t start_idx);

  static void sample_bound_fast_scan(float *sample_dist, float &samllest_dist,
                                     float &largest_dist, const uint8_t *LUT,
                                     const uint8_t *packed_code, uint32_t len,
                                     // Factor *ptr_fac,
                                     const float *sqr_x, const float *factor_ip,
                                     const float *factor_ppc,
                                     const float *error, const float sqr_y,
                                     const float vl, const float width,
                                     const float sumq);

  void save(char *filename);
  void load(char *filename);
};

template <uint32_t D, uint32_t B>
void IVFRN<D, B>::build_codebook_from_samples(
    std::vector<ProbeInfo<B>> &probe_infos, uint32_t nprobe, uint32_t k,
    TopKBufferSoA &upper_KNNs) const {
  float sample_ratio = 0.01f;
  uint32_t num_clusters_to_sample =
      std::max(1, static_cast<int>(nprobe * sample_ratio));
  uint32_t total_sample_size = 0;
  for (uint32_t i = 0; i < num_clusters_to_sample; ++i) {
    const auto &info = probe_infos[i];
    uint32_t cluster_id = info.cluster_id;
    total_sample_size += len[cluster_id];
  }
  float PORTABLE_ALIGN32 sampled_dist[total_sample_size + 32];
  float global_min_dist = std::numeric_limits<float>::infinity();
  float global_max_dist = 0.0f;
  uint32_t offset = 0;
  for (int pb = 0; pb < num_clusters_to_sample; pb++) {
    ProbeInfo<B> &info = probe_infos[pb];
    uint32_t c = info.cluster_id;
#if defined(FAST_SCAN)
    sample_bound_fast_scan(
        sampled_dist + offset, global_min_dist, global_max_dist, info.LUT,
        packed_code + packed_start[c], len[c], sqr_x + start[c],
        factor_ip + start[c], factor_ppc + start[c], error + start[c],
        info.sqr_y, info.vl, info.width, info.sum_q);
#endif
    offset += int(len[c] / 32) * 32;
  }

  if (offset > k) {
    std::nth_element(sampled_dist, sampled_dist + k, sampled_dist + offset);
    global_max_dist = sampled_dist[k - 1];
    offset = k;
  }
  global_max_dist = global_max_dist + 1e-5;

  int32_t PSEUDO_BUCKETS = upper_KNNs.get_logical_bucket_num();
  int32_t PHYSICLA_BUCKETS = upper_KNNs.get_physical_bucket_num();
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

    for (int k = 0; k < 32; ++k) {
      ++tmp_count[std::min(tmp_code[k], PSEUDO_BUCKETS - 1)];
    }
  }

  uint32_t NUM_BUCKETS = upper_KNNs.get_physical_bucket_num();
  uint32_t target = (offset + NUM_BUCKETS - 1) / NUM_BUCKETS;
  uint32_t cur_bucket = 0;
  uint32_t acc = 0;

  uint8_t *code_lut_ = upper_KNNs.get_code_lut();
  for (uint32_t i = 0; i < PSEUDO_BUCKETS; ++i) {
    code_lut_[i] = static_cast<uint8_t>(cur_bucket);
    acc += tmp_count[i];

    if (acc >= target && cur_bucket + 1 < NUM_BUCKETS) {
      ++cur_bucket;
      acc = 0;
    }
  }
  upper_KNNs.set_bounds(global_min_dist, global_max_dist, delta);
  // // cur_bucket = 0;
  // uint32_t sample_target_k = int(sample_ratio * k);

  // acc = 0;
  // for (uint32_t i = 0; i < PSEUDO_BUCKETS; ++i) {
  //   acc += tmp_count[i];

  //   if (acc >= sample_target_k) {
  //     cur_bucket = code_lut_[i];
  //     while (code_lut_[i] == cur_bucket) {
  //       i++;
  //     }
  //     upper_KNNs.set_top_logical_bucket_id(i);
  //     break;
  //   }
  // }

  // upper_KNNs.set_marginal_logical_bucket_id(1e7);
  // std::cerr << " i: " << i << std::endl;
}

template <uint32_t D, uint32_t B>
void IVFRN<D, B>::rerank(float *query, float *data, uint32_t k,
                         TopKBufferSoA &upper_KNNs,
                         std::vector<std::pair<float, uint32_t>> &KNNs) const {
  const int num_buckets = upper_KNNs.get_physical_bucket_num();
  int th_code = upper_KNNs.get_physical_threshold_bucket_id();

  const float delta = upper_KNNs.get_delta();
  const float lowest = upper_KNNs.get_lower();
  const float inv_delta = 1.0f / delta;
  constexpr size_t CL = (D * sizeof(float) + 63) / 64;

  // Move entries from upper buckets to lower
  for (int i = th_code; i < num_buckets; ++i) {
    auto &bucket = upper_KNNs.get_buffer()[i];
    const float *lb_ptr = bucket.val_data();
    const uint32_t *id_ptr = bucket.idx_data();
    const uint32_t sz = bucket.size();

    for (uint32_t j = 0; j < sz; ++j) {
      int lb_code =
          std::clamp(static_cast<int>((*lb_ptr - lowest) * inv_delta), 0, 255);
      upper_KNNs.push_lower(lb_code, *lb_ptr, *id_ptr);
      ++lb_ptr;
      ++id_ptr;
    }
    bucket.clear();
  }
  //   upper_KNNs.flush_all_upper();

  int lower_idx = 0;
  int upper_idx = upper_KNNs.get_physical_threshold_bucket_id() - 1;

  while (upper_idx > lower_idx) {
    // Process upper buckets
    auto process_bucket = [&](auto &bucket) {
      const uint32_t sz = bucket.size();
      const uint32_t *id_ptr = bucket.idx_data();
      memory::mem_prefetch_l1(
          reinterpret_cast<const char *>(data + 1ull * id_ptr[0] * D), CL);
      for (uint32_t t = 0; t < sz; ++t) {
        if (t + 1 < sz) {
          memory::mem_prefetch_l1(
              reinterpret_cast<const char *>(data + 1ull * id_ptr[t + 1] * D),
              CL);
        }
        float dist = sqr_dist<D>(query, data + 1ull * id_ptr[t] * D);
        rerank_count++;
        int code =
            std::clamp(static_cast<int>((dist - lowest) * inv_delta), 0, 255);
        upper_KNNs.push_exact(code, dist, id_ptr[t]);
      }
      bucket.clear();
    };
    process_bucket(upper_KNNs.get_buffer()[upper_idx]);
    process_bucket(upper_KNNs.get_lower_buffer()[lower_idx]);

    upper_KNNs.update_th_code();
    int new_th_code = upper_KNNs.get_physical_threshold_bucket_id();
    int new_lg_th_code = upper_KNNs.get_logical_threshold_bucket_id();
    // Move intermediate buckets into lower buffer
    for (int t = new_th_code; t < th_code; ++t) {
      auto &bucket = upper_KNNs.get_buffer()[t];
      const uint32_t sz = bucket.size();
      const uint32_t *id_ptr = bucket.idx_data();
      const float *data_ptr = bucket.val_data();

      for (uint32_t ti = 0; ti < sz; ++ti) {
        int code = std::clamp(
            static_cast<int>((data_ptr[ti] - lowest) * inv_delta), 0, 255);
        if (code < new_lg_th_code)
          upper_KNNs.push_lower(code, data_ptr[ti], id_ptr[ti]);
      }
      bucket.clear();
    }

    th_code = new_th_code;
    upper_idx = th_code - 1;

    lower_idx = 0;
    while (lower_idx < upper_idx &&
           upper_KNNs.get_lower_buffer()[lower_idx].size() == 0) {
      ++lower_idx;
    }
  }

  auto collect_bucket = [&](auto &bucket) {
    const uint32_t sz = bucket.size();
    const uint32_t *id_ptr = bucket.idx_data();
    const float *val_ptr = bucket.val_data();
    for (uint32_t j = 0; j < sz; ++j) {
      KNNs.emplace_back(val_ptr[j], *(id + id_ptr[j]));
    }
  };

  // 3. Collect final results
  for (int idx = 0; idx < upper_idx; ++idx) {
    collect_bucket(upper_KNNs.get_buffer()[idx]);
    collect_bucket(upper_KNNs.get_exact_buffer()[idx]);
  }

  // 4. Special case for last bucket
  // collect_bucket(upper_KNNs.get_buffer()[upper_idx]);

  int remaining_size = k - static_cast<int>(KNNs.size());

  if (remaining_size > 0) {
    MaxHeap<float, unsigned> tempres;
    float distK = std::numeric_limits<float>::max();
    auto &ex_bucket = upper_KNNs.get_exact_buffer()[upper_idx];
    const float *data_ptr = ex_bucket.val_data();
    const uint32_t *id_ptr = ex_bucket.idx_data();
    for (uint32_t j = 0; j < ex_bucket.size(); ++j) {
      if (data_ptr[j] < distK) {
        tempres.push(data_ptr[j], id_ptr[j]);
        if (tempres.size() > remaining_size)
          tempres.pop();
        if (tempres.size() == remaining_size)
          distK = tempres.top_key();
      }
    }

    // MaxHeap<float, unsigned> candidate;
    // auto &upper_bucket = upper_KNNs.get_buffer()[upper_idx];
    // data_ptr = upper_bucket.val_data();
    // id_ptr = upper_bucket.idx_data();
    // if (upper_bucket.size() > 0) {
    //   for (uint32_t j = 0; j < upper_bucket.size(); ++j) {
    //     if (data_ptr[j] < distK)
    //       candidate.push_unsorted(-data_ptr[j], id_ptr[j]);
    //   }
    // }
    // auto &low_bucket = upper_KNNs.get_lower_buffer()[lower_idx];
    // data_ptr = low_bucket.val_data();
    // id_ptr = low_bucket.idx_data();
    // if (low_bucket.size() > 0) {
    //   for (uint32_t j = 0; j < low_bucket.size(); ++j) {
    //     if (data_ptr[j] < distK)
    //       candidate.push_unsorted(-data_ptr[j], id_ptr[j]);
    //   }
    // }

    // memory::mem_prefetch_l1(
    //     reinterpret_cast<const char *>(data + 1ull * candidate.top_data() *
    //     D), CL);

    // while ((!candidate.empty()) && (candidate.top_key() + distK > 0)) {
    //   unsigned id = candidate.top_data();
    //   candidate.pop();
    //   if (!candidate.empty()) {
    //     unsigned next_id = candidate.top_data();
    //     memory::mem_prefetch_l1(
    //         reinterpret_cast<const char *>(data + 1ull * next_id * D), CL);
    //   }
    //   float dist = sqr_dist<D>(query, data + 1ull * id * D);
    //   rerank_count++;
    //   if (dist < distK) {
    //     tempres.push(dist, id);
    //     if (tempres.size() > remaining_size)
    //       tempres.pop();
    //     if (tempres.size() == remaining_size)
    //       distK = tempres.top_key();
    //   }
    // }

    auto &upper_bucket = upper_KNNs.get_buffer()[upper_idx];
    data_ptr = upper_bucket.val_data();
    id_ptr = upper_bucket.idx_data();
    if (upper_bucket.size() > 0) {
      memory::mem_prefetch_l1(
          reinterpret_cast<const char *>(data + 1ull * id_ptr[0] * D), CL);
      for (uint32_t j = 0; j < upper_bucket.size(); ++j) {
        if (j + 1 < upper_bucket.size()) {
          memory::mem_prefetch_l1(
              reinterpret_cast<const char *>(data + 1ull * id_ptr[j + 1] * D),
              CL);
        }
        if (data_ptr[j] < distK) {
          float dist = sqr_dist<D>(query, data + 1ull * id_ptr[j] * D);
          rerank_count++;

          if (dist < distK) {
            tempres.push(dist, id_ptr[j]);
            if (tempres.size() > remaining_size) {
              tempres.pop();
            }
            if (tempres.size() == remaining_size) {
              distK = tempres.top_key();
            }
          }
        }
      }
    }

    auto &low_bucket = upper_KNNs.get_lower_buffer()[lower_idx];
    data_ptr = low_bucket.val_data();
    id_ptr = low_bucket.idx_data();
    if (low_bucket.size() > 0) {
      memory::mem_prefetch_l1(
          reinterpret_cast<const char *>(data + 1ull * id_ptr[0] * D), CL);
      for (uint32_t j = 0; j < low_bucket.size(); ++j) {
        if (j + 1 < low_bucket.size()) {
          memory::mem_prefetch_l1(
              reinterpret_cast<const char *>(data + 1ull * id_ptr[j + 1] * D),
              CL);
        }
        if (data_ptr[j] < distK) {
          float dist = sqr_dist<D>(query, data + 1ull * id_ptr[j] * D);
          rerank_count++;
          if (dist < distK) {
            tempres.push(dist, id_ptr[j]);
            if (tempres.size() > remaining_size) {
              tempres.pop();
            }
            if (tempres.size() == remaining_size) {
              distK = tempres.top_key();
            }
          }
        }
      }
    }

    auto &data = tempres.get_data();
    for (uint32_t j = 0; j < data.size(); j++) {
      KNNs.emplace_back(data[j].key, *(id + data[j].data));
    }
  }
}

template <uint32_t D, uint32_t B>
void IVFRN<D, B>::sample_bound_fast_scan(
    float *sample_dist, float &samllest_dist, float &largest_dist,
    const uint8_t *LUT, const uint8_t *packed_code, uint32_t len,
    const float *sqr_x, const float *factor_ip, const float *factor_ppc,
    const float *error, const float sqr_y, const float vl, const float width,
    const float sumq) {
  float y = std::sqrt(sqr_y);

  constexpr uint32_t SIZE = 32;
  uint32_t it = len / SIZE;

  float PORTABLE_ALIGN32 result_float[SIZE];

  const float *ptr_sqr_x = &sqr_x[0];
  const float *ptr_factor_ip = &factor_ip[0];
  const float *ptr_factor_ppc = &factor_ppc[0];
  const float *ptr_error = &error[0];

  const __m256 sqr_y_simd = _mm256_set1_ps(sqr_y);
  const __m256 vl_simd = _mm256_set1_ps(vl);
  const __m256 width_simd = _mm256_set1_ps(width);
  const __m256 y_simd = _mm256_set1_ps(y);

  float *ptr_dist = sample_dist;
  __m256 v_min = _mm256_set1_ps(samllest_dist);
  __m256 v_max = _mm256_set1_ps(largest_dist);
  while (it--) {
    uint16_t PORTABLE_ALIGN32 result[SIZE]; // 存储每个批次的计算结果
    accumulate<B>((SIZE / 32), packed_code, LUT, result);
    packed_code += SIZE * B / 8;

    for (uint32_t i = 0; i < SIZE; ++i) {
      result_float[i] = static_cast<float>(static_cast<int>(result[i]) - sumq);
    }
    float *ptr_result_float = &result_float[0];

    for (int i = 0; i < SIZE; i += 8) {
      __m256 result_simd = _mm256_loadu_ps(ptr_result_float);
      __m256 sqr_x_simd = _mm256_loadu_ps(ptr_sqr_x);
      __m256 factor_ip_simd = _mm256_loadu_ps(ptr_factor_ip);
      __m256 factor_ppc_simd = _mm256_loadu_ps(ptr_factor_ppc);
      __m256 error_simd = _mm256_loadu_ps(ptr_error);
      // result *= factor_ip → result = result * factor_ip
      result_simd = _mm256_mul_ps(result_simd, factor_ip_simd);

      // result = result * width + sqr_x
      result_simd = _mm256_fmadd_ps(result_simd, width_simd, sqr_x_simd);

      // result += sqr_y
      result_simd = _mm256_add_ps(result_simd, sqr_y_simd);

      // result += factor_ppc * vl
      result_simd = _mm256_fmadd_ps(factor_ppc_simd, vl_simd, result_simd);
      v_min = _mm256_min_ps(v_min, result_simd);
      v_max = _mm256_max_ps(v_max, result_simd);

      _mm256_storeu_ps(ptr_dist, result_simd);

      ptr_result_float += 8;
      ptr_sqr_x += 8;
      ptr_factor_ip += 8;
      ptr_factor_ppc += 8;
      ptr_error += 8;
      ptr_dist += 8;
    }
  }
  /* -------- 4. 把 8 lane 的 min / max 归并成标量 -------- */
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

    float min_val = _mm_cvtss_f32(min128);
    float max_val = _mm_cvtss_f32(max128);

    samllest_dist = std::min(samllest_dist, min_val);
    largest_dist = std::max(largest_dist, max_val);
  }
}

// search impl
template <uint32_t D, uint32_t B>
std::vector<std::pair<float, uint32_t>>
IVFRN<D, B>::improved_search(float *query, float *rd_query, uint32_t k,
                             uint32_t nprobe, TopKBufferSoA &upper_KNNs) const {
  // The default value of distK is +inf
  std::vector<std::pair<float, uint32_t>> KNNs;
  KNNs.reserve(k);

  Result centroid_dist[numC];
  float *ptr_c = centroid;
  for (int i = 0; i < C; i++) {
    centroid_dist[i].first = sqr_dist<B>(rd_query, ptr_c);
    centroid_dist[i].second = i;
    ptr_c += B;
    rerank_count += 1;
  }
  std::partial_sort(centroid_dist, centroid_dist + nprobe,
                    centroid_dist + numC);
  uint8_t PORTABLE_ALIGN64 byte_query[B];

  std::vector<ProbeInfo<B>> probe_infos(nprobe); // 存储每个 pb 的信息
  Result *ptr_centroid_dist = (&centroid_dist[0]);
  // ===========================================================================================================
  // Scan the first nprobe clusters.
  for (int pb = 0; pb < nprobe; pb++) {
    uint32_t c = ptr_centroid_dist->second;
    float sqr_y = ptr_centroid_dist->first;
    ptr_centroid_dist++;
    float vl, vr;
    space.range(rd_query, centroid + c * B, vl, vr);
    float width = (vr - vl) / ((1 << B_QUERY) - 1);
    uint32_t sum_q = 0;
    space.quantize(byte_query, rd_query, centroid + c * B, u, vl, width, sum_q);
    ProbeInfo<B> &info = probe_infos[pb];
    info.sqr_y = sqr_y;
    info.vl = vl;
    info.width = width;
    info.sum_q = sum_q;
    info.cluster_id = c;
#if defined(FAST_SCAN) // Look-Up-Table Representation
    pack_LUT<B>(byte_query, info.LUT);
    for (int i = 0; i < B / 4 * 16; i++)
      info.LUT[i] <<= 1;
#endif
  }

  build_codebook_from_samples(probe_infos, nprobe, k, upper_KNNs);

  for (int pb = 0; pb < nprobe; pb++) {
    ProbeInfo<B> &info = probe_infos[pb];
    uint32_t c = info.cluster_id;
    float *ptr_data = data + 1ull * D * start[c];
#if defined(FAST_SCAN)
    bound_fast_scan(upper_KNNs, query, ptr_data, k, info.LUT,
                    packed_code + packed_start[c], len[c], sqr_x + start[c],
                    factor_ip + start[c], factor_ppc + start[c],
                    error + start[c], info.sqr_y, info.vl, info.width,
                    info.sum_q, start[c]);
#endif
    upper_KNNs.update_th_code();
  }
  rerank(query, data, k, upper_KNNs, KNNs);
  return KNNs;
}

template <uint32_t D, uint32_t B>
void IVFRN<D, B>::bound_fast_scan(
    TopKBufferSoA &upper_KNNs, float *query, float *data, const uint32_t k,
    const uint8_t *LUT, const uint8_t *packed_code, uint32_t len,
    const float *sqr_x, const float *factor_ip, const float *factor_ppc,
    const float *error, const float sqr_y, const float vl, const float width,
    const float sumq, const uint32_t start_idx) {
  float y = std::sqrt(sqr_y);
  constexpr uint32_t SIZE = 32;
  uint32_t it = len / SIZE;
  uint32_t remain = len - it * SIZE;
  uint32_t nblk_remain = (remain + 31) / 32;
  uint32_t idx = start_idx;
  float PORTABLE_ALIGN32 result_float[SIZE];
  float PORTABLE_ALIGN32 lower_dist[SIZE];   // 用于存储每个样本的最近距离
  float PORTABLE_ALIGN32 upper_dist[SIZE];   // 用于存储每个样本的最远距离
  int32_t PORTABLE_ALIGN32 lower_code[SIZE]; // 用于存储每个样本的最近距离
  int32_t PORTABLE_ALIGN32 upper_code[SIZE]; // 用于存储每个样本的最近距离

  int32_t lg_th_code = upper_KNNs.get_logical_threshold_bucket_id();
  int32_t lg_bucket_num = upper_KNNs.get_logical_bucket_num();
  float delta = upper_KNNs.get_delta();
  float inv_delta = 1.0 / delta;

  float lower = upper_KNNs.get_lower();
  const float *ptr_sqr_x = &sqr_x[0];
  const float *ptr_factor_ip = &factor_ip[0];
  const float *ptr_factor_ppc = &factor_ppc[0];
  const float *ptr_error = &error[0];

  const __m256 v_inv_delta = _mm256_set1_ps(inv_delta);
  const __m256 sqr_y_simd = _mm256_set1_ps(sqr_y);
  const __m256 vl_simd = _mm256_set1_ps(vl);
  const __m256 width_simd = _mm256_set1_ps(width);
  const __m256 y_simd = _mm256_set1_ps(y);
  const __m256 v_lower = _mm256_set1_ps(lower);
  const __m256i v_th_epi = _mm256_set1_epi32(lg_th_code);
  const uint8_t *bucket_lookup = upper_KNNs.get_code_lut();
  //   int top_logical_bucket_id = upper_KNNs.get_top_logical_bucket_id();

  float *ptr_data = data;
  while (it--) {
    float *ptr_lower_dist = &lower_dist[0];
    float *ptr_upper_dist = &upper_dist[0];
    int32_t *ptr_upper_code = &upper_code[0];
    int32_t *ptr_lower_code = &lower_code[0];

    uint16_t PORTABLE_ALIGN32 result[SIZE]; // 存储每个批次的计算结果
    accumulate<B>((SIZE / 32), packed_code, LUT, result);
    packed_code += SIZE * B / 8;

    for (uint32_t i = 0; i < SIZE; ++i) {
      result_float[i] = static_cast<float>(static_cast<int>(result[i]) - sumq);
    }
    float *ptr_result_float = &result_float[0];

    for (int i = 0; i < SIZE; i += 8) {
      // ------- 1. 距离计算，与之前相同 -------
      __m256 result_simd = _mm256_loadu_ps(ptr_result_float);
      __m256 sqr_x_simd = _mm256_loadu_ps(ptr_sqr_x);
      __m256 factor_ip_simd = _mm256_loadu_ps(ptr_factor_ip);
      __m256 factor_ppc_simd = _mm256_loadu_ps(ptr_factor_ppc);
      __m256 error_simd = _mm256_loadu_ps(ptr_error);

      result_simd = _mm256_mul_ps(result_simd, factor_ip_simd);
      result_simd = _mm256_fmadd_ps(result_simd, width_simd, sqr_x_simd);
      result_simd = _mm256_add_ps(result_simd, sqr_y_simd);
      result_simd = _mm256_fmadd_ps(factor_ppc_simd, vl_simd, result_simd);

      // ---------------- upper ----------------
      __m256 upper_dist = _mm256_fmadd_ps(error_simd, y_simd, result_simd);
      _mm256_storeu_ps(ptr_upper_dist, upper_dist);

      // uid = floor((upper_dist - lower) * inv_deltaU)
      __m256i uid = _mm256_cvttps_epi32(
          _mm256_mul_ps(_mm256_sub_ps(upper_dist, v_lower), v_inv_delta));

      _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr_upper_code), uid);

      // ---------------- lower ----------------
      __m256 lower_dist = _mm256_fnmadd_ps(error_simd, y_simd, result_simd);
      _mm256_storeu_ps(ptr_lower_dist, lower_dist);

      uid = _mm256_cvttps_epi32(
          _mm256_mul_ps(_mm256_sub_ps(lower_dist, v_lower), v_inv_delta));

      _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr_lower_code), uid);

      // ------- pointer bump --------
      ptr_result_float += 8;
      ptr_sqr_x += 8;
      ptr_factor_ip += 8;
      ptr_factor_ppc += 8;
      ptr_error += 8;
      ptr_upper_dist += 8;
      ptr_upper_code += 8;
      ptr_lower_dist += 8;
      ptr_lower_code += 8;
    }

    ptr_lower_dist = &lower_dist[0];
    ptr_upper_dist = &upper_dist[0];
    ptr_upper_code = &upper_code[0];
    ptr_lower_code = &lower_code[0];

    for (int i = 0; i < SIZE; i += 8) {
      __m256i v_code = _mm256_loadu_si256( // 一次读 32B 对齐；若未对齐换 loadu
          reinterpret_cast<const __m256i *>(ptr_lower_code));
      __m256i v_gt = _mm256_cmpgt_epi32(v_th_epi, v_code); // v_th > code ?
      uint32_t mask_keep = _mm256_movemask_ps(_mm256_castsi256_ps(v_gt));
      while (mask_keep) {
        int lane = _tzcnt_u32(mask_keep);
        mask_keep &= mask_keep - 1;
        if (ptr_upper_code[lane] < lg_th_code) {
          // if (ptr_upper_code[lane] < top_logical_bucket_id) {
          ptr_upper_code[lane] = std::clamp(ptr_upper_code[lane], 0, 255);
          upper_KNNs.push(ptr_upper_code[lane], ptr_lower_dist[lane],
                          idx + lane);
          // } else {
          //   float dist = sqr_dist<D>(query, ptr_data + lane * D);
          //   rerank_count++;
          //   int code = std::clamp(static_cast<int>((dist - lower) *
          //   inv_delta),
          //                         0, 255);
          //   if (code < lg_bucket_num) {
          //     upper_KNNs.push_exact(code, dist, (idx + lane));
          //   }
          // }
        } else {
          // if (ptr_lower_code[lane] < top_logical_bucket_id) {
          //   float dist = sqr_dist<D>(query, ptr_data + lane * D);
          //   rerank_count++;
          //   int code = std::clamp(static_cast<int>((dist - lower) *
          //   inv_delta),
          //                         0, 255);
          //   if (code < lg_bucket_num) {
          //     upper_KNNs.push_exact(code, dist, (idx + lane));
          //   }
          // } else {
          ptr_lower_code[lane] = std::clamp(ptr_lower_code[lane], 0, 255);
          upper_KNNs.push_lower(ptr_lower_code[lane], ptr_lower_dist[lane],
                                idx + lane);
          // }
        }
      }
      ptr_upper_dist += 8;
      ptr_lower_dist += 8;
      ptr_upper_code += 8;
      ptr_lower_code += 8;
      idx += 8;
      ptr_data += D * 8;
    }
  }

  {
    float *ptr_lower_dist = &lower_dist[0];
    float *ptr_upper_dist = &upper_dist[0];
    int32_t *ptr_lower_code = &lower_code[0];
    int32_t *ptr_upper_code = &upper_code[0];
    uint16_t PORTABLE_ALIGN32 result[SIZE];
    accumulate<B>(nblk_remain, packed_code, LUT, result);

    for (uint32_t i = 0; i < remain; ++i) {
      result_float[i] = static_cast<float>(static_cast<int>(result[i]) - sumq);
    }

    float *ptr_result_float = &result_float[0];

    for (int i = 0; i < remain; i++) {
      float tmp_dist = (*ptr_sqr_x) + sqr_y + (*ptr_factor_ppc) * vl +
                       result_float[i] * (*ptr_factor_ip) * width;
      float error_bound = y * (*ptr_error);
      *ptr_upper_dist = tmp_dist + error_bound;
      *ptr_lower_dist = tmp_dist - error_bound;
      *ptr_upper_code = ((*ptr_upper_dist) - lower) * inv_delta;
      *ptr_lower_code = ((*ptr_lower_dist) - lower) * inv_delta;
      ptr_result_float++;
      ptr_lower_dist++;
      ptr_upper_dist++;
      ptr_sqr_x++;
      ptr_factor_ip++;
      ptr_factor_ppc++;
      ptr_error++;
      ptr_upper_code++;
      ptr_lower_code++;
    }

    ptr_lower_dist = &lower_dist[0];
    ptr_upper_dist = &upper_dist[0];
    ptr_upper_code = &upper_code[0];
    ptr_lower_code = &lower_code[0];
    for (int i = 0; i < remain; i++) {
      // if (*ptr_upper_code < lg_th_code) {
      //   upper_KNNs.push(std::clamp(*ptr_upper_code, 0, lg_bucket_num - 1),
      //                   *ptr_lower_dist, idx);
      // } else if (*ptr_lower_code < lg_th_code) {
      //   upper_KNNs.push_lower(std::clamp(*ptr_lower_code, 0, lg_bucket_num -
      //   1),
      //                         *ptr_lower_dist, idx);
      // }
      if (*ptr_upper_code < lg_th_code) {
        // if (*ptr_upper_code < top_logical_bucket_id) {
        *ptr_upper_code = std::clamp(*ptr_upper_code, 0, 255);
        upper_KNNs.push(*ptr_upper_code, *ptr_lower_dist, idx);
        // } else {
        //   float dist = sqr_dist<D>(query, ptr_data);
        //   rerank_count++;
        //   int code =
        //       std::clamp(static_cast<int>((dist - lower) * inv_delta), 0,
        //       255);
        //   if (code < lg_bucket_num) {
        //     upper_KNNs.push_exact(code, dist, (idx));
        //   }
        // }
      } else if (*ptr_lower_code < lg_th_code) {
        // if (*ptr_lower_code < top_logical_bucket_id) {
        //   float dist = sqr_dist<D>(query, ptr_data);
        //   rerank_count++;
        //   int code =
        //       std::clamp(static_cast<int>((dist - lower) * inv_delta), 0,
        //       255);
        //   if (code < lg_bucket_num) {
        //     upper_KNNs.push_exact(code, dist, (idx));
        //   }
        // } else {
        *ptr_lower_code = std::clamp(*ptr_lower_code, 0, 255);
        upper_KNNs.push_lower(*ptr_lower_code, *ptr_lower_dist, idx);
        // }
      }
      ptr_lower_dist++;
      ptr_upper_dist++;
      ptr_upper_code++;
      ptr_lower_code++;
      idx += 1;
      ptr_data += D;
    }
  }
}

// Save and Load Functions
template <uint32_t D, uint32_t B> void IVFRN<D, B>::save(char *filename) {
  std::ofstream output(filename, std::ios::binary);

  uint32_t d = D;
  uint32_t b = B;
  output.write((char *)&N, sizeof(uint32_t));
  output.write((char *)&d, sizeof(uint32_t));
  output.write((char *)&C, sizeof(uint32_t));
  output.write((char *)&b, sizeof(uint32_t));

  output.write((char *)start, C * sizeof(uint32_t));
  output.write((char *)len, C * sizeof(uint32_t));
  output.write((char *)id, N * sizeof(uint32_t));
  output.write((char *)dist_to_c, N * sizeof(float));
  output.write((char *)x0, N * sizeof(float));

  output.write((char *)centroid, C * B * sizeof(float));
  output.write((char *)data, 1ull * N * D * sizeof(float));
  output.write((char *)binary_code, 1ull * N * B / 64 * sizeof(uint64_t));

  output.close();
  std::cerr << "Saved!" << std::endl;
}

// load impl
template <uint32_t D, uint32_t B> void IVFRN<D, B>::load(char *filename) {
  std::ifstream input(filename, std::ios::binary);

  if (!input.is_open())
    throw std::runtime_error("Cannot open file");

  uint32_t d;
  uint32_t b;
  input.read((char *)&N, sizeof(uint32_t));
  input.read((char *)&d, sizeof(uint32_t));
  input.read((char *)&C, sizeof(uint32_t));
  input.read((char *)&b, sizeof(uint32_t));

  std::cerr << d << std::endl;
  assert(d == D);
  assert(b == B);

  u = new float[B];
#if defined(RANDOM_QUERY_QUANTIZATION)
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> uniform(0.0, 1.0);
  for (int i = 0; i < B; i++)
    u[i] = uniform(gen);
#else
  for (int i = 0; i < B; i++)
    u[i] = 0.5;
#endif

  centroid = new float[C * B];
  // data = new float[1ull * N * D];
  data = static_cast<float *>(aligned_alloc(
      64, round_up_to_multiple(1ull * N * D * sizeof(float), 64)));

  size_t alloc_size =
      round_up_to_multiple(1ull * N * B / 64 * sizeof(uint64_t), 256);
  binary_code = static_cast<uint64_t *>(aligned_alloc(256, alloc_size));

  start = new uint32_t[C];
  len = new uint32_t[C];
  id = static_cast<uint32_t *>(
      aligned_alloc(32, round_up_to_multiple(N * sizeof(uint32_t), 32)));
  dist_to_c = static_cast<float *>(
      aligned_alloc(32, round_up_to_multiple(N * sizeof(float), 32)));
  x0 = static_cast<float *>(
      aligned_alloc(32, round_up_to_multiple(N * sizeof(float), 32)));

  sqr_x = static_cast<float *>(
      aligned_alloc(32, round_up_to_multiple(N * sizeof(float), 32)));
  factor_ip = static_cast<float *>(
      aligned_alloc(32, round_up_to_multiple(N * sizeof(float), 32)));
  factor_ppc = static_cast<float *>(
      aligned_alloc(32, round_up_to_multiple(N * sizeof(float), 32)));
  error = static_cast<float *>(
      aligned_alloc(32, round_up_to_multiple(N * sizeof(float), 32)));

  input.read((char *)start, C * sizeof(uint32_t));
  input.read((char *)len, C * sizeof(uint32_t));
  input.read((char *)id, N * sizeof(uint32_t));
  input.read((char *)dist_to_c, N * sizeof(float));
  input.read((char *)x0, N * sizeof(float));

  input.read((char *)centroid, C * B * sizeof(float));
  input.read((char *)data, 1ull * N * D * sizeof(float));
  input.read((char *)binary_code, 1ull * N * B / 64 * sizeof(uint64_t));

#if defined(FAST_SCAN)
  packed_start = new uint32_t[C];
  uint32_t cur = 0;
  for (int i = 0; i < C; i++) {
    packed_start[i] = cur;
    cur += (len[i] + 31) / 32 * 32 * B / 8;
  }
  packed_code = static_cast<uint8_t *>(
      aligned_alloc(32, round_up_to_multiple(cur * sizeof(uint8_t), 32)));
  for (int i = 0; i < C; i++) {
    pack_codes<B>(binary_code + 1ull * start[i] * (B / 64), len[i],
                  packed_code + packed_start[i]);
  }
#endif

  for (int i = 0; i < N; i++) {
    long double x_x0 = (long double)dist_to_c[i] / x0[i];
    sqr_x[i] = dist_to_c[i] * dist_to_c[i];
    error[i] =
        2 * max_x1 * std::sqrt(x_x0 * x_x0 - dist_to_c[i] * dist_to_c[i]);
    factor_ppc[i] =
        -2 / fac_norm * x_x0 *
        ((float)space.popcount(binary_code + static_cast<size_t>(i) * B / 64) *
             2 -
         B);
    factor_ip[i] = -2 / fac_norm * x_x0;
  }
  input.close();
  std::cerr << "IVFRN data loaded." << std::endl;
}

// ==============================================================================================================================
// Construction and Deconstruction Functions
template <uint32_t D, uint32_t B> IVFRN<D, B>::IVFRN() {
  N = C = 0;
  start = len = id = NULL;
  x0 = dist_to_c = centroid = data = NULL;
  binary_code = NULL;
  u = NULL;
}

template <uint32_t D, uint32_t B>
IVFRN<D, B>::IVFRN(const Matrix<float> &X, const Matrix<float> &_centroids,
                   const Matrix<float> &dist_to_centroid,
                   const Matrix<float> &_x0, const Matrix<uint32_t> &cluster_id,
                   const Matrix<uint64_t> &binary) {
  u = NULL;

  N = X.n;
  C = _centroids.n;

  // check uint64_t
  assert(B % 64 == 0);
  assert(B >= D);

  start = new uint32_t[C];
  len = new uint32_t[C];
  id = new uint32_t[N];
  dist_to_c = new float[N];
  x0 = new float[N];

  memset(len, 0, C * sizeof(uint32_t));
  for (int i = 0; i < N; i++)
    len[cluster_id.data[i]]++;
  int sum = 0;
  for (int i = 0; i < C; i++) {
    start[i] = sum;
    sum += len[i];
  }
  for (int i = 0; i < N; i++) {
    id[start[cluster_id.data[i]]] = i;
    dist_to_c[start[cluster_id.data[i]]] = dist_to_centroid.data[i];
    x0[start[cluster_id.data[i]]] = _x0.data[i];
    start[cluster_id.data[i]]++;
  }
  for (int i = 0; i < C; i++) {
    start[i] -= len[i];
  }

  centroid = new float[C * B];
  data = new float[1ull * N * D];
  binary_code = new uint64_t[1ull * N * B / 64];

  std::memcpy(centroid, _centroids.data, C * B * sizeof(float));
  float *data_ptr = data;
  uint64_t *binary_code_ptr = binary_code;

  for (int i = 0; i < N; i++) {
    int x = id[i];
    std::memcpy(data_ptr, X.data + 1ull * x * D, D * sizeof(float));
    std::memcpy(binary_code_ptr, binary.data + 1ull * x * (B / 64),
                (B / 64) * sizeof(uint64_t));
    data_ptr += D;
    binary_code_ptr += B / 64;
  }
}

template <uint32_t D, uint32_t B> IVFRN<D, B>::~IVFRN() {
  if (id != NULL)
    delete[] id;
  if (dist_to_c != NULL)
    delete[] dist_to_c;
  if (len != NULL)
    delete[] len;
  if (start != NULL)
    delete[] start;
  if (x0 != NULL)
    delete[] x0;
  if (data != NULL)
    delete[] data;
  if (u != NULL)
    delete[] u;
  if (binary_code != NULL)
    std::free(binary_code);
  // if(pack_codes  != NULL) std::free(pack_codes);
  if (centroid != NULL)
    std::free(centroid);
}

#pragma once
#include <chrono>
#include <fcntl.h>
#include <immintrin.h>
#include <libaio.h>
#include <limits>
#include <queue>
#include <unordered_set>
#ifndef WIN32
#include <sys/resource.h>
#endif
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))
#include "robin_hood.h"
#include <mutex>

class AsyncVectorMap {
private:
  int fd;
  io_context_t ctx;
  size_t D;
  size_t record_size; // 每条记录大小（纯 raw float: D * sizeof(float)）
  unsigned max_events; // 记录队列最大容量
  float *pool_base;
  size_t max_cache_size;
  size_t pool_used; // 当前 pool 已分配的槽数
  robin_hood::unordered_flat_map<uint32_t, uint32_t> id_to_ptr_;

  // 流水线状态：在 bound_fast_scan 完成后逐批提交的 iocb
  std::vector<struct iocb> pipeline_cbs_;
  std::vector<struct iocb *> pipeline_p_cbs_;
  std::vector<struct io_event> pipeline_events_;
  size_t pipeline_submitted_; // 已向内核提交的总数
  size_t pipeline_completed_; // 已收割完成的总数

public:
  AsyncVectorMap(const char *path, size_t cache_capacity, size_t d,
                 unsigned max_ctx = 4096)
      : D(d), max_events(max_ctx), max_cache_size(cache_capacity),
        pool_used(0), pipeline_submitted_(0), pipeline_completed_(0) {

    fd = open(path, O_RDONLY);
    if (fd < 0) {
      perror("Failed to open base file");
      exit(-1);
    }

    // 纯 raw float 格式（base_aligned_grouped.fvecs），无 per-record dim header
    record_size = D * sizeof(float);
    ctx = 0;
    if (io_setup(max_ctx, &ctx) != 0) {
      perror("io_setup failed");
      exit(-1);
    }

    size_t total_bytes = max_cache_size * D * sizeof(float);
    if (posix_memalign((void **)&pool_base, 64, total_bytes) != 0) {
      perror("Pool allocation failed");
      exit(-1);
    }

    id_to_ptr_.reserve(cache_capacity);
    pipeline_cbs_.reserve(cache_capacity);
    pipeline_p_cbs_.reserve(cache_capacity);
    pipeline_events_.resize(cache_capacity);
  }

  void reset() {
    id_to_ptr_.clear();
    pool_used = 0;
    pipeline_submitted_ = 0;
    pipeline_completed_ = 0;
    pipeline_cbs_.clear();
    pipeline_p_cbs_.clear();
  }

  inline float *get_vector(uint32_t id) const {
    auto it = id_to_ptr_.find(id);
    if (it != id_to_ptr_.end()) {
      return pool_base + (static_cast<size_t>(it->second) * D);
    } else {
      std::cerr << "Error: AsyncVectorMap miss id " << id << std::endl;
      exit(-1);
    }
  }

  // -----------------------------------------------------------------------
  // 流水线接口：在每个 cluster 的 bound_fast_scan 完成后调用
  // 传入该 cluster 的候选 global_ids（已做 id[] 转换），立即提交 AIO 请求
  // -----------------------------------------------------------------------
  void submit_ids_async(const std::vector<uint32_t> &global_ids) {
    for (uint32_t gid : global_ids) {
      if (id_to_ptr_.count(gid))
        continue; // 已经在 pool 中，跳过

      size_t slot = pool_used++;
      float *target_ptr = pool_base + slot * D;
      size_t offset = 1ULL * gid * record_size;

      pipeline_cbs_.emplace_back();
      struct iocb &cb = pipeline_cbs_.back();
      io_prep_pread(&cb, fd, target_ptr, D * sizeof(float), offset);
      cb.data = (void *)(uintptr_t)gid;
      pipeline_p_cbs_.push_back(&cb);

      // 标记 slot，防止同一 id 重复分配
      id_to_ptr_[gid] = static_cast<uint32_t>(slot);
    }

    // 将新增的、尚未提交的 iocb 批量提交给内核
    size_t total = pipeline_p_cbs_.size();
    while (pipeline_submitted_ < total) {
      size_t in_flight = pipeline_submitted_ - pipeline_completed_;
      if (in_flight >= max_events) break; // 队列满，等收割后再提交

      size_t to_submit =
          std::min(total - pipeline_submitted_, max_events - in_flight);
      int ret = io_submit(ctx, (long)to_submit,
                          pipeline_p_cbs_.data() + pipeline_submitted_);
      if (ret > 0) {
        pipeline_submitted_ += ret;
      } else if (ret == -EAGAIN) {
        break;
      } else {
        fprintf(stderr, "io_submit fatal: %s\n", strerror(-ret));
        exit(-1);
      }
    }
  }

  // -----------------------------------------------------------------------
  // 流水线结束：等待所有已提交的 IO 完成
  // -----------------------------------------------------------------------
  void wait_all() {
    // 先把剩余未提交的全部提交
    size_t total = pipeline_p_cbs_.size();
    while (pipeline_submitted_ < total) {
      size_t in_flight = pipeline_submitted_ - pipeline_completed_;
      // 先收割一些，腾出队列空间
      if (in_flight >= max_events) {
        int r = io_getevents(ctx, 1, (int)in_flight,
                             pipeline_events_.data() + pipeline_completed_,
                             nullptr);
        if (r > 0) pipeline_completed_ += r;
        else if (r < 0) { fprintf(stderr, "io_getevents fatal: %s\n", strerror(-r)); exit(-1); }
        in_flight = pipeline_submitted_ - pipeline_completed_;
      }
      size_t to_submit =
          std::min(total - pipeline_submitted_, max_events - in_flight);
      int ret = io_submit(ctx, (long)to_submit,
                          pipeline_p_cbs_.data() + pipeline_submitted_);
      if (ret > 0) pipeline_submitted_ += ret;
      else if (ret == -EAGAIN) {
        // 收割后重试
        int r = io_getevents(ctx, 1, (int)(pipeline_submitted_ - pipeline_completed_),
                             pipeline_events_.data() + pipeline_completed_, nullptr);
        if (r > 0) pipeline_completed_ += r;
      } else {
        fprintf(stderr, "io_submit fatal: %s\n", strerror(-ret));
        exit(-1);
      }
    }

    // 等待所有已提交的完成
    while (pipeline_completed_ < pipeline_submitted_) {
      int r = io_getevents(ctx, 1,
                           (int)(pipeline_submitted_ - pipeline_completed_),
                           pipeline_events_.data() + pipeline_completed_,
                           nullptr);
      if (r > 0) {
        pipeline_completed_ += r;
      } else if (r < 0) {
        fprintf(stderr, "io_getevents fatal: %s\n", strerror(-r));
        exit(-1);
      }
    }
    // 注意：id_to_ptr_ 已在 submit_ids_async 中提前写入，无需再建立映射
  }

  // -----------------------------------------------------------------------
  // 原有批量接口（保留兼容）
  // -----------------------------------------------------------------------
  void fetch_to_cache(const std::vector<uint32_t> &miss_ids) {
    if (miss_ids.empty())
      return;
    size_t n = miss_ids.size();
    std::vector<struct iocb> cbs(n);
    std::vector<struct iocb *> p_cbs(n);
    for (size_t i = 0; i < n; ++i) {
      float *target_ptr = pool_base + (i * D);
      size_t offset = 1ULL * miss_ids[i] * record_size;
      io_prep_pread(&cbs[i], fd, target_ptr, D * sizeof(float), offset);
      cbs[i].data = (void *)(uintptr_t)miss_ids[i];
      p_cbs[i] = &cbs[i];
    }
    const size_t BATCH_SIZE = max_events;
    size_t submitted = 0;
    size_t completed = 0;
    std::vector<struct io_event> events(n);

    while (completed < n) {
      while (submitted < n && (submitted - completed) < BATCH_SIZE) {
        size_t to_submit =
            std::min(n - submitted, BATCH_SIZE - (submitted - completed));
        int ret = io_submit(ctx, to_submit, p_cbs.data() + submitted);
        if (ret < 0) {
          if (ret == -EAGAIN) break;
          fprintf(stderr, "io_submit fatal: %s\n", strerror(-ret));
          exit(-1);
        }
        submitted += ret;
      }
      if (completed < submitted) {
        int r = io_getevents(ctx, 1, submitted - completed,
                             events.data() + completed, nullptr);
        if (r > 0) completed += r;
        else if (r < 0) { fprintf(stderr, "io_getevents fatal: %s\n", strerror(-r)); exit(-1); }
      }
    }
    for (size_t i = 0; i < n; ++i) {
      uint32_t id = (uint32_t)(uintptr_t)events[i].obj->data;
      float *buf_ptr = (float *)events[i].obj->u.c.buf;
      uint32_t pool_idx = (buf_ptr - pool_base) / D;
      id_to_ptr_[id] = pool_idx;
    }
  }

  //   void fetch_vectors(const std::vector<uint32_t> &ids, float *data_ptr) {
  //     size_t total_n = ids.size();
  //     if (total_n == 0)
  //       return;

  //     // 分批策略：每批次大小不能超过 max_events
  //     // 建议设置为 1024 或 2048，留一点缓冲空间
  //     //   const size_t BATCH_SIZE = std::min((size_t)max_events,
  //     (size_t)4096);

  //     std::vector<struct iocb> cbs(total_n);
  //     std::vector<struct iocb *> p_cbs(total_n);
  //     std::vector<struct io_event> events(total_n);

  //     //   size_t processed = 0;

  //     //   while (processed < total_n) {
  //     //     size_t current_batch_size = std::min(BATCH_SIZE, total_n -
  //     //     processed);

  //     // 2. 准备 IOCB
  //     for (size_t i = 0; i < total_n; ++i) {
  //       long long offset = (long long)ids[i] * record_size +
  //       sizeof(unsigned);

  //       void *target_buffer = data_ptr + (1ull * idx * D);

  //       memset(&cbs[i], 0, sizeof(struct iocb));
  //       io_prep_pread(&cbs[i], fd, target_buffer, D * sizeof(float), offset);
  //       p_cbs[i] = &cbs[i];
  //     }

  //     int ret = io_submit(ctx, current_batch_size, p_cbs.data());

  //     if (ret < 0) {
  //       perror("io_submit failed"); // 可能是 -EAGAIN (队列满)
  //       break;
  //     }

  //     int submitted_count = ret; // 实际提交成功的数量
  //     int completed_so_far = 0;
  //     while (completed_so_far < submitted_count) {
  //       int wait_nr = submitted_count - completed_so_far;
  //       int r = io_getevents(ctx, wait_nr, wait_nr, events.data(), nullptr);

  //       if (r < 0) {
  //         perror("io_getevents failed");
  //         break;
  //       }
  //       completed_so_far += r;
  //     }
  //     // processed += submitted_count;
  //     //   }
  //   }

  ~AsyncVectorMap() {
    if (pool_base)
      free(pool_base);
    io_destroy(ctx);
    close(fd);
  }
};

typedef std::pair<float, uint32_t> Result;
typedef std::priority_queue<Result> ResultHeap;

constexpr uint32_t FLAG_MASK = 0x8000'0000u;  // 最高位
constexpr uint32_t VALUE_MASK = 0x7FFF'FFFFu; // 低31位

// === 打包：最高位置1（即标记为 exact sorted）===
inline uint32_t pack(uint32_t value31) {
  // 置最高位为1
  return value31 | FLAG_MASK;
}

// === 解包：判断是否 exact（最高位是不是1）===
inline bool isExact(uint32_t packed) { return (packed & FLAG_MASK) != 0; }

// === 解包：取低31位（把最高位清0，直接返回 value 部分）===
inline uint32_t getValue(uint32_t packed) { return packed & VALUE_MASK; }

template <typename KeyType, typename DataType> class MaxHeap {
public:
  class Item {
  public:
    KeyType key;
    DataType data;
    Item() {}
    Item(const KeyType &k, const DataType &d, bool f = false)
        : key(k), data(d) {}

    bool operator<(const Item &i2) const { return key < i2.key; }
  };
  using value_type = DataType;

  const Item &top_item() const { return v_[0]; }

  const DataType &top_data() const { return v_[0].data; }

  DataType &top_data() { return v_[0].data; }

  KeyType top_key() const { return this->v_[0].key; }

  bool empty() const { return num_elements_ == 0; }

  void pop() {
    num_elements_ -= 1;
    v_[0] = v_[num_elements_];
    heap_down(0);
  }

  void push_unsorted(const KeyType key, const DataType data) {
    if (v_.size() == static_cast<uint32_t>(num_elements_)) {
      v_.emplace_back(Item(key, data));
    } else {
      v_[num_elements_].key = key;
      v_[num_elements_].data = data;
    }
    num_elements_ += 1;
  }

  void push(const KeyType key, const DataType data) {
    push_unsorted(key, data);
    heap_up(num_elements_ - 1);
  }

  void heapify() {
    int_fast32_t rightmost = parent(num_elements_ - 1);
    for (int_fast32_t cur_loc = rightmost; cur_loc >= 0; --cur_loc) {
      heap_down(cur_loc);
    }
  }

  void reset() { num_elements_ = 0; }

  int_fast32_t size() const { return num_elements_; }

  void resize(uint32_t new_size) { v_.resize(new_size); }

  void replace_top(const KeyType &key, const DataType &data) {
    this->v_[0].key = key;
    this->v_[0].data = data;
    this->heap_down(0);
  }

  void replace_top_key(const KeyType &key) {
    this->v_[0].key = key;
    this->heap_down(0);
  }

  const std::vector<Item> &get_data() const { return this->v_; }

protected:
  int_fast32_t lchild(int_fast32_t x) { return 2 * x + 1; }

  int_fast32_t rchild(int_fast32_t x) { return 2 * x + 2; }

  int_fast32_t parent(int_fast32_t x) { return (x - 1) / 2; }

  void swap_entries(int_fast32_t a, int_fast32_t b) {
    Item tmp = v_[a];
    v_[a] = v_[b];
    v_[b] = tmp;
  }

  void heap_up(int_fast32_t cur_loc) {
    int_fast32_t p = parent(cur_loc);
    while (cur_loc > 0 && v_[p].key < v_[cur_loc].key) {
      swap_entries(p, cur_loc);
      cur_loc = p;
      p = parent(cur_loc);
    }
  }

  void heap_down(int_fast32_t cur_loc) {
    while (true) {
      int_fast32_t lc = lchild(cur_loc);
      int_fast32_t rc = rchild(cur_loc);
      if (lc >= num_elements_) {
        return;
      }

      if (v_[cur_loc].key >= v_[lc].key) {
        if (rc >= num_elements_ || v_[cur_loc].key >= v_[rc].key) {
          return;
        } else {
          swap_entries(cur_loc, rc);
          cur_loc = rc;
        }
      } else {
        if (rc >= num_elements_ || v_[lc].key >= v_[rc].key) {
          swap_entries(cur_loc, lc);
          cur_loc = lc;
        } else {
          swap_entries(cur_loc, rc);
          cur_loc = rc;
        }
      }
    }
  }

  std::vector<Item> v_;
  uint32_t num_elements_ = 0;
};

struct PairBucket {
  std::vector<float> val;    // 距离值
  std::vector<uint32_t> idx; // 对应 ID

  /* -------- 预分配 -------- */
  void reserve(std::size_t new_cap) {
    val.reserve(new_cap);
    idx.reserve(new_cap);
  }

  /* -------- 插入 -------- */
  inline void emplace(float v, uint32_t id_) {
    if (val.size() == val.capacity()) {
      // 容量不足时翻倍（首次至少 8）
      std::size_t new_cap = val.capacity() * 2;
      reserve(new_cap);
    }
    val.push_back(v);
    idx.push_back(id_);
  }

  inline void clear() {
    val.clear();
    idx.clear();
  }

  std::size_t size() const { return val.size(); }
  bool empty() const { return val.empty(); }

  const float *val_data() const { return val.data(); }
  const uint32_t *idx_data() const { return idx.data(); }
  float *data_val() { return val.data(); }
  uint32_t *data_idx() { return idx.data(); }
  void free_memory() {
    val.clear();
    idx.clear();
    // val.shrink_to_fit();
    // idx.shrink_to_fit();
  }

  PairBucket() = default;
};

struct PairIOBucket {
  std::vector<float> val;         // 距离值
  std::vector<uint32_t> idx;      // 对应 ID
  std::vector<float *> data_ptrs; // 对应数据的指针

  /* -------- 预分配 -------- */
  void reserve(std::size_t new_cap) {
    val.reserve(new_cap);
    idx.reserve(new_cap);
    data_ptrs.reserve(new_cap);
  }

  /* -------- 插入 (默认指针为 null) -------- */
  inline void emplace(float v, uint32_t id_) {
    // 注：std::vector 内部会自动处理扩容，
    // 但如果你想手动控制扩容逻辑，可以保留这段：
    if (val.size() == val.capacity()) {
      std::size_t new_cap = val.capacity() == 0 ? 8 : val.capacity() * 2;
      reserve(new_cap);
    }

    val.push_back(v);
    idx.push_back(id_);
    data_ptrs.push_back(nullptr); // 初始化为 nullptr，方便后续判断
  }

  /* -------- 更新特定位置的指针 -------- */
  inline void set_ptr(std::size_t pos, float *ptr) {
    if (pos < data_ptrs.size()) {
      data_ptrs[pos] = ptr;
    }
  }

  inline void clear() {
    val.clear();
    idx.clear();
    data_ptrs.clear();
  }

  std::size_t size() const { return val.size(); }
  bool empty() const { return val.empty(); }

  const float *val_data() const { return val.data(); }
  const uint32_t *idx_data() const { return idx.data(); }
  float *const *ptr_data() const { return data_ptrs.data(); }

  void free_memory() {
    val.clear();
    idx.clear();
    data_ptrs.clear();
    // 如果需要彻底释放内存（归还给系统）：
  }

  // 修复构造函数名称与结构体名一致
  PairIOBucket() = default;
};

// struct QSPairBucket {
//   std::vector<float> upper_val; // 距离值
//   std::vector<float> lower_val; // 距离值
//   std::vector<uint32_t> idx;    // 对应 ID

//   /* -------- 预分配 -------- */
//   void reserve(std::size_t new_cap) {
//     lower_val.reserve(new_cap);
//     upper_val.reserve(new_cap);
//     idx.reserve(new_cap);
//   }

//   /* -------- 插入 -------- */
//   inline void emplace(float lb, float ub, uint32_t id_) {
//     if (lower_val.size() == lower_val.capacity()) {
//       // 容量不足时翻倍（首次至少 8）
//       std::size_t new_cap = lower_val.capacity() * 2;
//       reserve(new_cap);
//     }
//     lower_val.push_back(lb);
//     upper_val.push_back(ub);
//     idx.push_back(id_);
//   }

//   inline void clear() {
//     lower_val.clear();
//     upper_val.clear();
//     idx.clear();
//   }

//   std::size_t size() const { return idx.size(); }
//   bool empty() const { return idx.empty(); }

//   const float *upper_val_data() const { return upper_val.data(); }
//   const float *lower_val_data() const { return lower_val.data(); }
//   const uint32_t *idx_data() const { return idx.data(); }

//   void free_memory() {
//     upper_val.clear();
//     lower_val.clear();
//     idx.clear();
//   }

//   QSPairBucket() = default;
// };

class TopKBufferSoA {
public:
  TopKBufferSoA(uint32_t k,                   // top-k
                uint32_t physical_bucket_num, // 物理桶数
                uint32_t logical_bucket_num)  // 桶数
      : k_(k), physical_bucket_num_(physical_bucket_num),
        logical_bucket_num_(logical_bucket_num) {
    bucketed_buffer_.resize(physical_bucket_num_);
    exact_buffer_.resize(physical_bucket_num_);
    lower_buffer_.resize(physical_bucket_num_);
    code_lut_ = new PORTABLE_ALIGN64 uint8_t[logical_bucket_num_];
    for (uint32_t i = 0; i < physical_bucket_num_; ++i) {
      bucketed_buffer_[i].reserve(k_ * 5);
      exact_buffer_[i].reserve(k_ * 5);
      lower_buffer_[i].reserve(k_ * 5);
    }
    logical_threshold_bucket_id_ = logical_bucket_num_;   // 最右桶
    physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
  }

  uint8_t *get_code_lut() { return code_lut_; }

  void set_bounds(float lowest, float upper, float delta) // ← 多一个阈值数组
  {
    lower_ = lowest;
    upper_ = upper;
    delta_ = delta;
    logical_threshold_bucket_id_ = 1e7;                   // 最右桶
    physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
  }

  inline void push(uint32_t b_logical, float lb, uint32_t id) {
    const uint32_t b = code_lut_[b_logical]; // physical bucket id
    PairBucket &bucket = bucketed_buffer_[b];
    bucket.emplace(lb, id);
  }

  inline void push_exact(uint32_t b_logical, float lb, uint32_t id) {
    const uint32_t b = code_lut_[b_logical]; // physical bucket id
    PairBucket &bucket = exact_buffer_[b];
    bucket.emplace(lb, id);
  }

  inline void push_lower(uint32_t b_logical, float lb, uint32_t id) {
    const uint32_t b = code_lut_[b_logical]; // physical bucket id
    PairBucket &bucket = lower_buffer_[b];
    bucket.emplace(lb, id);
  }

  void reset() {
    for (auto &B : bucketed_buffer_)
      B.clear();
    for (auto &B : exact_buffer_)
      B.clear();
    for (auto &B : lower_buffer_)
      B.clear();
    logical_threshold_bucket_id_ = 1e7;
    physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
    // predict_logical_bucket_id_ = 0;
  }

  void update_th_code() {
    uint32_t acc = 0;
    uint32_t j = 0;
    for (uint32_t i = 0; i < physical_bucket_num_; ++i) {
      acc += static_cast<uint32_t>(bucketed_buffer_[i].size()) +
             static_cast<uint32_t>(exact_buffer_[i].size());

      while (j < logical_bucket_num_ && code_lut_[j] == i) {
        ++j;
      }

      if (acc >= k_) {                         // 一旦达到 k_
        logical_threshold_bucket_id_ = j;      // 当前 pseudo id 即阈值
        physical_threshold_bucket_id_ = i + 1; // 当前物理桶 id
        return;
      }
    }
    logical_threshold_bucket_id_ = 1e7;
    physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
  }

  uint32_t get_logical_bucket_num() const { return logical_bucket_num_; }
  uint32_t get_physical_bucket_num() const { return physical_bucket_num_; }
  uint32_t get_logical_threshold_bucket_id() const {
    return logical_threshold_bucket_id_;
  }
  uint32_t get_physical_threshold_bucket_id() const {
    return physical_threshold_bucket_id_;
  }
  uint32_t get_predict_logical_bucket_id() const {
    return predict_logical_bucket_id_;
  }
  uint32_t get_predict_physical_bucket_id() const {
    return code_lut_[predict_logical_bucket_id_];
  }

  void set_predict_logical_bucket_id(uint32_t id) {
    predict_logical_bucket_id_ = id;
  }

  float get_delta() const { return delta_; }
  float get_lower() const { return lower_; }
  float get_upper() const { return upper_; }

  auto &get_buffer() { return bucketed_buffer_; }
  auto &get_exact_buffer() { return exact_buffer_; }
  auto &get_lower_buffer() { return lower_buffer_; }

private:
  uint32_t k_;
  uint32_t predict_logical_bucket_id_;
  uint32_t logical_bucket_num_;
  uint32_t logical_threshold_bucket_id_;
  uint32_t physical_bucket_num_;
  uint32_t physical_threshold_bucket_id_;
  float lower_{}, upper_{}, delta_{};
  PORTABLE_ALIGN64 uint8_t *code_lut_;
  std::vector<PairBucket> bucketed_buffer_;
  std::vector<PairBucket> exact_buffer_;
  std::vector<PairBucket> lower_buffer_;
};

template <uint32_t B> struct ProbeInfo {
  float sqr_y;
  float vl;
  float width;
  uint32_t sum_q;
  uint32_t cluster_id;
#if defined(FAST_SCAN)
  uint8_t PORTABLE_ALIGN32 LUT[B / 4 * 16]; // aligned for SIMD
#endif
};

// class TopKBufferSoAWithIO {
// public:
//   TopKBufferSoAWithIO(uint32_t k,                   // top-k
//                       uint32_t physical_bucket_num, // 物理桶数
//                       uint32_t logical_bucket_num)  // 桶数
//       : k_(k), physical_bucket_num_(physical_bucket_num),
//         logical_bucket_num_(logical_bucket_num) {
//     bucketed_buffer_.resize(physical_bucket_num_);
//     exact_buffer_.resize(physical_bucket_num_);
//     lower_buffer_.resize(physical_bucket_num_);
//     code_lut_ = new PORTABLE_ALIGN64 uint8_t[logical_bucket_num_];
//     for (uint32_t i = 0; i < physical_bucket_num_; ++i) {
//       bucketed_buffer_[i].reserve(k_ * 5);
//       exact_buffer_[i].reserve(k_ * 5);
//       lower_buffer_[i].reserve(k_ * 5);
//     }
//     logical_threshold_bucket_id_ = logical_bucket_num_;   // 最右桶
//     physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
//   }

//   uint8_t *get_code_lut() { return code_lut_; }

//   void set_bounds(float lowest, float upper, float delta) // ← 多一个阈值数组
//   {
//     lower_ = lowest;
//     upper_ = upper;
//     delta_ = delta;
//     logical_threshold_bucket_id_ = 1e7;                   // 最右桶
//     physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
//   }

//   inline void push(uint32_t b_logical, float lb, uint32_t id) {
//     const uint32_t b = code_lut_[b_logical]; // physical bucket id
//     PairIOBucket &bucket = bucketed_buffer_[b];
//     bucket.emplace(lb, id);
//   }

//   inline void push_exact(uint32_t b_logical, float lb, uint32_t id) {
//     const uint32_t b = code_lut_[b_logical]; // physical bucket id
//     PairIOBucket &bucket = exact_buffer_[b];
//     bucket.emplace(lb, id);
//   }

//   inline void push_lower(uint32_t b_logical, float lb, uint32_t id) {
//     const uint32_t b = code_lut_[b_logical]; // physical bucket id
//     PairIOBucket &bucket = lower_buffer_[b];
//     bucket.emplace(lb, id);
//   }

//   void reset() {
//     for (auto &B : bucketed_buffer_)
//       B.clear();
//     for (auto &B : exact_buffer_)
//       B.clear();
//     for (auto &B : lower_buffer_)
//       B.clear();
//     logical_threshold_bucket_id_ = 1e7;
//     physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
//     // predict_logical_bucket_id_ = 0;
//   }

//   void update_th_code() {
//     uint32_t acc = 0;
//     uint32_t j = 0;
//     for (uint32_t i = 0; i < physical_bucket_num_; ++i) {
//       acc += static_cast<uint32_t>(bucketed_buffer_[i].size()) +
//              static_cast<uint32_t>(exact_buffer_[i].size());

//       while (j < logical_bucket_num_ && code_lut_[j] == i) {
//         ++j;
//       }

//       if (acc >= k_) {                         // 一旦达到 k_
//         logical_threshold_bucket_id_ = j;      // 当前 pseudo id 即阈值
//         physical_threshold_bucket_id_ = i + 1; // 当前物理桶 id
//         return;
//       }
//     }
//     logical_threshold_bucket_id_ = 1e7;
//     physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
//   }

//   uint32_t get_logical_bucket_num() const { return logical_bucket_num_; }
//   uint32_t get_physical_bucket_num() const { return physical_bucket_num_; }
//   uint32_t get_logical_threshold_bucket_id() const {
//     return logical_threshold_bucket_id_;
//   }
//   uint32_t get_physical_threshold_bucket_id() const {
//     return physical_threshold_bucket_id_;
//   }
//   uint32_t get_predict_logical_bucket_id() const {
//     return predict_logical_bucket_id_;
//   }
//   uint32_t get_predict_physical_bucket_id() const {
//     return code_lut_[predict_logical_bucket_id_];
//   }

//   void set_predict_logical_bucket_id(uint32_t id) {
//     predict_logical_bucket_id_ = id;
//   }

//   float get_delta() const { return delta_; }
//   float get_lower() const { return lower_; }
//   float get_upper() const { return upper_; }

//   auto &get_buffer() { return bucketed_buffer_; }
//   auto &get_exact_buffer() { return exact_buffer_; }
//   auto &get_lower_buffer() { return lower_buffer_; }

// private:
//   uint32_t k_;
//   uint32_t predict_logical_bucket_id_;
//   uint32_t logical_bucket_num_;
//   uint32_t logical_threshold_bucket_id_;
//   uint32_t physical_bucket_num_;
//   uint32_t physical_threshold_bucket_id_;
//   float lower_{}, upper_{}, delta_{};
//   PORTABLE_ALIGN64 uint8_t *code_lut_;
//   std::vector<PairIOBucket> bucketed_buffer_;
//   std::vector<PairIOBucket> exact_buffer_;
//   std::vector<PairIOBucket> lower_buffer_;
// };

static inline void prefetch_l1(const void *addr) {
#if defined(__SSE2__)
  _mm_prefetch(addr, _MM_HINT_T0);
#else
  __builtin_prefetch(addr, 0, 3);
#endif
}

static inline void prefetch_l2(const void *addr) {
#if defined(__SSE2__)
  _mm_prefetch((const char *)addr, _MM_HINT_T1);
#else
  __builtin_prefetch(addr, 0, 2);
#endif
}

inline void mem_prefetch_l1(const char *ptr, uint32_t num_lines) {
  switch (num_lines) {
  default:
    [[fallthrough]];
  case 20:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 19:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 18:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 17:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 16:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 15:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 14:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 13:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 12:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 11:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 10:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 9:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 8:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 7:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 6:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 5:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 4:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 3:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 2:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 1:
    prefetch_l1(ptr);
    ptr += 64;
    [[fallthrough]];
  case 0:
    break;
  }
}

inline void mem_prefetch_l2(const char *ptr, uint32_t num_lines) {
  switch (num_lines) {
  default:
    [[fallthrough]];
  case 20:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 19:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 18:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 17:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 16:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 15:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 14:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 13:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 12:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 11:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 10:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 9:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 8:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 7:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 6:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 5:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 4:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 3:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 2:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 1:
    prefetch_l2(ptr);
    ptr += 64;
    [[fallthrough]];
  case 0:
    break;
  }
}

namespace Detail {
double constexpr sqrtNewtonRaphson(double x, double curr, double prev) {
  return curr == prev ? curr
                      : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
}
} // namespace Detail

double constexpr const_sqrt(double x) {
  return x >= 0 && x < std::numeric_limits<double>::infinity()
             ? Detail::sqrtNewtonRaphson(x, x, 0)
             : std::numeric_limits<double>::quiet_NaN();
}

void print_binary(uint64_t v) {
  for (int i = 0; i < 64; i++) {
    std::cerr << ((v >> (63 - i)) & 1);
  }
}

void print_binary(uint8_t v) {
  for (int i = 0; i < 8; i++) {
    std::cerr << ((v >> (7 - i)) & 1);
  }
}

inline uint32_t reverseBits(uint32_t n) {
  n = (n >> 1) & 0x55555555 | (n << 1) & 0xaaaaaaaa;
  n = (n >> 2) & 0x33333333 | (n << 2) & 0xcccccccc;
  n = (n >> 4) & 0x0f0f0f0f | (n << 4) & 0xf0f0f0f0;
  n = (n >> 8) & 0x00ff00ff | (n << 8) & 0xff00ff00;
  n = (n >> 16) & 0x0000ffff | (n << 16) & 0xffff0000;
  return n;
}

ResultHeap getGroundtruth(const Matrix<float> &X, const Matrix<float> &Q,
                          size_t query, unsigned *groundtruth, size_t k) {
  ResultHeap ret;
  for (int i = 0; i < k; i++) {
    unsigned gt = groundtruth[i];
    ret.push(std::make_pair(Q.dist(query, X, gt), gt));
  }
  return ret;
}

float getRatio(int q, const Matrix<float> &Q, const Matrix<float> &X,
               const Matrix<unsigned> &G, ResultHeap KNNs) {
  ResultHeap gt;
  int k = KNNs.size();
  for (int i = 0; i < k; i++) {
    gt.emplace(Q.dist(q, X, G.data[q * G.d + i]), G.data[q * G.d + i]);
  }
  long double ret = 0;
  int valid_k = 0;
  while (gt.size()) {
    if (gt.top().first > 1e-5) {
      ret += std::sqrt(KNNs.top().first / gt.top().first);
      valid_k++;
    }
    gt.pop();
    KNNs.pop();
  }
  if (valid_k == 0)
    return 1.0 * k;
  return ret / valid_k * k;
}

int getRecall(ResultHeap &result, ResultHeap &gt) {
  int correct = 0;

  std::unordered_set<unsigned> g;
  int ret = 0;

  while (gt.size()) {
    g.insert(gt.top().second);
    // std::cerr << "ID - " << gt.top().second << " dist - " << gt.top().first
    // << std::endl;
    gt.pop();
  }

  while (result.size()) {
    // std::cerr << "ID - " << result.top().second << " dist - " <<
    // result.top().first << std::endl;
    if (g.find(result.top().second) != g.end()) {
      ret++;
    }
    result.pop();
  }

  return ret;
}

#ifndef WIN32
void GetCurTime(rusage *curTime) {
  int ret = getrusage(RUSAGE_SELF, curTime);
  if (ret != 0) {
    fprintf(stderr,
            "The running time info couldn't be collected successfully.\n");
    // FreeData( 2);
    exit(0);
  }
}

/*
 * GetTime is used to get the 'float' format time from the start and end rusage
 * structure.
 *
 * @Param timeStart, timeEnd indicate the two time points.
 * @Param userTime, sysTime get back the time information.
 *
 * @Return void.
 */
void GetTime(struct rusage *timeStart, struct rusage *timeEnd, float *userTime,
             float *sysTime) {
  (*userTime) =
      ((float)(timeEnd->ru_utime.tv_sec - timeStart->ru_utime.tv_sec)) +
      ((float)(timeEnd->ru_utime.tv_usec - timeStart->ru_utime.tv_usec)) * 1e-6;
  (*sysTime) =
      ((float)(timeEnd->ru_stime.tv_sec - timeStart->ru_stime.tv_sec)) +
      ((float)(timeEnd->ru_stime.tv_usec - timeStart->ru_stime.tv_usec)) * 1e-6;
}

#endif

#if defined(_WIN32)
#include <psapi.h>
#include <windows.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) ||                 \
    (defined(__APPLE__) && defined(__MACH__))

#include <sys/resource.h>
#include <unistd.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) ||                              \
    (defined(__sun__) || defined(__sun) ||                                     \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) ||              \
    defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
size_t getPeakRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) ||                              \
    (defined(__sun__) || defined(__sun) ||                                     \
     defined(sun) && (defined(__SVR4) || defined(__svr4__)))
  /* AIX and Solaris ------------------------------------------ */
  struct psinfo psinfo;
  int fd = -1;
  if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
    return (size_t)0L; /* Can't open? */
  if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
    close(fd);
    return (size_t)0L; /* Can't read? */
  }
  close(fd);
  return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) ||                 \
    (defined(__APPLE__) && defined(__MACH__))
  /* BSD, Linux, and OSX -------------------------------------- */
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
  return (size_t)rusage.ru_maxrss;
#else
  return (size_t)(rusage.ru_maxrss * 1024L);
#endif

#else
  /* Unknown OS ----------------------------------------------- */
  return (size_t)0L; /* Unsupported. */
#endif
}

/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
size_t getCurrentRSS() {
#if defined(_WIN32)
  /* Windows -------------------------------------------------- */
  PROCESS_MEMORY_COUNTERS info;
  GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
  return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
  /* OSX ------------------------------------------------------ */
  struct mach_task_basic_info info;
  mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                &infoCount) != KERN_SUCCESS)
    return (size_t)0L; /* Can't access? */
  return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) ||              \
    defined(__gnu_linux__)
  /* Linux ---------------------------------------------------- */
  long rss = 0L;
  FILE *fp = NULL;
  if ((fp = fopen("/proc/self/statm", "r")) == NULL)
    return (size_t)0L; /* Can't open? */
  if (fscanf(fp, "%*s%ld", &rss) != 1) {
    fclose(fp);
    return (size_t)0L; /* Can't read? */
  }
  fclose(fp);
  return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);

#else
  /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
  return (size_t)0L; /* Unsupported. */
#endif
}
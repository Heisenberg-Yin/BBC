#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <libaio.h>
#include <liburing.h>
#include <limits>
#include <sstream>
#include <unistd.h>
#include <unordered_map>
#include <vector>
#include <x86intrin.h>
#ifndef WIN32
#include <sys/resource.h>
#endif
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

// 定义上下文结构，用于追踪每个 Bucket 的读取进度
struct BucketContext {
  float *buffer;
  uint32_t completed_cnt;
  uint32_t capability;
  uint32_t dim; // 记录维度，方便内部偏移计算

  BucketContext(uint32_t cap, uint32_t d)
      : buffer(nullptr), capability(cap), dim(d), completed_cnt(0) {
    size_t alignment = 4096;
    size_t raw_bytes = 1ull * capability * dim * sizeof(float);

    // 关键修正：确保 total_bytes 是 alignment 的倍数
    size_t total_bytes = (raw_bytes + alignment - 1) & ~(alignment - 1);

    buffer = static_cast<float *>(std::aligned_alloc(alignment, total_bytes));
    if (!buffer) {
      throw std::bad_alloc();
    }
    // 建议清零，避免脏数据
    memset(buffer, 0, total_bytes);
  }

  // 增加析构函数防止泄露
  ~BucketContext() {
    if (buffer) {
      free(buffer); // posix_memalign 分配的内存必须用 free 释放
      buffer = nullptr;
    }
  }

  // 提供一个辅助函数计算偏移量
  float *get_vector_ptr(uint32_t idx) {
    return buffer + (static_cast<size_t>(idx) * dim);
  }

  float *get_vector_ptr() { return buffer; }
};

class AsyncVectorMap {
private:
  int fd;
  struct io_uring ring;
  size_t D; // 向量维度
  size_t record_size;
  uint32_t pending_reqs = 0; // 当前在途（In-flight）的任务数
  std::vector<uint32_t> to_fetch_ids;
  uint32_t processed_idx;
  uint32_t batch_size_ = 4096;
  bool is_direct_io = false;

public:
  /**
   * @param path 文件路径
   * @param d 向量维度
   * @param entries 队列深度（推荐 256, 512 或更高，取决于磁盘并行度）
   */
  AsyncVectorMap(const char *path, size_t d, unsigned n_cand,
                 unsigned entries = 32768)
      : D(d) {

    record_size = D * sizeof(float);

    if (record_size % 512 == 0) {
      fd = open(path, O_RDONLY | O_DIRECT);
      if (fd >= 0)
        is_direct_io = true;
    }

    // 2. 如果 O_DIRECT 不可用或不支持，回退到普通 IO
    if (fd < 0) {
      fd = open(path, O_RDONLY);
      is_direct_io = false;
      if (fd < 0)
        throw std::runtime_error("Open failed: " +
                                 std::string(strerror(errno)));
    }

    std::cout << "[AsyncVectorMap] Mode: "
              << (is_direct_io ? "O_DIRECT" : "Buffered IO") << std::endl;

    // 3. 初始化 io_uring
    if (io_uring_queue_init(entries, &ring, 0) < 0) {
      close(fd);
      throw std::runtime_error("io_uring_queue_init failed");
    }

    to_fetch_ids.reserve(2 * n_cand);
    processed_idx = 0;
  }

  void reset() {
    to_fetch_ids.clear();
    processed_idx = 0;
  }

  uint32_t get_batch_size() { return batch_size_; }
  uint32_t get_pending_reqs() { return pending_reqs; }
  std::vector<uint32_t> &get_pending_ids() { return to_fetch_ids; }

  /**
   * 提交固定大小（batch_size_）的批次并等待完成
   * @param ctx 包含缓冲区的上下文
   */
  void submit_fetch(BucketContext *ctx) {
    // 1. 确定本批次的边界
    uint32_t total_size = to_fetch_ids.size();
    uint32_t submitted_this_run = 0;
    while (processed_idx < total_size && submitted_this_run < batch_size_) {
      struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
      if (!sqe) {
        // 如果 SQ 满了，立刻提交，然后收割一部分 CQE 释放空间
        io_uring_submit(&ring);
        return; // 或者在这里做个等待
      }
      long long offset =
          static_cast<long long>(to_fetch_ids[processed_idx]) * record_size;
      float *dest = ctx->get_vector_ptr(processed_idx);
      io_uring_prep_read(sqe, fd, dest, record_size, offset);
      io_uring_sqe_set_data(sqe, ctx);

      processed_idx++;
      submitted_this_run++;
      pending_reqs++;
    }

    // 3. 提交本批次的所有请求到内核
    // 批量提交
    if (submitted_this_run > 0) {
      int ret = io_uring_submit(&ring);
      if (ret < 0) {
        fprintf(stderr, "io_uring_submit failed: %s\n", strerror(-ret));
      }
    }
    // 4. 【关键】阻塞等待本批次的请求全部完成
    // 这样可以确保函数返回时，这 batch_size_ 条数据已经写到了内存中
    // wait_completions(current_batch_count);
  }

  void reap_completions() {
    struct io_uring_cqe *cqe;
    unsigned head;
    uint32_t completed_count = 0;

    io_uring_for_each_cqe(&ring, head, cqe) {
      completed_count++;
      // 关键：拿回的是 ctx 指针，而不是索引
      BucketContext *ctx = (BucketContext *)io_uring_cqe_get_data(cqe);

      if (cqe->res < 0) {
        fprintf(stderr, "Read error: %s\n", strerror(-cqe->res));
      } else {
        // 增加该 Context 的完成计数
        ctx->completed_cnt++;
      }
      pending_reqs--;
    }
    if (completed_count > 0)
      io_uring_cq_advance(&ring, completed_count);
    // std::cerr << "pending requests: " << pending_reqs << std::endl;
  }

  ~AsyncVectorMap() {
    io_uring_queue_exit(&ring);
    if (fd >= 0)
      close(fd);
  }
};

class AlignedSyncVectorMap {
private:
  int fd;
  io_context_t ctx;
  size_t D;
  size_t record_size;
  unsigned max_events; // 记录队列最大容量

public:
  // 修改：保存 max_ctx 到成员变量
  AlignedSyncVectorMap(const char *path, size_t d, unsigned max_ctx = 4096)
      : D(d), max_events(max_ctx) {

    // 如果不用 O_DIRECT，libaio 经常会退化成同步阻塞，但兼容性更好。
    // 如果追求极致性能且能处理对齐，建议加上 O_DIRECT。

    record_size = D * sizeof(float);
    bool is_direct_io = false;
    if (record_size % 512 == 0) {
      fd = open(path, O_RDONLY | O_DIRECT);
      if (fd >= 0)
        is_direct_io = true;
    }

    // 2. 如果 O_DIRECT 不可用或不支持，回退到普通 IO
    if (fd < 0) {
      fd = open(path, O_RDONLY);
      is_direct_io = false;
      if (fd < 0)
        throw std::runtime_error("Open failed: " +
                                 std::string(strerror(errno)));
    }

    std::cout << "[AsyncVectorMap] Mode: "
              << (is_direct_io ? "O_DIRECT" : "Buffered IO") << std::endl;

    ctx = 0;
    if (io_setup(max_ctx, &ctx) != 0) {
      perror("io_setup failed");
      exit(-1);
    }
  }

  void fetch_vectors(const std::vector<uint32_t> &ids, float *data_ptr) {
    size_t total_n = ids.size();
    if (total_n == 0)
      return;

    // 分批策略：每批次大小不能超过 max_events
    // 建议设置为 1024 或 2048，留一点缓冲空间
    const size_t BATCH_SIZE = std::min((size_t)max_events, (size_t)4096);

    std::vector<struct iocb> cbs(BATCH_SIZE);
    std::vector<struct iocb *> p_cbs(BATCH_SIZE);
    std::vector<struct io_event> events(BATCH_SIZE);

    size_t processed = 0;

    while (processed < total_n) {
      size_t current_batch_size = std::min(BATCH_SIZE, total_n - processed);

      // 2. 准备 IOCB
      for (size_t i = 0; i < current_batch_size; ++i) {
        size_t idx = processed + i;
        long long offset = (long long)ids[idx] * record_size;

        void *target_buffer = data_ptr + (1ull * idx * D);

        memset(&cbs[i], 0, sizeof(struct iocb));
        io_prep_pread(&cbs[i], fd, target_buffer, D * sizeof(float), offset);
        p_cbs[i] = &cbs[i];
      }

      int ret = io_submit(ctx, current_batch_size, p_cbs.data());

      if (ret < 0) {
        perror("io_submit failed"); // 可能是 -EAGAIN (队列满)
        break;
      }

      int submitted_count = ret; // 实际提交成功的数量
      int completed_so_far = 0;
      while (completed_so_far < submitted_count) {
        int wait_nr = submitted_count - completed_so_far;
        int r = io_getevents(ctx, wait_nr, wait_nr, events.data(), nullptr);

        if (r < 0) {
          perror("io_getevents failed");
          break;
        }
        completed_so_far += r;
      }
      processed += submitted_count;
    }
  }

  ~AlignedSyncVectorMap() {
    io_destroy(ctx);
    close(fd);
  }
};

class SyncVectorMap {
private:
  int fd;
  io_context_t ctx;
  size_t D;
  size_t record_size;
  unsigned max_events; // 记录队列最大容量

public:
  // 修改：保存 max_ctx 到成员变量
  SyncVectorMap(const char *path, size_t d, unsigned max_ctx = 4096)
      : D(d), max_events(max_ctx) {

    // 如果不用 O_DIRECT，libaio 经常会退化成同步阻塞，但兼容性更好。
    // 如果追求极致性能且能处理对齐，建议加上 O_DIRECT。
    fd = open(path, O_RDONLY);
    if (fd < 0) {
      perror("Failed to open base file");
      exit(-1);
    }

    record_size = sizeof(unsigned) + D * sizeof(float);
    ctx = 0;
    if (io_setup(max_ctx, &ctx) != 0) {
      perror("io_setup failed");
      exit(-1);
    }
  }

  void fetch_vectors(const std::vector<uint32_t> &ids, float *data_ptr) {
    size_t total_n = ids.size();
    if (total_n == 0)
      return;

    // 分批策略：每批次大小不能超过 max_events
    // 建议设置为 1024 或 2048，留一点缓冲空间
    const size_t BATCH_SIZE = std::min((size_t)max_events, (size_t)4096);

    std::vector<struct iocb> cbs(BATCH_SIZE);
    std::vector<struct iocb *> p_cbs(BATCH_SIZE);
    std::vector<struct io_event> events(BATCH_SIZE);

    size_t processed = 0;

    while (processed < total_n) {
      size_t current_batch_size = std::min(BATCH_SIZE, total_n - processed);

      // 2. 准备 IOCB
      for (size_t i = 0; i < current_batch_size; ++i) {
        size_t idx = processed + i;
        long long offset = (long long)ids[idx] * record_size + sizeof(unsigned);

        void *target_buffer = data_ptr + (1ull * idx * D);

        memset(&cbs[i], 0, sizeof(struct iocb));
        io_prep_pread(&cbs[i], fd, target_buffer, D * sizeof(float), offset);
        p_cbs[i] = &cbs[i];
      }

      int ret = io_submit(ctx, current_batch_size, p_cbs.data());

      if (ret < 0) {
        perror("io_submit failed"); // 可能是 -EAGAIN (队列满)
        break;
      }

      int submitted_count = ret; // 实际提交成功的数量
      int completed_so_far = 0;
      while (completed_so_far < submitted_count) {
        int wait_nr = submitted_count - completed_so_far;
        int r = io_getevents(ctx, wait_nr, wait_nr, events.data(), nullptr);

        if (r < 0) {
          perror("io_getevents failed");
          break;
        }
        completed_so_far += r;
      }
      processed += submitted_count;
    }
  }

  ~SyncVectorMap() {
    io_destroy(ctx);
    close(fd);
  }
};

constexpr size_t div_round_up(size_t val, size_t div) {
  return (val / div) + static_cast<size_t>((val % div) != 0);
}

constexpr size_t round_up_to_multiple(size_t val, size_t multiple_of) {
  return multiple_of * (div_round_up(val, multiple_of));
}

inline size_t ceil_log2(size_t val) {
  size_t res = 0;
  for (size_t i = 0; i < 31; ++i) {
    if ((1U << i) >= val) {
      res = i;
      break;
    }
  }
  return res;
}

namespace memory {
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

inline void mem_prefetch_l1(const char *ptr, size_t num_lines) {
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

inline void mem_prefetch_l2(const char *ptr, size_t num_lines) {
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
} // namespace memory

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

struct ProbeInfo {
  float min_val = 0;
  float max_val = 0;
  float scale = 0;
  float factor = 0;
  uint32_t cluster_id;
  uint8_t PORTABLE_ALIGN32 *LUT_uint8; // aligned for SIMD
  float PORTABLE_ALIGN32 *LUT_float;   // aligned for SIMD
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

  // --- 获取 const 指针 (用于只读场景) ---
  const float *val_data() const { return val.data(); }
  const uint32_t *idx_data() const { return idx.data(); }

  // 也可以直接提供简单的获取函数名
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

class TopKBufferSoA {
public:
  TopKBufferSoA(uint32_t k,                   // top-k
                uint32_t physical_bucket_num, // 物理桶数
                uint32_t logical_bucket_num)  // 桶数
      : k_(k), physical_bucket_num_(physical_bucket_num),
        logical_bucket_num_(logical_bucket_num) {
    bucketed_buffer_.resize(physical_bucket_num_);
    exact_buffer_.resize(physical_bucket_num_);
    code_lut_ = new PORTABLE_ALIGN64 uint8_t[logical_bucket_num_];
    for (uint32_t i = 0; i < physical_bucket_num_; ++i) {
      bucketed_buffer_[i].reserve(k_ * 5);
      exact_buffer_[i].reserve(k_ * 5);
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

  void reset() {
    for (auto &B : bucketed_buffer_)
      B.clear();
    for (auto &B : exact_buffer_)
      B.clear();
    logical_threshold_bucket_id_ = 1e7;
    physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
    predict_logical_bucket_id_ = 0;
  }

  void update_th_code(int visited_cluster_num, int nprobe) {
    uint32_t acc = 0;
    uint32_t j = 0;
    float sample_ratio = (1.0 * visited_cluster_num) / nprobe;
    uint32_t threshold = std::max(1u, static_cast<uint32_t>(sample_ratio * k_));
    bool update_flag = true;
    for (uint32_t i = 0; i < physical_bucket_num_; ++i) {
      acc += static_cast<uint32_t>(bucketed_buffer_[i].size()) +
             static_cast<uint32_t>(exact_buffer_[i].size());

      while (j < logical_bucket_num_ && code_lut_[j] == i) {
        ++j;
      }

      if (update_flag) {
        if (acc >= threshold) {
          predict_logical_bucket_id_ = j;
          update_flag = false;
        }
      }

      if (acc >= k_) {                         // 一旦达到 k_
        logical_threshold_bucket_id_ = j;      // 当前 pseudo id 即阈值
        physical_threshold_bucket_id_ = i + 1; // 当前物理桶 id
        predict_logical_bucket_id_ =
            std::min(predict_logical_bucket_id_, logical_threshold_bucket_id_);
        return;
      }
    }
    logical_threshold_bucket_id_ = 1e7;
    physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
    predict_logical_bucket_id_ =
        std::min(logical_threshold_bucket_id_, predict_logical_bucket_id_);
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
};

class AsyTopKBufferSoA {
public:
  AsyTopKBufferSoA(uint32_t k,                   // top-k
                   uint32_t physical_bucket_num, // 物理桶数
                   uint32_t logical_bucket_num)  // 桶数
      : k_(k), physical_bucket_num_(physical_bucket_num),
        logical_bucket_num_(logical_bucket_num) {
    bucketed_buffer_.resize(physical_bucket_num_);
    exact_buffer_.resize(physical_bucket_num_);
    code_lut_ = new PORTABLE_ALIGN64 uint8_t[logical_bucket_num_];
    processed_indices_.resize(physical_bucket_num_, 0);
    for (uint32_t i = 0; i < physical_bucket_num_; ++i) {
      bucketed_buffer_[i].reserve(k_ * 5);
      exact_buffer_[i].reserve(k_ * 5);
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

  //   void reset() {
  //     for (auto &B : bucketed_buffer_)
  //       B.clear();
  //     for (auto &B : exact_buffer_)
  //       B.clear();
  //     logical_threshold_bucket_id_ = 1e7;
  //     physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
  //     predict_logical_bucket_id_ = 0;
  //   }

  void reset() {
    for (auto &B : bucketed_buffer_)
      B.clear();
    for (auto &B : exact_buffer_)
      B.clear();
    // 重置所有进度索引
    std::fill(processed_indices_.begin(), processed_indices_.end(), 0);
    logical_threshold_bucket_id_ = 1e7;
    physical_threshold_bucket_id_ = physical_bucket_num_;
    predict_logical_bucket_id_ = 0;
  }

  void update_th_code(int visited_cluster_num, int nprobe,
                      AsyncVectorMap *as_v_map, BucketContext *bucket_context) {
    // std::vector<uint32_t> &to_fetch_ids = as_v_map->get_pending_ids();
    uint32_t acc = 0;
    uint32_t j = 0;
    float sample_ratio = (1.0 * visited_cluster_num) / nprobe;
    uint32_t threshold = std::max(1u, static_cast<uint32_t>(sample_ratio * k_));
    bool update_flag = true;
    // uint32_t batch_size = as_v_map->get_batch_size();
    // uint32_t current_processed = 0;

    for (uint32_t i = 0; i < physical_bucket_num_; ++i) {
      // while (processed_indices_[i] < bucketed_buffer_[i].size() &&
      //        current_processed < batch_size) {
      //   to_fetch_ids.push_back(bucketed_buffer_[i].idx_data()[processed_indices_[i]]);
      //   processed_indices_[i]++;
      //   current_processed++;
      // }

      acc += static_cast<uint32_t>(bucketed_buffer_[i].size()) +
             static_cast<uint32_t>(exact_buffer_[i].size());

      while (j < logical_bucket_num_ && code_lut_[j] == i) {
        ++j;
      }

      if (update_flag) {
        if (acc >= threshold) {
          predict_logical_bucket_id_ = j;
          update_flag = false;
        }
      }

      if (acc >= k_) {                         // 一旦达到 k_
        logical_threshold_bucket_id_ = j;      // 当前 pseudo id 即阈值
        physical_threshold_bucket_id_ = i + 1; // 当前物理桶 id
        predict_logical_bucket_id_ =
            std::min(predict_logical_bucket_id_, logical_threshold_bucket_id_);
        // if (current_processed > 0) {
        //   as_v_map->submit_fetch(bucket_context);
        //   // 顺便收割一下，防止 CQ 队列满导致的卡顿
        // }
        // as_v_map->reap_completions();
        return;
      }
    }
    logical_threshold_bucket_id_ = 1e7;
    physical_threshold_bucket_id_ = physical_bucket_num_; // 最右桶
    predict_logical_bucket_id_ =
        std::min(logical_threshold_bucket_id_, predict_logical_bucket_id_);
    // if (current_processed > 0) {
    //   as_v_map->submit_fetch(bucket_context);
    //   // 顺便收割一下，防止 CQ 队列满导致的卡顿
    // }
    // as_v_map->reap_completions();
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

private:
  uint32_t k_;
  uint32_t predict_logical_bucket_id_;
  uint32_t logical_bucket_num_;
  uint32_t logical_threshold_bucket_id_;
  uint32_t physical_bucket_num_;
  uint32_t physical_threshold_bucket_id_;
  uint32_t batch_size_ = 512; // 每次异步读取的数量
  float lower_{}, upper_{}, delta_{};
  PORTABLE_ALIGN64 uint8_t *code_lut_;
  std::vector<PairBucket> bucketed_buffer_;
  std::vector<PairBucket> exact_buffer_;
  std::vector<uint32_t> processed_indices_;
};
// class AIOTopKBufferSoA {
// public:
//   AIOTopKBufferSoA(uint32_t k, uint32_t phys_n, uint32_t log_n, int fd)
//       : k_(k), physical_bucket_num_(phys_n), logical_bucket_num_(log_n),
//         fd_(fd) {
//     // fd_ file document name
//     // 初始化 libaio
//     io_setup(1024, &aio_ctx_);

//     // 状态追踪
//     is_loading_.resize(physical_bucket_num_, false);
//     is_ready_.resize(physical_bucket_num_, false);

//     // 为每个物理桶分配对齐内存 (假设桶大小固定)
//     bucket_buffers_.resize(physical_bucket_num_);
//     for (uint32_t i = 0; i < physical_bucket_num_; ++i) {
//       bucket_buffers_[i] = (uint8_t *)aligned_alloc(MAX_BUCKET_BYTES);
//     }

//     code_lut_ = new uint8_t[logical_bucket_num_];
//     // ... 其他初始化 ...
//   }

//   // 关键逻辑：根据最新的阈值，读取所有需要的桶
//   void trigger_reads_by_threshold() {
//     // 获取当前逻辑阈值
//     uint32_t th_log = logical_threshold_bucket_id_;

//     // 遍历所有逻辑桶，直到阈值位置
//     for (uint32_t j = 0; j <= th_log && j < logical_bucket_num_; ++j) {
//       uint32_t b_phys = code_lut_[j];

//       // 如果该物理桶还没加载，且没在加载中，则发起 AIO 读
//       if (!is_ready_[b_phys] && !is_loading_[b_phys]) {
//         submit_aio_read(b_phys);
//       }
//     }
//   }

//   void submit_aio_read(uint32_t b_phys) {
//     struct iocb *cb = new iocb; // 生产环境建议用池化
//     // 假设每个桶在文件中的偏移是固定的
//     long long offset = (long long)b_phys * MAX_BUCKET_BYTES;

//     io_prep_pread(cb, fd_, bucket_buffers_[b_phys], MAX_BUCKET_BYTES,
//     offset); cb->data = (void *)(uintptr_t)b_phys; // 回传物理 ID

//     if (io_submit(aio_ctx_, 1, &cb) == 1) {
//       is_loading_[b_phys] = true;
//     } else {
//       delete cb;
//     }
//   }

//   // 非阻塞轮询 IO 完成情况
//   void poll_io() {
//     struct io_event events[32];
//     struct timespec timeout = {0, 0}; // 立即返回
//     int num = io_getevents(aio_ctx_, 0, 32, events, &timeout);

//     for (int i = 0; i < num; ++i) {
//       uint32_t b_phys = (uint32_t)(uintptr_t)events[i].data;
//       is_loading_[b_phys] = false;
//       is_ready_[b_phys] = true;
//       delete (struct iocb *)events[i].obj;
//     }
//   }

//   // 修改原有的 update_th_code，在末尾加入触发逻辑
//   void update_th_code(int visited_cluster_num, int nprobe) {
//     // ... 原有的逻辑，计算出新的 logical_threshold_bucket_id_ ...

//     // 重点：计算完阈值后，立即触发 IO
//     trigger_reads_by_threshold();
//     // 顺便检查一下之前发起的读有没有完工的
//     poll_io();
//   }

//   // 检查某个逻辑桶的数据是否已经 Ready，可以进行精确计算了
//   bool is_bucket_accessible(uint32_t b_logical) {
//     uint32_t b_phys = code_lut_[b_logical];
//     return is_ready_[b_phys];
//   }

// private:
//   io_context_t aio_ctx_;
//   int fd_;
//   const size_t MAX_BUCKET_BYTES = 65536; // 64KB 对齐读取

//   std::vector<PairBucket> bucketed_buffer_;
//   std::vector<bool> is_loading_;
//   std::vector<bool> is_ready_;

//   uint32_t k_, logical_bucket_num_, physical_bucket_num_;
//   uint32_t logical_threshold_bucket_id_;
//   uint8_t *code_lut_;
// };

inline void accumulate_one_block(uint8_t *codes, uint8_t *LUT, uint16_t *result,
                                 unsigned dim) {
  __m256i low_mask = _mm256_set1_epi8(0xf);
  __m256i accu[4];
  for (int i = 0; i < 4; i++) {
    accu[i] = _mm256_setzero_si256();
  }

  uint32_t M = dim / 4;

  for (int m = 0; m < M; m += 2) {
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
  __m256i dis0 =
      _mm256_add_epi16(_mm256_permute2f128_si256(accu[0], accu[1], 0x21),
                       _mm256_blend_epi32(accu[0], accu[1], 0xF0));
  _mm256_store_si256((__m256i *)(result + 0), dis0);

  accu[2] = _mm256_sub_epi16(accu[2], _mm256_slli_epi16(accu[3], 8));
  __m256i dis1 =
      _mm256_add_epi16(_mm256_permute2f128_si256(accu[2], accu[3], 0x21),
                       _mm256_blend_epi32(accu[2], accu[3], 0xF0));
  _mm256_store_si256((__m256i *)(result + 16), dis1);
}

inline void accumulate(uint32_t nblk, uint8_t *codes, uint8_t *LUT,
                       uint16_t *result, unsigned code_size) {
  for (int i = 0; i < nblk; i++) {
    accumulate_one_block(codes, LUT, result, code_size * 8);
    codes += 32 * code_size;
    result += 32;
  }
}

inline void get_matrix_column(const uint8_t *src, size_t m, size_t n, int64_t i,
                              int64_t j, std::array<uint8_t, 32> &dest) {
  for (int64_t k = 0; k < dest.size(); k++) {
    if (k + i >= 0 && k + i < m) {
      dest[k] = src[(k + i) * n + j];
    } else {
      dest[k] = 0;
    }
  }
}

void pack_codes(const uint8_t *codes, uint32_t ncode, uint8_t *blocks,
                uint32_t B) {

  uint32_t ncode_pad = (ncode + 31) / 32 * 32;
  uint32_t M = B / 4;
  const uint8_t bbs = 32;
  memset(blocks, 0, ncode_pad * M / 2);

  const uint8_t perm0[16] = {0, 8,  1, 9,  2, 10, 3, 11,
                             4, 12, 5, 13, 6, 14, 7, 15};
  uint8_t *codes2 = blocks;
  for (int blk = 0; blk < ncode_pad; blk += bbs) {
    // enumerate m
    for (int m = 0; m < M; m += 2) {
      std::array<uint8_t, 32> c, c0, c1;
      get_matrix_column(codes, ncode, M / 2, blk, m / 2, c);
      for (int j = 0; j < 32; j++) {
        c0[j] = c[j] & 15;
        c1[j] = c[j] >> 4;
      }
      for (int j = 0; j < 16; j++) {
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

inline float compute_l2_distance(const float *d, const float *q, uint32_t L) {
  float PORTABLE_ALIGN32 TmpRes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint32_t num_blk16 = L >> 4;
  uint32_t l = L & 0b1111;

  __m256 diff, v1, v2;
  __m256 sum = _mm256_set1_ps(0);
  for (int i = 0; i < num_blk16; i++) {
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
  for (int i = 0; i < l / 8; i++) {
    v1 = _mm256_loadu_ps(d);
    v2 = _mm256_loadu_ps(q);
    d += 8;
    q += 8;
    diff = _mm256_sub_ps(v1, v2);
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
  }
  _mm256_store_ps(TmpRes, sum);

  float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] +
              TmpRes[5] + TmpRes[6] + TmpRes[7];

  for (int i = 0; i < l % 8; i++) {
    float tmp = (*q) - (*d);
    ret += tmp * tmp;
    d++;
    q++;
  }
  return ret;
}

inline float compute_sub(const float *d, const float *q, float *sub,
                         uint32_t L) {
  float PORTABLE_ALIGN32 TmpRes[8] = {0};
  uint32_t num_blk16 = L >> 4;
  uint32_t l = L & 0xF;

  __m256 diff, v1, v2;
  __m256 sum = _mm256_setzero_ps();

  // 处理每 16 个元素（分两次 8 个一组）
  for (uint32_t i = 0; i < num_blk16; i++) {
    // 第一组 8 个
    v1 = _mm256_loadu_ps(d);
    v2 = _mm256_loadu_ps(q);
    diff = _mm256_sub_ps(v1, v2);
    _mm256_storeu_ps(sub, diff); // 保存差值
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    d += 8;
    q += 8;
    sub += 8;

    // 第二组 8 个
    v1 = _mm256_loadu_ps(d);
    v2 = _mm256_loadu_ps(q);
    diff = _mm256_sub_ps(v1, v2);
    _mm256_storeu_ps(sub, diff); // 保存差值
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    d += 8;
    q += 8;
    sub += 8;
  }

  // 处理剩余整 8 个的部分
  for (uint32_t i = 0; i < l / 8; i++) {
    v1 = _mm256_loadu_ps(d);
    v2 = _mm256_loadu_ps(q);
    diff = _mm256_sub_ps(v1, v2);
    _mm256_storeu_ps(sub, diff); // 保存差值
    sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    d += 8;
    q += 8;
    sub += 8;
  }

  // 水平累加 SIMD 和
  _mm256_store_ps(TmpRes, sum);
  float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] +
              TmpRes[5] + TmpRes[6] + TmpRes[7];

  // 标量尾部处理
  for (uint32_t i = 0; i < (l % 8); i++) {
    float t = *d - *q;
    *sub = t; // 保存差值
    ret += t * t;
    d++;
    q++;
    sub++;
  }

  return ret;
}

inline float hsum4(__m128 v) {
  v = _mm_hadd_ps(v, v); // 4 -> 2
  v = _mm_hadd_ps(v, v); // 2 -> 1
  return _mm_cvtss_f32(v);
}

class Parameters {
public:
  template <typename T> inline void set(const std::string &name, const T &val) {
    std::stringstream ss;
    ss << val;
    params[name] = ss.str();
  }

  template <typename T> inline T get(const std::string &name) const {
    auto item = params.find(name);
    if (item == params.end()) {
      throw std::invalid_argument("Invalid paramter name : " + name + ".");
    } else {
      return ConvertStrToValue<T>(item->second);
    }
  }

  inline std::string toString() const {
    std::string res;
    for (auto &param : params) {
      res += param.first;
      res += ":";
      res += param.second;
      res += " ";
    }
    return res;
  }

private:
  std::unordered_map<std::string, std::string> params;

  template <typename T>
  inline T ConvertStrToValue(const std::string &str) const {
    std::stringstream sstream(str);
    T value;
    if (!(sstream >> value) || !sstream.eof()) {
      std::stringstream err;
      err << "Fail to convert value" << str
          << " to type: " << typeid(value).name();
      throw std::runtime_error(err.str());
    }
    return value;
  }
};

template <typename T>
inline void load_data(const char *filename, T *&data, unsigned &num,
                      unsigned &dim) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Error opening file " << filename << std::endl;
    exit(-1);
  }

  // 读取维度信息
  in.read((char *)&dim, 4);
  if (in.fail()) {
    std::cerr << "Error reading dimension from file " << filename << std::endl;
    exit(-1);
  }

  // 获取文件大小
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  auto f_size = (size_t)ss;

  // 计算数据数量
  num = (unsigned)(f_size / (dim + 1) / 4);

  size_t total_size = (size_t)num * dim;
  // 分配内存
  try {
    data = new T[total_size];
  } catch (std::bad_alloc &) {
    std::cerr << "Memory allocation failed for data in " << filename
              << std::endl;
    exit(-1);
  }

  in.seekg(0, std::ios::beg);
  // 分块读取数据
  const size_t block_size =
      10000 * dim; // 每次读取10000个数据块，可以根据需要调整
  size_t offset = 0;

  while (offset < total_size) {
    size_t remaining = total_size - offset;
    size_t current_block_size = std::min(block_size, remaining);

    for (size_t i = 0; i < current_block_size / dim; ++i) {
      // 读取并验证维度信息
      unsigned current_dim;
      in.read(reinterpret_cast<char *>(&current_dim), sizeof(current_dim));
      if (in.fail() || current_dim != dim) {
        std::cerr << "Error reading dimension or dimension mismatch in file "
                  << filename << " at index " << (offset / dim + i)
                  << std::endl;
        delete[] data;
        exit(-1);
      }

      in.read(reinterpret_cast<char *>(data + offset + i * dim),
              dim * sizeof(T));
      if (in.fail()) {
        std::cerr << "Error reading data from file " << filename << " at index "
                  << (offset / dim + i) << std::endl;
        delete[] data;
        exit(-1);
      }
    }

    offset += current_block_size;
  }

  in.close();

  // 输出调试信息
  std::cout << "Loaded " << num << " entries from " << filename
            << " with dimension " << dim << std::endl;
}

void load_bvecs(const char *filename, uint8_t *&data, unsigned &num,
                unsigned &dim) {
  std::ifstream in(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "Error opening file " << filename << std::endl;
    exit(-1);
  }

  // 读取维度信息
  in.read((char *)&dim, 4);
  if (in.fail()) {
    std::cerr << "Error reading dimension from file " << filename << std::endl;
    exit(-1);
  }

  // 获取文件大小
  in.seekg(0, std::ios::end);
  std::ios::pos_type ss = in.tellg();
  auto f_size = (size_t)ss;

  // 计算数据数量
  num = (unsigned)(f_size / (dim + 4));

  // std::cerr << "Number of quantization code is: " << num << " dim is: " <<
  // dim << std::endl;

  size_t total_size = (size_t)num * dim;
  // 分配内存
  try {
    data = new uint8_t[total_size];
  } catch (std::bad_alloc &) {
    std::cerr << "Memory allocation failed for data in " << filename
              << std::endl;
    exit(-1);
  }

  in.seekg(0, std::ios::beg);
  // 分块读取数据
  const size_t block_size =
      10000 * dim; // 每次读取10000个数据块，可以根据需要调整
  size_t offset = 0;

  while (offset < total_size) {
    size_t remaining = total_size - offset;
    size_t current_block_size = std::min(block_size, remaining);

    for (size_t i = 0; i < current_block_size / dim; ++i) {
      // 读取并验证维度信息
      unsigned current_dim;
      in.read(reinterpret_cast<char *>(&current_dim), sizeof(current_dim));
      // std::cerr << "Reading dim: " << current_dim << std::endl;
      if (in.fail() || current_dim != dim) {
        std::cerr << "Error reading dimension or dimension mismatch in file "
                  << filename << " at index " << (offset / dim + i)
                  << std::endl;
        delete[] data;
        exit(-1);
      }

      in.read(reinterpret_cast<char *>(data + offset + i * dim),
              dim * sizeof(uint8_t));
      if (in.fail()) {
        std::cerr << "Error reading data from file " << filename << " at index "
                  << (offset / dim + i) << std::endl;
        delete[] data;
        exit(-1);
      }
    }

    offset += current_block_size;
  }

  in.close();

  // 输出调试信息
  std::cout << "Loaded " << num << " entries from " << filename
            << " with dimension " << dim << std::endl;
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

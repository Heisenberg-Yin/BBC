#pragma once
#include <chrono>
#include <queue>
#include <unordered_set>
#include <limits>
#include <memory.h>

#ifndef WIN32
#include <sys/resource.h>
#endif

typedef std::pair<float, uint32_t> Result;
typedef std::priority_queue<Result> ResultHeap;

#pragma once
#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>
#include <cstdint>
#include <x86intrin.h>
#include <sched.h>

struct TSCTimer
{
    // 序列化 + 读 TSC
    static inline uint64_t start()
    {
        uint32_t eax, ebx, ecx, edx;
        // CPUID 序列化: EAX=0
        asm volatile(
            "xor %%eax, %%eax\n\t"
            "cpuid\n\t"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : /* no inputs */
            : "memory");
        return __rdtsc();
    }

    // 读 TSC + 序列化
    static inline uint64_t stop()
    {
        uint32_t aux;
        uint64_t t = __rdtscp(&aux); // RDTSCP: serializing read
        // 再做一次 CPUID 保证后续指令不乱序
        uint32_t eax, ebx, ecx, edx;
        asm volatile(
            "xor %%eax, %%eax\n\t"
            "cpuid\n\t"
            : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
            : /* no inputs */
            : "memory");
        return t;
    }
};

struct Candidate
{
    uint32_t id;
    float dist_lb;
    float dist_ub;
    bool exact;
    Candidate() = default;

    Candidate(uint32_t _id, float _dist_lb, float _dist_ub, bool _exact = false)
        : id(_id), dist_lb(_dist_lb), dist_ub(_dist_ub), exact(_exact)
    {
    }
};
// struct CmpCandMin
// {
//     constexpr bool operator()(const Candidate &a, const Candidate &b) const noexcept
//     {
//         return a.dist > b.dist; // 小根堆：dist_lb 小的先出
//     }
// };

// struct CmpCandMax
// {
//     constexpr bool operator()(const Candidate &a, const Candidate &b) const noexcept
//     {
//         return a.dist < b.dist;
//     }
// };

struct CmpCandlbMin
{
    constexpr bool operator()(const Candidate &a, const Candidate &b) const noexcept
    {
        return a.dist_lb > b.dist_lb; // 小根堆：dist_lb 小的先出
    }
};

struct CmpCandlbMax
{
    constexpr bool operator()(const Candidate &a, const Candidate &b) const noexcept
    {
        return a.dist_lb < b.dist_lb;
    }
};

struct CmpCandubMin
{
    constexpr bool operator()(const Candidate &a, const Candidate &b) const noexcept
    {
        return a.dist_ub > b.dist_ub; // 小根堆：dist_ub 小的先出
    }
};

struct CmpCandubMax
{
    constexpr bool operator()(const Candidate &a, const Candidate &b) const noexcept
    {
        return a.dist_ub < b.dist_ub; // 小根堆：dist_ub 小的先出
    }
};

static inline void prefetch_l1(const void *addr)
{
#if defined(__SSE2__)
    _mm_prefetch(addr, _MM_HINT_T0);
#else
    __builtin_prefetch(addr, 0, 3);
#endif
}

static inline void prefetch_l2(const void *addr)
{
#if defined(__SSE2__)
    _mm_prefetch((const char *)addr, _MM_HINT_T1);
#else
    __builtin_prefetch(addr, 0, 2);
#endif
}

inline void mem_prefetch_l1(const char *ptr, uint32_t num_lines)
{
    switch (num_lines)
    {
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

inline void mem_prefetch_l2(const char *ptr, uint32_t num_lines)
{
    switch (num_lines)
    {
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

// struct ResultPool
// {
// public:
//     ResultPool(uint32_t capacity)
//         : ids_(capacity + 1), distances_(capacity + 1), capacity_(capacity) {}

//     void insert(uint32_t u, float dist)
//     {
//         if (size_ == capacity_ && dist > distances_[size_ - 1])
//         {
//             return;
//         }
//         uint32_t lo = find_bsearch(dist);
//         std::memmove(&ids_[lo + 1], &ids_[lo], (size_ - lo) * sizeof(uint32_t));
//         ids_[lo] = u;
//         std::memmove(&distances_[lo + 1], &distances_[lo], (size_ - lo) * sizeof(float));
//         distances_[lo] = dist;
//         size_ += (size_ < capacity_);
//         return;
//     }

//     uint32_t find_bsearch(float dist) const
//     {
//         uint32_t lo = 0, len = size_;
//         uint32_t half;
//         while (len > 1)
//         {
//             half = len >> 1;
//             len -= half;
//             lo += (distances_[lo + half - 1] < dist) * half;
//         }
//         return (lo < size_ && distances_[lo] < dist) ? lo + 1 : lo;
//     }

//     float distk()
//     {
//         return size_ == capacity_ ? distances_[size_ - 1]
//                                   : std::numeric_limits<float>::max();
//     }

//     void copy_results(uint32_t *KNN) { std::copy(ids_.begin(), ids_.end() - 1, KNN); }

//     //    private:
//     std::vector<uint32_t, memory::align_allocator<uint32_t>> ids_;
//     std::vector<float, memory::align_allocator<float>> distances_;
//     uint32_t size_ = 0, capacity_;
// };

namespace Detail
{
    double constexpr sqrtNewtonRaphson(double x, double curr, double prev)
    {
        return curr == prev
                   ? curr
                   : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
    }
}

/*
 * Constexpr version of the square root
 * Return value:
 *	- For a finite and non-negative value of "x", returns an approximation for the square root of "x"
 *   - Otherwise, returns NaN
 */
double constexpr const_sqrt(double x)
{
    return x >= 0 && x < std::numeric_limits<double>::infinity()
               ? Detail::sqrtNewtonRaphson(x, x, 0)
               : std::numeric_limits<double>::quiet_NaN();
}

void print_binary(uint64_t v)
{
    for (int i = 0; i < 64; i++)
    {
        std::cerr << ((v >> (63 - i)) & 1);
    }
}

void print_binary(uint8_t v)
{
    for (int i = 0; i < 8; i++)
    {
        std::cerr << ((v >> (7 - i)) & 1);
    }
}

inline uint32_t reverseBits(uint32_t n)
{
    n = (n >> 1) & 0x55555555 | (n << 1) & 0xaaaaaaaa;
    n = (n >> 2) & 0x33333333 | (n << 2) & 0xcccccccc;
    n = (n >> 4) & 0x0f0f0f0f | (n << 4) & 0xf0f0f0f0;
    n = (n >> 8) & 0x00ff00ff | (n << 8) & 0xff00ff00;
    n = (n >> 16) & 0x0000ffff | (n << 16) & 0xffff0000;
    return n;
}

ResultHeap getGroundtruth(const Matrix<float> &X, const Matrix<float> &Q, size_t query,
                          unsigned *groundtruth, size_t k)
{
    ResultHeap ret;
    for (int i = 0; i < k; i++)
    {
        unsigned gt = groundtruth[i];
        ret.push(std::make_pair(Q.dist(query, X, gt), gt));
    }
    return ret;
}

float getRatio(int q, const Matrix<float> &Q, const Matrix<float> &X, const Matrix<uint32_t> &G, ResultHeap KNNs)
{
    ResultHeap gt;
    int k = KNNs.size();
    for (int i = 0; i < k; i++)
    {
        gt.emplace(Q.dist(q, X, G.data[q * G.d + i]), G.data[q * G.d + i]);
    }
    long double ret = 0;
    int valid_k = 0;
    while (gt.size())
    {
        if (gt.top().first > 1e-5)
        {
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

int getRecall(ResultHeap &result, ResultHeap &gt)
{
    int correct = 0;

    std::unordered_set<unsigned> g;
    int ret = 0;

    while (gt.size())
    {
        g.insert(gt.top().second);
        // std::cerr << "ID - " << gt.top().second << " dist - " << gt.top().first << std::endl;
        gt.pop();
    }

    while (result.size())
    {
        // std::cerr << "ID - " << result.top().second << " dist - " << result.top().first << std::endl;
        if (g.find(result.top().second) != g.end())
        {
            ret++;
        }
        result.pop();
    }

    return ret;
}



#ifndef WIN32
void
GetCurTime(rusage *curTime)
{
    int ret = getrusage(RUSAGE_SELF, curTime);
    if (ret != 0)
    {
        fprintf(stderr, "The running time info couldn't be collected successfully.\n");
        // FreeData( 2);
        exit(0);
    }
}

static void pin2core(int core_id = 0)
{
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(core_id, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
}
/*
 * GetTime is used to get the 'float' format time from the start and end rusage structure.
 *
 * @Param timeStart, timeEnd indicate the two time points.
 * @Param userTime, sysTime get back the time information.
 *
 * @Return void.
 */
void GetTime(struct rusage *timeStart, struct rusage *timeEnd, float *userTime, float *sysTime)
{
    (*userTime) = ((float)(timeEnd->ru_utime.tv_sec - timeStart->ru_utime.tv_sec)) +
                  ((float)(timeEnd->ru_utime.tv_usec - timeStart->ru_utime.tv_usec)) * 1e-6;
    (*sysTime) = ((float)(timeEnd->ru_stime.tv_sec - timeStart->ru_stime.tv_sec)) +
                 ((float)(timeEnd->ru_stime.tv_usec - timeStart->ru_stime.tv_usec)) * 1e-6;
}

#endif

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
size_t getPeakRSS()
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L; /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo))
    {
        close(fd);
        return (size_t)0L; /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
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
size_t getCurrentRSS()
{
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L; /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t)0L; /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1)
    {
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
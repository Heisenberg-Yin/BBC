#pragma once
#include <x86intrin.h>

#include <sstream>
#include <unordered_map>
#include <chrono>
#include <limits>
#ifndef WIN32
#include <sys/resource.h>
#endif
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))

static inline void prefetch_l1(const void *addr)
{
#if defined(__SSE2__)
    _mm_prefetch(addr, _MM_HINT_T0);
#else
    __builtin_prefetch(addr, 0, 3);
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


inline void accumulate_one_block(uint8_t* codes, uint8_t* LUT, uint16_t* result, unsigned dim){
    __m256i low_mask = _mm256_set1_epi8(0xf);
    __m256i accu[4];
    for(int i=0;i<4;i++){
        accu[i] = _mm256_setzero_si256();   
    }

    uint32_t M = dim / 4;

    for(int m=0;m<M;m+=2){
        __m256i c   = _mm256_load_si256((__m256i const*)codes);
        __m256i lo  = _mm256_and_si256(c, low_mask);
        __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(c, 4), low_mask);

        __m256i lut = _mm256_load_si256((__m256i const*)LUT);

        __m256i res_lo = _mm256_shuffle_epi8(lut, lo);
        __m256i res_hi = _mm256_shuffle_epi8(lut, hi);

        accu[0] = _mm256_add_epi16(accu[0], res_lo);
        accu[1] = _mm256_add_epi16(accu[1], _mm256_srli_epi16(res_lo, 8));

        accu[2] = _mm256_add_epi16(accu[2], res_hi);
        accu[3] = _mm256_add_epi16(accu[3], _mm256_srli_epi16(res_hi, 8));
        
        codes += 32;
        LUT   += 32;
    }

    accu[0] = _mm256_sub_epi16(accu[0], _mm256_slli_epi16(accu[1], 8));
    __m256i dis0 = _mm256_add_epi16(_mm256_permute2f128_si256(accu[0], accu[1], 0x21),_mm256_blend_epi32(accu[0], accu[1], 0xF0));
    _mm256_store_si256((__m256i*)(result + 0 ), dis0);
    
    accu[2] = _mm256_sub_epi16(accu[2], _mm256_slli_epi16(accu[3], 8));
    __m256i dis1 = _mm256_add_epi16(_mm256_permute2f128_si256(accu[2], accu[3], 0x21),_mm256_blend_epi32(accu[2], accu[3], 0xF0));
    _mm256_store_si256((__m256i*)(result + 16), dis1);    
}

inline void accumulate(uint32_t nblk, uint8_t* codes, uint8_t* LUT, uint16_t* result, unsigned code_size){
    for(int i=0;i<nblk;i++){
        accumulate_one_block(codes, LUT, result, code_size*8);
        codes  += 32 * code_size;
        result += 32;
    }
}


inline void get_matrix_column(const uint8_t *src, size_t m, size_t n, int64_t i, int64_t j, std::array<uint8_t, 32>& dest) {
    for (int64_t k = 0; k < dest.size(); k++) {
        if (k + i >= 0 && k + i < m) {
            dest[k] = src[(k + i) * n + j];
        } 
        else {
            dest[k] = 0;
        }
    }
}

void pack_codes(const uint8_t* codes, uint32_t ncode, uint8_t* blocks, uint32_t B){
    
    uint32_t ncode_pad = (ncode + 31) / 32 * 32;
    uint32_t M = B / 4;
    const uint8_t bbs = 32;    
    memset(blocks, 0, ncode_pad * M / 2);

    const uint8_t perm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
    uint8_t* codes2 = blocks;
    for(int blk=0;blk<ncode_pad;blk+=bbs){
        // enumerate m
        for(int m=0;m<M;m+=2){
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



inline float compute_l2_distance(const float *d, const float *q, uint32_t L)
{
    float PORTABLE_ALIGN32 TmpRes[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t num_blk16 = L >> 4;
    uint32_t l = L & 0b1111;

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

inline float compute_sub(const float *d, const float *q, float *sub, uint32_t L)
{
    float PORTABLE_ALIGN32 TmpRes[8] = {0};
    uint32_t num_blk16 = L >> 4;
    uint32_t l = L & 0xF;

    __m256 diff, v1, v2;
    __m256 sum = _mm256_setzero_ps();

    // 处理每 16 个元素（分两次 8 个一组）
    for (uint32_t i = 0; i < num_blk16; i++)
    {
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
    for (uint32_t i = 0; i < l / 8; i++)
    {
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
    float ret = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    // 标量尾部处理
    for (uint32_t i = 0; i < (l % 8); i++)
    {
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
    v = _mm_hadd_ps(v, v);   // 4 -> 2
    v = _mm_hadd_ps(v, v);   // 2 -> 1
    return _mm_cvtss_f32(v);
}

class Parameters
{
public:
    template <typename T>
    inline void set(const std::string &name, const T &val)
    {
        std::stringstream ss;
        ss << val;
        params[name] = ss.str();
    }

    template <typename T>
    inline T get(const std::string &name) const
    {
        auto item = params.find(name);
        if (item == params.end())
        {
            throw std::invalid_argument("Invalid paramter name : " + name + ".");
        }
        else
        {
            return ConvertStrToValue<T>(item->second);
        }
    }

    inline std::string toString() const
    {
        std::string res;
        for (auto &param : params)
        {
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
    inline T ConvertStrToValue(const std::string &str) const
    {
        std::stringstream sstream(str);
        T value;
        if (!(sstream >> value) || !sstream.eof())
        {
            std::stringstream err;
            err << "Fail to convert value" << str << " to type: " << typeid(value).name();
            throw std::runtime_error(err.str());
        }
        return value;
    }
};

template <typename T>
inline void load_data(const char *filename, T *&data, unsigned &num, unsigned &dim)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(-1);
    }

    // 读取维度信息
    in.read((char *)&dim, 4);
    if (in.fail())
    {
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
    try
    {
        data = new T[total_size];
    }
    catch (std::bad_alloc &)
    {
        std::cerr << "Memory allocation failed for data in " << filename << std::endl;
        exit(-1);
    }

    in.seekg(0, std::ios::beg);
    // 分块读取数据
    const size_t block_size = 10000 * dim; // 每次读取10000个数据块，可以根据需要调整
    size_t offset = 0;

    while (offset < total_size)
    {
        size_t remaining = total_size - offset;
        size_t current_block_size = std::min(block_size, remaining);

        for (size_t i = 0; i < current_block_size / dim; ++i)
        {
            // 读取并验证维度信息
            unsigned current_dim;
            in.read(reinterpret_cast<char *>(&current_dim), sizeof(current_dim));
            if (in.fail() || current_dim != dim)
            {
                std::cerr << "Error reading dimension or dimension mismatch in file " << filename << " at index " << (offset / dim + i) << std::endl;
                delete[] data;
                exit(-1);
            }

            in.read(reinterpret_cast<char *>(data + offset + i * dim), dim * sizeof(T));
            if (in.fail())
            {
                std::cerr << "Error reading data from file " << filename << " at index " << (offset / dim + i) << std::endl;
                delete[] data;
                exit(-1);
            }
        }

        offset += current_block_size;
    }

    in.close();

    // 输出调试信息
    std::cout << "Loaded " << num << " entries from " << filename << " with dimension " << dim << std::endl;
}

void load_bvecs(const char *filename,
                uint8_t *&data,
                unsigned &num,
                unsigned &dim)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Error opening file " << filename << std::endl;
        exit(-1);
    }

    // 读取维度信息
    in.read((char *)&dim, 4);
    if (in.fail())
    {
        std::cerr << "Error reading dimension from file " << filename << std::endl;
        exit(-1);
    }

    // 获取文件大小
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    auto f_size = (size_t)ss;

    // 计算数据数量
    num = (unsigned)(f_size / (dim + 4));

    // std::cerr << "Number of quantization code is: " << num << " dim is: " << dim << std::endl;

    size_t total_size = (size_t)num * dim;
    // 分配内存
    try
    {
        data = new uint8_t[total_size];
    }
    catch (std::bad_alloc &)
    {
        std::cerr << "Memory allocation failed for data in " << filename << std::endl;
        exit(-1);
    }

    in.seekg(0, std::ios::beg);
    // 分块读取数据
    const size_t block_size = 10000 * dim; // 每次读取10000个数据块，可以根据需要调整
    size_t offset = 0;

    while (offset < total_size)
    {
        size_t remaining = total_size - offset;
        size_t current_block_size = std::min(block_size, remaining);

        for (size_t i = 0; i < current_block_size / dim; ++i)
        {
            // 读取并验证维度信息
            unsigned current_dim;
            in.read(reinterpret_cast<char *>(&current_dim), sizeof(current_dim));
            // std::cerr << "Reading dim: " << current_dim << std::endl;
            if (in.fail() || current_dim != dim)
            {
                std::cerr << "Error reading dimension or dimension mismatch in file " << filename << " at index " << (offset / dim + i) << std::endl;
                delete[] data;
                exit(-1);
            }

            in.read(reinterpret_cast<char *>(data + offset + i * dim), dim * sizeof(uint8_t));
            if (in.fail())
            {
                std::cerr << "Error reading data from file " << filename << " at index " << (offset / dim + i) << std::endl;
                delete[] data;
                exit(-1);
            }
        }

        offset += current_block_size;
    }

    in.close();

    // 输出调试信息
    std::cout << "Loaded " << num << " entries from " << filename << " with dimension " << dim << std::endl;
}

#ifndef WIN32
void GetCurTime(rusage *curTime)
{
    int ret = getrusage(RUSAGE_SELF, curTime);
    if (ret != 0)
    {
        fprintf(stderr, "The running time info couldn't be collected successfully.\n");
        // FreeData( 2);
        exit(0);
    }
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

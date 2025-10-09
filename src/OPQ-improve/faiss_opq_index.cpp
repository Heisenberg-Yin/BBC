#include <faiss/IndexFlat.h>
#include <matrix.hpp>
#include <utils.hpp>
#include <fastscan.hpp>
#include <faiss/IndexPQFastScan.h>
#include <faiss/index_io.h>
#include <iostream>
#include <getopt.h>
#include <fstream>
#include <queue>
#include <random>

int main(int argc, char *argv[])
{

    const struct option longopts[] = {
        {"help", no_argument, 0, 'h'},
        {"dataset", required_argument, 0, 'd'},
        {"source", required_argument, 0, 's'},

    };

    int ind;
    int iarg = 0;
    opterr = 1; // getopt error message (off: 0)

    char dataset[256] = "";
    char source[256] = "";

    while (iarg != -1)
    {
        iarg = getopt_long(argc, argv, "d:s:", longopts, &ind);
        switch (iarg)
        {
        case 'd':
            if (optarg)
            {
                strcpy(dataset, optarg);
            }
            break;
        case 's':
            if (optarg)
            {
                strcpy(source, optarg);
            }
            break;
        }
    }

    char data_path[256], centroid_path[256], rotated_path[256];
    sprintf(data_path, "%sbase.fvecs", source);
    sprintf(centroid_path, "%scentroid_4096.fvecs", source);
    sprintf(rotated_path, "./data/rotation_%s.fvecs", dataset);
    unsigned n, d, n_centroids, d_centroids, d_rot;
    Matrix<float> centroid(centroid_path);
    Matrix<float> base_data(data_path);
    Matrix<float> opq(rotated_path);
    n = base_data.n;
    d = base_data.d;
    n_centroids = centroid.n;
    d_centroids = centroid.d;
    d_rot = opq.d;

    int numC = n_centroids; // 粗聚类中心数目
    int M = d / 4;
    int b = 4;
    int sample_size = std::min((unsigned)1000000, n);
    std::cerr << "Data dim: " << d << ", base size: " << n << ", coarse centers: " << numC << std::endl;
    std::vector<faiss::idx_t> sample_assign(sample_size);
    std::vector<float> sample_dist(sample_size);

    faiss::IndexFlatL2 quantizer(d);
    quantizer.add(numC, centroid.data);
    quantizer.search(sample_size, base_data.data, 1, sample_dist.data(), sample_assign.data());

    Matrix<float> sample_xb(sample_size, d);
    for (unsigned i = 0; i < sample_size; ++i)
    {
        compute_sub(base_data.data + 1ull * i * d, centroid.data + sample_assign[i] * d, sample_xb.data + 1ull * i * d, d);
    }
    sample_xb = mul(sample_xb, opq);

    faiss::IndexPQFastScan index(d, M, b);

    index.train(sample_size, sample_xb.data); // 训练PQ矩阵
    float *cookbook = index.pq.centroids.data();
    std::cerr << "M: " << index.pq.M << "ksub: " << index.pq.ksub << "dsub: " << index.pq.dsub;

    unsigned code_size = index.pq.code_size;
    std::cerr << "PQ code size: " << code_size << std::endl;

    uint8_t *codes = new uint8_t[1ull * n * code_size];
    memset(codes, 0, 1ull * n * code_size);

    unsigned batch_size = 10000;

    std::vector<unsigned> all_assign;
    std::vector<faiss::idx_t> batch_assign(batch_size);
    std::vector<float> batch_dist(batch_size);

    for (unsigned i = 0; i < n; i += batch_size)
    {
        unsigned current_batch_size = std::min(batch_size, n - i);
        Matrix<float> xb(current_batch_size, d);
        // resize 到实际批次大小（保证 operator[] 有效）
        quantizer.search(current_batch_size, base_data.data + 1ull * i * d, 1, batch_dist.data(), batch_assign.data());

        for (unsigned j = 0; j < current_batch_size; ++j)
        {
            compute_sub(base_data.data + 1ull * (i + j) * d, centroid.data + batch_assign[j] * d, xb.data + 1ull * j * d, d);
            all_assign.emplace_back(static_cast<unsigned>(batch_assign[j]));
        }
        xb = mul(xb, opq);
        index.sa_encode(current_batch_size, xb.data, codes + 1ull * i * code_size);
    }

    IVF_PQFastScan ivfpq(n, M, 1<<b, code_size, centroid, base_data.data, codes, all_assign.data(), cookbook);    
    char index_path[256];
    sprintf(index_path, "./data/ivfpq_%s.index", dataset);
    ivfpq.save(index_path);

    return 0;
}

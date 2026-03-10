#include <faiss/IndexFlat.h>
#include <faiss/IndexPQFastScan.h>
#include <faiss/index_io.h>
#include <fastscan.hpp>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <matrix.hpp>
#include <queue>
#include <random>
#include <utils.hpp>

void read_fvecs_batch(std::ifstream &ifs, float *data, unsigned batch_size,
                      unsigned d) {
  for (unsigned i = 0; i < batch_size; ++i) {
    unsigned dim;
    if (!ifs.read((char *)&dim, sizeof(unsigned)))
      break;
    if (dim != d) {
      std::cerr << "Dimension mismatch!" << std::endl;
      exit(-1);
    }
    ifs.read((char *)(data + i * d), d * sizeof(float));
  }
}

int main(int argc, char *argv[]) {

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

  while (iarg != -1) {
    iarg = getopt_long(argc, argv, "d:s:", longopts, &ind);
    switch (iarg) {
    case 'd':
      if (optarg) {
        strcpy(dataset, optarg);
      }
      break;
    case 's':
      if (optarg) {
        strcpy(source, optarg);
      }
      break;
    }
  }

  char data_path[256], centroid_path[256], rotated_path[256];
  sprintf(data_path, "%sbase.fvecs", source);
  sprintf(centroid_path, "%scentroid_4096.fvecs", source);
  sprintf(rotated_path, "%srotation_%s.fvecs", source, dataset);
  unsigned n, d, n_centroids, d_centroids, d_rot;

  Matrix<float> centroid(centroid_path);
  Matrix<float> opq(rotated_path);
  n_centroids = centroid.n;
  d_centroids = centroid.d;
  d_rot = opq.d;
  int sample_size = 1000000;
  int numC = n_centroids; // 粗聚类中心数目

  std::ifstream base_file(data_path, std::ios::binary);
  unsigned d_check;
  base_file.read((char *)&d_check, sizeof(unsigned));
  d = d_check;
  base_file.seekg(0, std::ios::end);
  size_t file_size = base_file.tellg();
  n = file_size / (sizeof(unsigned) + d * sizeof(float));
  std::cerr << "Data dim: " << d << ", base size: " << n
            << ", coarse centers: " << numC << std::endl;
  int M = d / 4;
  int b = 4;
  base_file.seekg(0, std::ios::beg); // 重置到开头

  // 2. 训练阶段：只读取 sample_size 大小的数据
  std::cerr << "Loading " << sample_size << " vectors for training..."
            << std::endl;
  Matrix<float> sample_data(sample_size, d);
  read_fvecs_batch(base_file, sample_data.data, sample_size, d);
  //   d = d_centroids;
  //   Matrix<float> base_data(data_path);
  //   n = base_data.n;
  //   d = base_data.d;

  std::vector<faiss::idx_t> sample_assign(sample_size);
  std::vector<float> sample_dist(sample_size);

  faiss::IndexFlatL2 quantizer(d);
  quantizer.add(numC, centroid.data);
  quantizer.search(sample_size, sample_data.data, 1, sample_dist.data(),
                   sample_assign.data());

  Matrix<float> sample_xb(sample_size, d);
  for (unsigned i = 0; i < sample_size; ++i) {
    compute_sub(sample_data.data + 1ull * i * d,
                centroid.data + sample_assign[i] * d,
                sample_xb.data + 1ull * i * d, d);
  }
  sample_xb = mul(sample_xb, opq);

  faiss::IndexPQFastScan index(d, M, b);

  index.train(sample_size, sample_xb.data); // 训练PQ矩阵
  float *cookbook = index.pq.centroids.data();
  std::cerr << "M: " << index.pq.M << "ksub: " << index.pq.ksub
            << "dsub: " << index.pq.dsub;

  unsigned code_size = index.pq.code_size;
  std::cerr << "PQ code size: " << code_size << std::endl;

  uint8_t *codes = new uint8_t[1ull * n * code_size];
  memset(codes, 0, 1ull * n * code_size);

  unsigned batch_size = 50000;
  float *batch_raw_data = new float[1ull * batch_size * d];
  std::vector<unsigned> all_assign;
  //   std::vector<faiss::idx_t> batch_assign(batch_size);
  //   std::vector<float> batch_dist(batch_size);
  base_file.seekg(0, std::ios::beg);

  for (unsigned i = 0; i < n; i += batch_size) {
    unsigned current_batch_size = std::min(batch_size, n - i);

    // 从磁盘读取当前批次
    read_fvecs_batch(base_file, batch_raw_data, current_batch_size, d);

    std::vector<faiss::idx_t> batch_assign(current_batch_size);
    std::vector<float> batch_dist(current_batch_size);
    quantizer.search(current_batch_size, batch_raw_data, 1, batch_dist.data(),
                     batch_assign.data());

    Matrix<float> xb(current_batch_size, d);
    for (unsigned j = 0; j < current_batch_size; ++j) {
      compute_sub(batch_raw_data + 1ull * j * d,
                  centroid.data + batch_assign[j] * d, xb.data + 1ull * j * d,
                  d);
      all_assign.emplace_back(static_cast<unsigned>(batch_assign[j]));
    }

    xb = mul(xb, opq);
    index.sa_encode(current_batch_size, xb.data,
                    codes + 1ull * i * index.pq.code_size);

    if (i % 1000000 == 0)
      std::cerr << "Processed " << i << " / " << n << " vectors..."
                << std::endl;
  }

  delete[] batch_raw_data;
  base_file.close();
  char index_path[256];
  sprintf(index_path, "./data/ivfpq_%s.index", dataset);
  // 4. 保存索引
  IVF_PQFastScan ivfpq(n, M, 1 << b, index.pq.code_size, centroid, codes,
                       all_assign.data(), index.pq.centroids.data());
  ivfpq.save_disk(index_path);

  return 0;
}

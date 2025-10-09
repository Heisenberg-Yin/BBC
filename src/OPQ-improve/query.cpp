
#include <fastscan.hpp>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <matrix.hpp>
#include <queue>
#include <random>
#include <utils.hpp>

using namespace std;
long double rotation_time = 0;
extern uint32_t rerank_count;

void test(const Matrix<float> &Q, const Matrix<float> &RandQ,
          const Matrix<unsigned> &G, IVF_PQFastScan &ivf, int k) {
  float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
  struct rusage run_start, run_end;

  // ========================================================================
  // Search Parameter

  vector<int> nprobes = {50, 100, 200, 300, 400, 500, 600, 700, 800};
  //   for (int i = 10; i <= 320; i += 30)
  //     nprobes.push_back(i);

  uint32_t NUM_BUCKETS = 48;
  uint32_t PSEUDO_BUCKETS = 256;

  int num_cand = 50000;
  for (auto nprobe : nprobes) {
    TopKBufferSoA KNNs(num_cand, NUM_BUCKETS, PSEUDO_BUCKETS);
    float total_time = 0;
    float total_ratio = 0;
    int correct = 0;
    rerank_count = 0;
    std::vector<std::pair<float, uint32_t>> result_ids_dist;
    for (int i = 0; i < Q.n; i++) {
      KNNs.reset();
      GetCurTime(&run_start);
      result_ids_dist =
          ivf.improved_search(Q.data + i * Q.d, RandQ.data + i * RandQ.d,
                              num_cand, k, nprobe, KNNs);
      GetCurTime(&run_end);
      GetTime(&run_start, &run_end, &usr_t, &sys_t);
      total_time += usr_t * 1e6;

      int tmp_correct = 0;

      std::sort(result_ids_dist.begin(), result_ids_dist.end(),
                [](const std::pair<float, uint32_t> &a,
                   const std::pair<float, uint32_t> &b) {
                  return a.second < b.second;
                });

      uint32_t p = 0, q = 0;
      while (p < result_ids_dist.size() && q < k) {
        if (result_ids_dist[p].second == G.data[i * G.d + q]) {
          ++tmp_correct;
          ++p;
          ++q;
        } else if (result_ids_dist[p].second < G.data[i * G.d + q]) {
          ++p;
        } else {
          ++q;
        }
      }
      correct += tmp_correct;
    }
    float time_us_per_query = total_time / Q.n + rotation_time;
    float recall = 1.0f * correct / (Q.n * k);
    float average_ratio = total_ratio / (Q.n * k);

    cout << "------------------------------------------------" << endl;
    cout << "nprobe = " << nprobe << " k = " << k << " num_cand = " << num_cand
         << endl;
    cout << "Recall = " << recall * 100.000 << "%\t"
         << "Ratio = " << average_ratio << endl;
    cout << "Time = " << time_us_per_query
         << " us \t QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
    cout << "average rerank_count: " << rerank_count / Q.n << endl;
  }
}

int main(int argc, char *argv[]) {

  const struct option longopts[] = {
      {"help", no_argument, 0, 'h'},
      {"dataset", required_argument, 0, 'd'},
      {"k", required_argument, 0, 'k'},
      {"source", required_argument, 0, 's'},

  };

  int ind;
  int iarg = 0;
  opterr = 1; // getopt error message (off: 0)

  char dataset[256] = "";
  char source[256] = "";
  int queryk = 0;
  while (iarg != -1) {
    iarg = getopt_long(argc, argv, "d:k:s:", longopts, &ind);
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
    case 'k':
      if (optarg) {
        queryk = atoi(optarg);
      }
      break;
    case 'h':
      std::cout << "Usage: " << argv[0] << " -d DATASET -s SOURCE -k K"
                << std::endl;
      return 0;
    }
  }

  std::cerr << "dataset: " << dataset << std::endl;
  std::cerr << "source: " << source << std::endl;
  std::cerr << "queryk: " << queryk << std::endl;

  char rotated_path[256] = "";
  sprintf(rotated_path, "../OPQ/data/rotation_%s.fvecs", dataset);
  Matrix<float> opq(rotated_path);

  char query_path[256] = "";
  sprintf(query_path, "%squery.fvecs", source, dataset);
  Matrix<float> query(query_path);

  char groundtruth_path[256] = "";
  sprintf(groundtruth_path, "%stop%d_results.ivecs", source, queryk);
  Matrix<unsigned> G(groundtruth_path);

  char index_path[256];
  sprintf(index_path, "../OPQ/data/ivfpq_%s.index", dataset);

  float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
  struct rusage run_start, run_end;
  GetCurTime(&run_start);

  Matrix<float> RandQ(query.n, query.d, query);
  RandQ = mul(RandQ, opq);

  GetCurTime(&run_end);
  GetTime(&run_start, &run_end, &usr_t, &sys_t);
  rotation_time = usr_t * 1e6 / query.n;

  IVF_PQFastScan ivfpq;
  ivfpq.load(index_path);

  ivfpq.rotate_centroid(opq);

  char result_file_view[256] = "";
  sprintf(result_file_view, "%s%s_ivfpq_scan.log", "./results/", dataset);
  freopen(result_file_view, "a", stdout);

  test(query, RandQ, G, ivfpq, queryk);
  fclose(stdout);
  return 0;
}
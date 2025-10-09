
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

void test(const Matrix<float> &Q, const Matrix<float> &RandQ,
          const Matrix<unsigned> &G, IVF_PQFastScan &ivf, int k) {
  float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
  struct rusage run_start, run_end;
  // ========================================================================
  // Search Parameter
  vector<int> nprobes = {50};
  //   for (int i = 10; i <= 320; i += 30)
  //     nprobes.push_back(i);

  int num_cand = 8000;
  for (auto nprobe : nprobes) {
    float total_time = 0;
    float total_ratio = 0;
    int correct = 0;
    for (int i = 0; i < Q.n; i++) {
      GetCurTime(&run_start);
      ResultHeap KNNs = ivf.search(Q.data + i * Q.d, RandQ.data + i * RandQ.d,
                                   num_cand, k, nprobe);
      GetCurTime(&run_end);
      GetTime(&run_start, &run_end, &usr_t, &sys_t);
      total_time += usr_t * 1e6;

      int tmp_correct = 0;

      std::vector<uint32_t> result_ids;
      result_ids.reserve(k);
      while (KNNs.empty() == false) {
        int id = KNNs.top().second;
        KNNs.pop();
        result_ids.push_back(id);
      }
      sort(result_ids.begin(), result_ids.end());

      uint32_t p = 0, q = 0;
      while (p < result_ids.size() && q < k) {
        if (result_ids[p] == G.data[i * G.d + q]) {
          ++tmp_correct;
          ++p;
          ++q; // 两边都匹配，向前走
        } else if (result_ids[p] < G.data[i * G.d + q]) {
          ++p; // result_ids 小，指针 p 向前
        } else {
          ++q; // ground_ids 小，指针 q 向前
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
  sprintf(rotated_path, "./data/rotation_%s.fvecs", dataset);
  Matrix<float> opq(rotated_path);

  char query_path[256] = "";
  sprintf(query_path, "%squery.fvecs", source, dataset);
  Matrix<float> query(query_path);

  char groundtruth_path[256] = "";
  sprintf(groundtruth_path, "%stop%d_results.ivecs", source, queryk);
  Matrix<unsigned> G(groundtruth_path);

  char index_path[256];
  sprintf(index_path, "./data/ivfpq_%s.index", dataset);

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
#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2
#include <fstream>
#include <iostream>

#include <cmath>
#include <ctime>
#include <getopt.h>
#include <ivf_rabitq.h>
#include <matrix.h>
#include <utils.h>

using namespace std;
extern uint32_t rerank_count;
const int MAXK = 100;

long double rotation_time = 0;

template <uint32_t D, uint32_t B>
void test(const Matrix<float> &Q, const Matrix<float> &RandQ,
          const Matrix<unsigned> &G, const IVFRN<D, B> &ivf, int k) {
  float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
  struct rusage run_start, run_end;

  // ========================================================================
  // Search Parameter
  vector<int> nprobes = {500};
  //   for (int i = 30; i <= 300; i += 30)
  //     nprobes.push_back(i);
  // ========================================================================

  for (auto nprobe : nprobes) {
    float total_time = 0;
    float total_ratio = 0;
    int correct = 0;
    rerank_count = 0;

    for (int i = 0; i < Q.n; i++) {
      GetCurTime(&run_start);
      ResultHeap KNNs =
          ivf.search(Q.data + i * Q.d, RandQ.data + i * RandQ.d, k, nprobe);
      GetCurTime(&run_end);
      GetTime(&run_start, &run_end, &usr_t, &sys_t);
      total_time += usr_t * 1e6;
      // total_ratio += getRatio(i, Q, X, G, KNNs);

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
      // std::cerr << "recall = " << tmp_correct << " / " << k << " " << i + 1
      // << " / " << Q.n << " " << usr_t * 1e6 << "us" << std::endl;
    }
    float time_us_per_query = total_time / Q.n + rotation_time;
    float recall = 1.0f * correct / (Q.n * k);
    float average_ratio = total_ratio / (Q.n * k);

    cout << "------------------------------------------------" << endl;
    cout << "nprobe = " << nprobe << " k = " << k << endl;
    cout << "Recall = " << recall * 100.000 << "%\t"
         << "Ratio = " << average_ratio << endl;
    cout << "Time = " << time_us_per_query
         << " us \t QPS = " << 1e6 / (time_us_per_query) << " query/s" << endl;
    cout << "average rerank_count: " << rerank_count / Q.n << endl;
  }
}

int main(int argc, char *argv[]) {

  const struct option longopts[] = {
      // General Parameter
      {"help", no_argument, 0, 'h'},

      // Query Parameter
      {"K", required_argument, 0, 'k'},

      // Indexing Path
      {"dataset", required_argument, 0, 'd'},
      {"source", required_argument, 0, 's'},
      {"result_path", required_argument, 0, 'r'},
  };

  int ind;
  int iarg = 0;
  opterr = 1; // getopt error message (off: 0)

  char dataset[256] = "";
  char source[256] = "";
  char result_path[256] = "";

  int subk = 0;

  while (iarg != -1) {
    iarg = getopt_long(argc, argv, "d:r:k:s:", longopts, &ind);
    switch (iarg) {
    case 'k':
      if (optarg)
        subk = atoi(optarg);
      break;
    case 's':
      if (optarg)
        strcpy(source, optarg);
      break;
    case 'r':
      if (optarg)
        strcpy(result_path, optarg);
      break;
    case 'd':
      if (optarg)
        strcpy(dataset, optarg);
      break;
    }
  }

  // ================================================================================================================================
  // Data Files
  char query_path[256] = "";
  sprintf(query_path, "%squery.fvecs", source);
  Matrix<float> Q(query_path);

  // char data_path[256] = "";
  // sprintf(data_path, "%s%s_base.fvecs", source, dataset);
  // Matrix<float> X(data_path);

  char groundtruth_path[256] = "";
  sprintf(groundtruth_path, "%stop%d_results.ivecs", source, subk);
  Matrix<unsigned> G(groundtruth_path);

  char transformation_path[256] = "";
  sprintf(transformation_path, "%sP_C%d_B%d.fvecs", source, numC, BB);
  Matrix<float> P(transformation_path);

  char index_path[256] = "";
  sprintf(index_path, "%sivfrabitq%d_B%d.index", source, numC, BB);
  std::cerr << index_path << std::endl;
#if defined(FAST_SCAN)
  char result_file_view[256] = "";
  sprintf(result_file_view, "%s%s_ivfrabitq%d_B%d_fast_scan.log", result_path,
          dataset, numC, BB);
#elif defined(SCAN)
  char result_file_view[256] = "";
  sprintf(result_file_view, "%s%s_ivfrabitq%d_B%d_scan.log", result_path,
          dataset, numC, BB);
#endif
  std::cerr << "Loading Succeed!" << std::endl;
  // ================================================================================================================================

  freopen(result_file_view, "a", stdout);

  IVFRN<DIM, BB> ivf;
  ivf.load(index_path);

  float sys_t, usr_t, usr_t_sum = 0, total_time = 0, search_time = 0;
  struct rusage run_start, run_end;
  GetCurTime(&run_start);

  Matrix<float> RandQ(Q.n, BB, Q);
  RandQ = mul(RandQ, P);

  GetCurTime(&run_end);
  GetTime(&run_start, &run_end, &usr_t, &sys_t);
  rotation_time = usr_t * 1e6 / Q.n;

  test(Q, RandQ, G, ivf, subk);

  return 0;
}

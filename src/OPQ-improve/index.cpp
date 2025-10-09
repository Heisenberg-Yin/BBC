// #include <matrix.hpp>
// #include <utils.hpp>
// #include <fastscan.hpp>
// #include <iostream>
// #include <getopt.h>
// #include <fstream>
// #include <queue>
// #include <random>
// #include <faiss/IndexFlat.h>

// int main(int argc, char *argv[])
// {

//     const struct option longopts[] = {
//         {"help", no_argument, 0, 'h'},
//         {"dataset", required_argument, 0, 'd'},
//         {"source", required_argument, 0, 's'},

//     };

//     int ind;
//     int iarg = 0;
//     opterr = 1; // getopt error message (off: 0)

//     char dataset[256] = "";
//     char source[256] = "";

//     while (iarg != -1)
//     {
//         iarg = getopt_long(argc, argv, "d:s:", longopts, &ind);
//         switch (iarg)
//         {
//         case 'd':
//             if (optarg)
//             {
//                 strcpy(dataset, optarg);
//             }
//             break;
//         case 's':
//             if (optarg)
//             {
//                 strcpy(source, optarg);
//             }
//             break;
//         }
//     }

//     char data_path[256], centroid_path[256], rotated_path[256], pq_path[256];
//     sprintf(data_path, "%sbase.fvecs", source);
//     sprintf(centroid_path, "%scentroid_4096.fvecs", source);
//     sprintf(rotated_path, "./data/rotation_%s.fvecs", dataset);
//     sprintf(pq_path, "./data/base_pq_%s.bvecs", dataset);
//     unsigned n, d, n_centroids, d_centroids, d_rot;
//     Matrix<float> centroid(centroid_path);
//     Matrix<float> base_data(data_path);
//     Matrix<float> opq(rotated_path);
//     n = base_data.n;
//     d = base_data.d;
//     n_centroids = centroid.n;
//     d_centroids = centroid.d;
//     d_rot = opq.d;

//     faiss::IndexFlatL2 quantizer(d);           // 创建L2距离的平坦索引
//     quantizer.add(n_centroids, centroid.data); // 加载所有centroids
//     std::vector<faiss::idx_t> assign(n);       // 保存每个base_data分到的中心id
//     std::vector<float> assign_dist(n);         // 距离，这里可选
//     quantizer.search(n, base_data.data, 1, assign_dist.data(), assign.data());

//     unsigned n_pq, code_size;
//     uint8_t *pq_code;
//     load_bvecs(pq_path, pq_code, n_pq, code_size);

//     unsigned *partition = new unsigned[n]();
//     for (int i = 0; i < assign.size(); i++)
//     {
//         partition[i] = assign[i];
//     }

//     IVF_PQFastScan ivfpq(n, code_size, centroid, base_data.data, pq_code, partition);

//     char index_path[256];
//     sprintf(index_path, "./data/ivfpq_%s.index", dataset);
//     ivfpq.save(index_path);
//     return 0;


// }

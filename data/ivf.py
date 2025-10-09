import numpy as np
import faiss
import struct
import os
from utils.io import *
import hnswlib
import matplotlib.pyplot as plt

source = "/mnt/hdd/yinziqi/yinziqi/large-heap/src/data/"

if __name__ == "__main__":

    # dataset = 'sift1M'
    dataset = "c4-10m"
    # dataset = "c4-15m-bge"
    print(f"Clustering - {dataset}")
    # path
    path = os.path.join(source, dataset)
    data_path = os.path.join(path, f"base.fvecs")
    X = read_fvecs(data_path)
    D = X.shape[1]
    K = 4096
    centroids_path = os.path.join(path, f"centroid_{K}.fvecs")
    dist_to_centroid_path = os.path.join(path, f"dist_to_centroid_{K}.fvecs")
    cluster_id_path = os.path.join(path, f"cluster_id_{K}.ivecs")

    # cluster data vectors
    index = faiss.index_factory(D, f"IVF{K},Flat")
    # res = faiss.StandardGpuResources()  # 创建 GPU 资源
    # index = faiss.index_cpu_to_gpu(res, 0, index)  # 将索引从 CPU 移动到 GPU 上
    index.verbose = True
    sample_size = 10000000  # 100 万个样本
    sampled_indices = np.random.choice(
        X.shape[0], sample_size, replace=False
    )  # 随机抽样索引
    sampled_X = X[sampled_indices]  # 从原始数据中选择这些样本
    index.train(sampled_X)

    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    dist_to_centroid, cluster_id = index.quantizer.search(X, 1)
    dist_to_centroid = dist_to_centroid**0.5

    to_fvecs(dist_to_centroid_path, dist_to_centroid)
    to_ivecs(cluster_id_path, cluster_id)
    to_fvecs(centroids_path, centroids)

    print(cluster_id)
    print(cluster_id.shape)
    cluster_id = cluster_id.ravel()  # 将 cluster_id 拉平为一维数组
    unique_clusters, counts = np.unique(
        cluster_id, return_counts=True
    )  
    # 统计每个聚类的节点数目

    # 计算并输出每个cluster中包含节点数目的中位数
    # cluster_id 的形状为 (N,1)，先将其拉平后再统计
    print(f"平均节点数: {np.mean(counts)}")
    print(f"最大节点数: {np.max(counts)}")
    print(f"最小节点数: {np.min(counts)}")

    plt.hist(counts, bins=50)
    plt.title("聚类节点数目分布")
    plt.xlabel("每个聚类的节点数")
    plt.ylabel("频率")
    plt.show()
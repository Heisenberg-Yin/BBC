import numpy as np
import struct
import time
import os
from utils.io import *
from tqdm import tqdm

source = "./"
datasets = ["c4-10m"]


def Orthogonal(D):
    G = np.random.randn(D, D).astype("float32")
    Q, _ = np.linalg.qr(G)
    return Q


def GenerateBinaryCode(X, P):
    XP = np.dot(X, P)
    binary_XP = XP > 0
    X0 = np.sum(
        XP * (2 * binary_XP - 1) / D**0.5, axis=1, keepdims=True
    ) / np.linalg.norm(XP, axis=1, keepdims=True)
    return binary_XP, X0


if __name__ == "__main__":

    for dataset in datasets:
        # path
        path = os.path.join(source, dataset)
        data_path = os.path.join(path, f"base.fvecs")
        C = 4096
        centroids_path = os.path.join(path, f"centroid_{C}.fvecs")
        dist_to_centroid_path = os.path.join(path, f"dist_to_centroid_{C}.fvecs")
        cluster_id_path = os.path.join(path, f"cluster_id_{C}.ivecs")

        X = read_fvecs(data_path)
        centroids = read_fvecs(centroids_path)
        cluster_id = read_ivecs(cluster_id_path)

        D = X.shape[1]
        B = (D + 63) // 64 * 64
        MAX_BD = max(D, B)
        N = X.shape[0]

        print(f"N - {N}, D - {D}, B - {B}, MAX_BD - {MAX_BD}")

        projection_path = os.path.join(path, f"P_C{C}_B{B}.fvecs")
        randomized_centroid_path = os.path.join(path, f"RandCentroid_C{C}_B{B}.fvecs")
        RN_path = os.path.join(path, f"RandNet_C{C}_B{B}.Ivecs")
        x0_path = os.path.join(path, f"x0_C{C}_B{B}.fvecs")
        print(
            f"projection_path - {projection_path}, randomized_centroid_path - {randomized_centroid_path}, RN_path - {RN_path}, x0_path - {x0_path}"
        )

        X_pad = np.pad(X, ((0, 0), (0, MAX_BD - D)), "constant")
        centroids_pad = np.pad(centroids, ((0, 0), (0, MAX_BD - D)), "constant")
        np.random.seed(0)

        print(
            "the shape of X_pad is ",
            X_pad.shape,
            "the shape of centroids_pad is ",
            centroids_pad.shape,
        )
        print("Generating random projection matrix...")

        # The inverse of an orthogonal matrix equals to its transpose.
        P = Orthogonal(MAX_BD)
        P = P.T
        print("saving projection matrix...")
        CP = np.dot(centroids_pad, P)
        print("saving randomized centroids...")
        cluster_id = np.squeeze(cluster_id)
        batch_size = 10000
        # 用于保存所有 batch 的 x0 和二值编码（uint64）结果
        all_x0 = []
        all_uint64_XP = []
        # 计算每个数据点的二值编码
        for i in tqdm(range(0, N, batch_size), desc="Processing batches"):
            end = min(i + batch_size, N)
            X_batch = X_pad[i:end]  # 当前批次数据
            cluster_id_batch = cluster_id[i:end]  # 当前批次对应的聚类索引
            # 计算投影后的数据，并减去对应聚类中心的投影
            XP_batch = np.dot(X_batch, P) - CP[cluster_id_batch]
            # 二值化处理：大于 0 为 True，否则为 False
            bin_XP_batch = XP_batch > 0
            # 计算 x0：内积归一化
            norm_XP_batch = np.linalg.norm(XP_batch, axis=1, keepdims=True)
            x0_batch = (
                np.sum(
                    XP_batch[:, :B] * (2 * bin_XP_batch[:, :B] - 1) / np.sqrt(B),
                    axis=1,
                    keepdims=True,
                )
                / norm_XP_batch
            )
            # 若计算结果非有限（例如除以零），则赋值为 0.8
            x0_batch[~np.isfinite(x0_batch)] = 0.8
            # 对二值矩阵进行打包：先取前 B 列，再展平，然后每 8 个 bit 组合成一个字节，最终转换为 uint64 类型
            bin_XP_batch = bin_XP_batch[:, :B].flatten()
            uint64_XP_batch = np.packbits(bin_XP_batch.reshape(-1, 8, 8)[:, ::-1]).view(
                np.uint64
            )
            uint64_XP_batch = uint64_XP_batch.reshape(-1, B >> 6)
            # 将当前 batch 的结果追加到列表中
            all_x0.append(x0_batch)
            all_uint64_XP.append(uint64_XP_batch)

        # 将所有 batch 的结果拼接成完整数组
        x0 = np.vstack(all_x0)
        uint64_XP = np.vstack(all_uint64_XP)
        # 将结果保存到文件
        # Output
        to_fvecs(randomized_centroid_path, CP)
        to_Ivecs(RN_path, uint64_XP)
        to_fvecs(x0_path, x0)
        to_fvecs(projection_path, P)

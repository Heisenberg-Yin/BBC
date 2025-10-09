import numpy as np
import torch
import os
import math
import struct
from tqdm import tqdm

source = "./deep100m/"

# source = './sift1M/'


def read_fvecs(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    # print(dim)
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    print("Read %d vectors of dimension %d from %s" % (fv.shape[0], dim, filename))
    return fv


def to_ivecs(filename, data):
    with open(filename, "wb") as f:
        for row in data:
            d = struct.pack("i", len(row))  # 写入向量长度
            f.write(d)
            for x in row:
                f.write(struct.pack("i", int(x)))  # 写入向量元素


def preprocess_and_search(base_vectors, query_vectors, k=10):
    print(
        f"Base vectors shape: {base_vectors.shape}, Query vectors shape: {query_vectors.shape}"
    )

    query_vectors = torch.from_numpy(query_vectors).float()
    base_vectors = torch.from_numpy(base_vectors).float()

    batch_size = 32

    base_vectors_square = torch.pow(base_vectors, 2).sum(1, keepdim=True)

    top_k_results = []
    for i in tqdm(range(0, query_vectors.size(0), batch_size)):
        end = min(i + batch_size, query_vectors.size(0))
        batch_query_vectors = query_vectors[i:end, :]

        batch_query_vectors_square = (
            torch.pow(batch_query_vectors, 2)
            .sum(1, keepdim=True)
            .expand(batch_query_vectors.size()[0], base_vectors.size()[0])
        )

        dist = batch_query_vectors_square + base_vectors_square.t()
        dist.addmm_(1, -2, batch_query_vectors, base_vectors.t())
        dist = dist.clamp(min=1e-12)

        topk_values, topk_indices = torch.topk(-dist, k, dim=1)
        # top_k_results.extend(topk_indices.tolist())
        # 对每一行的 indices 按升序排序，再加入结果列表
        for row in topk_indices.tolist():
            row.sort()
            top_k_results.append(row)

    return top_k_results


if __name__ == "__main__":
    base_path = os.path.join(source, f"base.fvecs")
    query_path = os.path.join(source, f"query.fvecs")
    print("Loading base and query vectors...")
    query = read_fvecs(query_path)
    data = read_fvecs(base_path)
    k = 500  # 设定top-k
    results = preprocess_and_search(data, query, k)
    results_array = np.array(results)
    ivecs_path = os.path.join(source, f"top{k}_results.ivecs")
    to_ivecs(ivecs_path, results_array)

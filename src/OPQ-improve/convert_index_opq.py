import argparse
import os
import numpy as np
import faiss
from collections import defaultdict


def write_fvecs(arr, filename):
    n, d = arr.shape
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(np.array([d], dtype=np.int32).tobytes())
            f.write(arr[i].astype(np.float32).tobytes())


def write_ivecs(arr, filename):
    n, d = arr.shape
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(np.array([d], dtype=np.int32).tobytes())
            f.write(arr[i].astype(np.int32).tobytes())


def write_bvecs(arr, filename):
    n, d = arr.shape
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(np.array([d], dtype=np.int32).tobytes())
            f.write(arr[i].tobytes())


def read_fvecs(filename):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, 1 + dim)
    return fv[:, 1:]


def main():
    parser = argparse.ArgumentParser(
        description="Export decoupled Faiss index components for ANN C++ fastscan"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Dataset root directory (with .fvecs files)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./data",
        help="Output directory for decoupled files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="c4-10m",
        help="Dataset name (subdir under data_root)",
    )
    args = parser.parse_args()

    dataset = args.dataset
    data_root = args.data_root
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 路径拼接
    faiss_index_path = os.path.join(data_root, f"ivfpq_opq_{dataset}.index")

    # 1. 加载 index
    index = faiss.read_index(faiss_index_path)
    assert isinstance(index, faiss.IndexPreTransform)

    print("index type:", type(index))
    if hasattr(index, "index"):
        print("index.index type:", type(index.index))
    else:
        print("index没有 .index 属性")

    opq = index.chain.at(0)
    pq = ivfpq.pq

    assert isinstance(
        ivfpq, faiss.IndexIVFPQ
    ), f"Base index is {type(ivfpq)}, but expected faiss.IndexIVFPQ (your file: {faiss_index_path})"
    pq = ivfpq.pq

    # 2. 导出 OPQ 旋转矩阵 (d, d)
    opq_mat = faiss.vector_to_array(opq.A).reshape(opq.d_out, opq.d_in)
    write_fvecs(opq_mat, os.path.join(out_dir, f"rotation_{dataset}.fvecs"))
    print(f"已保存 rotation_{dataset}.fvecs")

    # 3. 导出 coarse 聚类中心 (nlist, d)
    centroids = faiss.vector_to_array(ivfpq.quantizer.xb).reshape(ivfpq.nlist, ivfpq.d)
    write_fvecs(
        centroids,
        os.path.join(out_dir, os.path.join(out_dir, f"centroids_{dataset}.fvecs")),
    )
    print(f"已保存 centroids_{dataset}.fvecs")

    # 4. 导出 PQ 码本 (M*ksub, dsub)
    M = pq.M
    ksub = pq.ksub
    dsub = pq.d // M
    pq_codebooks = faiss.vector_to_array(pq.centroids).reshape(M, ksub, dsub)
    write_fvecs(
        pq_codebooks.reshape(M * ksub, dsub),
        os.path.join(out_dir, f"pq_codebook_{dataset}.fvecs"),
    )
    print(f"已保存 pq_codebook_{dataset}.fvecs")

    # 2. 读取 PQ codes，shape=(N * code_size)
    codes = faiss.vector_to_array(ivfpq.codes)
    N = ivfpq.ntotal
    code_size = ivfpq.code_size

    # 3. reshape 成 (N, code_size)
    pq_codes = codes.reshape(N, code_size)
    print(f"已加载 PQ codes，shape={pq_codes.shape}")

    write_bvecs(pq_codes, os.path.join(out_dir, f"pq_codes_{dataset}.bvecs"))
    print(f"已保存 pq_codes_{dataset}.bvecs")

    invlists = ivfpq.invlists
    nlist = ivfpq.nlist

    # 直接读取每个聚类的 base ids
    with open(os.path.join(out_dir, f"cluster_offsets_{dataset}.ivecs"), "wb") as f:
        for c in range(nlist):
            list_size = invlists.list_size(c)
            ids_ptr = faiss.rev_swig_ptr(invlists.get_ids(c), list_size)
            ids = np.copy(ids_ptr).astype(np.int32)  # 保证类型兼容
            f.write(np.array([len(ids)], dtype=np.int32).tobytes())
            f.write(ids.tobytes())
    print(
        f"已保存 cluster_offsets_{dataset}.ivecs（每个聚类的 base 索引，直接来自index）"
    )


if __name__ == "__main__":
    main()

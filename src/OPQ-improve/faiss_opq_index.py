import argparse
import os
import numpy as np
import faiss


def read_fvecs(filename, c_contiguous=True):
    print(f"Reading from {filename}.")
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv


def write_fvecs(arr, filename):
    n, d = arr.shape
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(np.array([d], dtype=np.int32).tobytes())
            f.write(arr[i].astype(np.float32).tobytes())


def write_bvecs(arr, filename):
    n, d = arr.shape
    with open(filename, "wb") as f:
        for i in range(n):
            f.write(np.array([d], dtype=np.int32).tobytes())
            f.write(arr[i].tobytes())


def extract_centroids(q: faiss.Index) -> np.ndarray:
    """Extract centroids from quantizer"""
    if hasattr(q, "xb"):
        return faiss.vector_to_array(q.xb).reshape(q.ntotal, q.d)
    if hasattr(q, "get_xb"):
        try:
            return faiss.vector_to_array(q.get_xb()).reshape(q.ntotal, q.d)
        except (AttributeError, AssertionError, TypeError):
            pass
    # Fallback
    out = np.empty((q.ntotal, q.d), dtype="float32")
    buf = np.empty(q.d, dtype="float32")
    for i in range(q.ntotal):
        q.reconstruct(i, buf)
        out[i] = buf
    return out


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train OPQ on residuals and save FastScan components.",
    )

    parser.add_argument("--dataset", default="c4-10m", help="dataset name")
    parser.add_argument("--source", default="./data/", help="root data directory")
    parser.add_argument(
        "--batch", type=int, default=200_000, help="batch size when assigning vectors"
    )
    parser.add_argument(
        "--sample", type=int, default=1_000_000, help="training sample size"
    )
    parser.add_argument(
        "--niter", type=int, default=25, help="k‑means iters for OPQ rotation"
    )
    parser.add_argument("--out", default="./data/", help="root data directory")
    parser.add_argument(
        "--k", type=int, default=100, help="batch size when assigning vectors"
    )

    args = parser.parse_args()

    path = os.path.join(args.source, args.dataset)
    centroid_f = os.path.join(
        "/mnt/hdd/yinziqi/yinziqi/large_aknn/data/", args.dataset, "centroid_4096.fvecs"
    )
    base_f = os.path.join(
        "/mnt/hdd/yinziqi/yinziqi/large_aknn/data/", args.dataset, "base.fvecs"
    )

    if not os.path.exists(base_f):
        raise FileNotFoundError(f"Missing file: {base_f}")
    if not os.path.exists(centroid_f):
        raise FileNotFoundError(f"Missing file: {centroid_f}")

    print("Loading coarse centroids …", flush=True)
    centroids = read_fvecs(centroid_f)
    nlist, d = centroids.shape
    print(f"  centroids: {nlist} × {d}")
    # Choose M (subvectors)
    M = d // 4
    if d % M != 0:
        raise ValueError(f"D={d} is not divisible by M={M}")
    b = 4  # bits per sub-quantizer
    k = 1 << b  # number of centroids per subquantizer
    d_sub = d // M
    print(f"Using M={M}, k={k}, d_sub={d_sub} for PQ")

    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(centroids.astype("float32", copy=False))
    # Sanity: verify write‑back
    if not np.allclose(extract_centroids(quantizer), centroids):
        print("[WARN] centroids in quantizer differ from file (shouldn’t happen)")

    print("Loading base vectors …", flush=True)
    xb = read_fvecs(base_f)  # (N, D)
    n, d = xb.shape
    print(f"  base: {n:,} × {d}")

    rng = np.random.default_rng(123)
    sample_idx = rng.choice(n, min(args.sample, n), replace=False)
    print(f"Training OPQ: M={M}, k={k}, sample={len(sample_idx):,}")
    sample_xb = xb[sample_idx]
    print("Sampling for training...")

    _, assign = quantizer.search(sample_xb, 1)  # assign.shape = (n, 1)

    assign = assign.reshape(-1)
    residuals = sample_xb - centroids[assign]
    # 5. 在残差空间训练OPQ
    opq = faiss.OPQMatrix(d, M)
    opq.verbose = True
    opq.train(residuals.astype(np.float32))
    # 6. 导出旋转矩阵
    A = faiss.vector_to_array(opq.A).reshape(d, d)

    write_fvecs(A, os.path.join(args.out, f"rotation_{args.dataset}.fvecs"))
    print("Saved rotation matrix.")


if __name__ == "__main__":
    main()

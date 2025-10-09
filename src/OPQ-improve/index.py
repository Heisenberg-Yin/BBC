import argparse
import os
import struct
import numpy as np
import faiss
import pickle
import collections
from tqdm import tqdm


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


# ---------- 转换 / 存储 ---------- #
def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)
    print(f"[IO] Pickle saved → {filename}")


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

    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(centroids.astype("float32", copy=False))

    # Sanity: verify write‑back
    if not np.allclose(extract_centroids(quantizer), centroids):
        print("[WARN] centroids in quantizer differ from file (shouldn’t happen)")

    print("Loading base vectors …", flush=True)
    xb = read_fvecs(base_f)  # (N, D)
    n, d = xb.shape
    print(f"  base: {n:,} × {d}")
    assign_ids = np.empty(n, dtype=np.int32)

    bs = args.batch
    print("Assigning vectors to centroids …", flush=True)
    for i0 in range(0, n, bs):
        x_batch = xb[i0 : i0 + bs]
        _, lbl = quantizer.search(x_batch, 1)  # (dist, labels)
        assign_ids[i0 : i0 + lbl.shape[0]] = lbl.reshape(-1)
        if (i0 // bs) % 10 == 0 or i0 + bs >= xb.shape[0]:
            print(f"  Processed {i0 + lbl.shape[0]:,} / {xb.shape[0]:,}", flush=True)

    # 统计每个cluster分配了哪些向量
    clusters = collections.defaultdict(list)
    for idx, cid in enumerate(assign_ids):
        clusters[cid].append(idx)

    print("Cluster assignment statistics:")
    for cid in sorted(clusters):
        print(f"cluster {cid}: {len(clusters[cid])} vectors")

    rng = np.random.default_rng(123)
    sample_idx = rng.choice(n, min(args.sample, n), replace=False)

    # ------------------------------------------------------------------
    # Compute residual vectors
    # ------------------------------------------------------------------
    print("Computing residual vectors …", flush=True)
    sample_residuals = xb[sample_idx] - centroids[assign_ids[sample_idx]]

    # Sample for training
    print("Sampling for training...")

    # Choose M (subvectors)
    M = d // 4
    if d % M != 0:
        raise ValueError(f"D={d} is not divisible by M={M}")
    b = 4  # bits per sub-quantizer
    k = 1 << b  # number of centroids per subquantizer
    d_sub = d // M
    print(f"Using M={M}, k={k}, d_sub={d_sub} for PQ")
    print(f"Training OPQ: M={M}, k={k}, sample={len(sample_residuals):,}")

    args.out = os.path.join(args.out, args.dataset, f"{args.dataset}.opq_{M}_{b}")
    print(f"Output path: {args.out}")
    # Train OPQ
    opq = faiss.OPQMatrix(d, M)
    opq.niter = args.niter
    opq.verbose = True
    opq.train(sample_residuals)

    rotation = faiss.vector_to_array(opq.A).astype(np.float32).reshape(d, d)

    # ------------------------------------------------------------------
    # Train PQ on rotated data
    # ------------------------------------------------------------------
    pq = faiss.ProductQuantizer(d, M, b)
    pq.verbose = True
    sample_residuals_rot = opq.apply_py(sample_residuals)
    pq.train(sample_residuals_rot)

    # Extract PQ codebooks
    pq_centroids = (
        faiss.vector_to_array(pq.centroids)
        .astype(np.float32, copy=False)
        .reshape(M, k, d_sub)
    )
    pkl_name = args.out + ".pkl"
    save_pickle((pq_centroids, rotation), pkl_name)


if __name__ == "__main__":
    main()

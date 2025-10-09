import argparse
import os
import struct
import numpy as np
import faiss
import collections
import gc
from tqdm import tqdm


def read_fvecs(filename, c_contiguous: bool = True) -> np.ndarray:
    """Load a .fvecs file into a 2‑D float32 NumPy array.

    The .fvecs format stores each vector as: <int32 dim> <float32 * dim>.
    If *c_contiguous* is *False* the array may share memory with the mmap’d
    buffer. Setting it to *True* forces a full copy so the data can be written
    to efficiently later on.
    """
    print(f"[I/O] Reading {filename} …", flush=True)
    buf = np.fromfile(filename, dtype=np.float32)
    if buf.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    dim = buf.view(np.int32)[0]
    if dim <= 0:
        raise ValueError(f"Corrupted header in {filename}: dim <= 0")

    buf = buf.reshape(-1, 1 + dim)
    if not np.all(buf.view(np.int32)[:, 0] == dim):
        raise IOError(f"Non‑uniform vector sizes in {filename}")

    data = buf[:, 1:]
    return data.copy() if c_contiguous else data


def to_bvecs(path: str, data: np.ndarray) -> None:
    """Write uint8 codes (n, m) to *path* in .bvecs format."""
    print(f"[I/O] ⇢ {path} (bvecs)")
    with open(path, "wb") as f:
        for v in data:
            f.write(struct.pack("I", len(v)))
            f.write(v.astype(np.uint8, copy=False).tobytes())


def to_fvecs(path: str, data: np.ndarray) -> None:
    """Write float32 matrix (n, d) to *path* in .fvecs format."""
    print(f"[I/O] ⇢ {path} (fvecs)")
    with open(path, "wb") as f:
        for v in tqdm(data, leave=False):
            f.write(struct.pack("I", len(v)))
            f.write(v.astype(np.float32, copy=False).tobytes())


def to_ivecs(path: str, data: np.ndarray, dtype: str = "I") -> None:
    """Write int32 / uint64 vectors (n, d) to *path* in .ivecs format."""
    if dtype not in {"I", "Q"}:
        raise ValueError("dtype must be 'I' (uint32) or 'Q' (uint64)")
    print(f"[I/O] ⇢ {path} (ivecs)")
    with open(path, "wb") as f:
        for v in tqdm(data, leave=False):
            f.write(struct.pack("I", len(v)))
            f.write(struct.pack(f"{dtype*len(v)}", *v))


def extract_centroids(q: faiss.Index) -> np.ndarray:
    """Return a (nlist, d) array of centroids from *q*."""
    if hasattr(q, "xb"):
        return faiss.vector_to_array(q.xb).reshape(q.ntotal, q.d)
    if hasattr(q, "get_xb"):
        try:
            return faiss.vector_to_array(q.get_xb()).reshape(q.ntotal, q.d)
        except (AttributeError, AssertionError, TypeError):
            pass
    # Fallback (slow)
    out = np.empty((q.ntotal, q.d), dtype="float32")
    tmp = np.empty(q.d, dtype="float32")
    for i in range(q.ntotal):
        q.reconstruct(i, tmp)
        out[i] = tmp
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train OPQ on residuals and save FastScan components (memory‑optimized).",
    )
    parser.add_argument("--dataset", default="c4-10m", help="dataset name")
    parser.add_argument("--source", default="./data/", help="root data directory")
    parser.add_argument(
        "--batch",
        type=int,
        default=200_000,
        help="batch size (for assignment/encoding)",
    )
    parser.add_argument(
        "--sample", type=int, default=1_000_000, help="OPQ / PQ training sample size"
    )
    parser.add_argument(
        "--niter", type=int, default=25, help="k‑means iterations for OPQ rotation"
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # Locate files
    # ----------------------------------------------------------------------
    path = os.path.join(args.source, args.dataset)
    centroid_f = os.path.join(
        "/mnt/hdd/yinziqi/yinziqi/large_aknn/data/", args.dataset, "centroid_4096.fvecs"
    )
    base_f = os.path.join(
        "/mnt/hdd/yinziqi/yinziqi/large_aknn/data/", args.dataset, "base.fvecs"
    )

    for f in (centroid_f, base_f):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing file: {f}")

    # ----------------------------------------------------------------------
    # Load / build quantizer
    # ----------------------------------------------------------------------
    centroids = read_fvecs(centroid_f)
    nlist, d = centroids.shape
    print(f"[+] Centroids: {nlist} × {d}")

    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(centroids.astype("float32", copy=False))

    if not np.allclose(extract_centroids(quantizer), centroids):
        print("[WARN] Loaded centroids differ from quantizer buffer!")

    # ----------------------------------------------------------------------
    # First pass: assign vectors to centroids (streamed)
    # ----------------------------------------------------------------------
    print("[Phase 1] Assigning base vectors to centroids …", flush=True)
    xb = read_fvecs(
        base_f
    )  # this may still be large; you can plug your own streaming reader here
    n, _ = xb.shape
    assign_ids = np.empty(n, dtype=np.int32)

    bs = args.batch
    for i0 in tqdm(range(0, n, bs), desc="Assign", unit="vec"):
        x_batch = xb[i0 : i0 + bs]
        _, lbl_batch = quantizer.search(x_batch, 1)
        assign_ids[i0 : i0 + lbl_batch.shape[0]] = lbl_batch.reshape(-1)

    # Build cluster → vector indices mapping (small – int32 per vector)
    clusters = collections.defaultdict(list)
    for idx, cid in enumerate(assign_ids):
        clusters[int(cid)].append(idx)

    print("[Phase 1] Done. Cluster population example:")
    for cid in list(sorted(clusters))[:10]:
        print(f"  • cluster {cid:<5}: {len(clusters[cid]):>8} vectors")

    # ----------------------------------------------------------------------
    # Sample residuals for OPQ / PQ training
    # ----------------------------------------------------------------------
    print("[Phase 2] Sampling residuals for OPQ / PQ training …")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=min(args.sample, n), replace=False)
    sample_residuals = xb[sample_idx] - centroids[assign_ids[sample_idx]]

    # Choose sub‑quantizer configuration
    M = d // 4  # number of sub‑vectors
    if d % M != 0:
        raise ValueError(f"D={d} not divisible by M={M}")
    b = 4  # bits per sub‑quantizer
    k = 1 << b
    d_sub = d // M

    # OPQ rotation
    print(
        f"[Phase 2] Training OPQ: M={M}, k={k}, niter={args.niter}, sample={len(sample_residuals):,}"
    )
    opq = faiss.OPQMatrix(d, M)
    opq.niter = args.niter
    opq.verbose = True
    opq.train(sample_residuals)

    # PQ codebooks (on rotated residuals)
    pq = faiss.ProductQuantizer(d, M, b)
    pq.verbose = True
    pq.train(opq.apply_py(sample_residuals))

    # Keep a copy of rotation + PQ codebooks
    rotation = faiss.vector_to_array(opq.A).astype(np.float32).reshape(d, d)
    pq_centroids = (
        faiss.vector_to_array(pq.centroids).astype(np.float32).reshape(M, k, d_sub)
    )

    del sample_residuals  # memory ‑‑ no longer needed
    gc.collect()
    code_bytes = pq.code_size
    # ----------------------------------------------------------------------
    # Second pass: encode ALL vectors (streamed)
    # ----------------------------------------------------------------------
    print("[Phase 3] Encoding all vectors with OPQ + PQ …", flush=True)
    codes = np.empty(
        (n, code_bytes), dtype=np.uint8
    )  # ~ n × M bytes – reasonable (<1 GB for 10 M×64)

    for i0 in tqdm(range(0, n, bs), desc="Encode", unit="vec"):
        idx = slice(i0, min(i0 + bs, n))
        res_batch = xb[idx] - centroids[assign_ids[idx]]
        codes[idx] = pq.compute_codes(opq.apply_py(res_batch))

    # At this point *xb* can be freed (usually the biggest consumer)
    del xb
    gc.collect()

    # ----------------------------------------------------------------------
    # Persist everything, grouped by cluster
    # ----------------------------------------------------------------------
    print("[Phase 4] Persisting cluster‑organized data …")

    cluster_codes_path = os.path.join(path, f"cluster_codes_C{nlist}_M{M}_B{b}.bvecs")
    cluster_ids_path = os.path.join(
        path, f"cluster_original_ids_C{nlist}_M{M}_B{b}.ivecs"
    )
    cluster_info_path = os.path.join(path, f"cluster_info_C{nlist}_M{M}_B{b}.ivecs")
    rotation_path = os.path.join(path, f"rotation_C{nlist}_M{M}_B{b}.fvecs")
    codebooks_path = os.path.join(path, f"codebooks_C{nlist}_M{M}_B{b}.fvecs")

    # 4a. Rotation + codebooks (small)
    to_fvecs(rotation_path, rotation)
    to_fvecs(codebooks_path, pq_centroids.reshape(-1, d_sub))

    # 4b. Cluster codes & ids – single pass to avoid huge temporaries
    cluster_info = []  # (cluster_size, )

    with open(cluster_codes_path, "wb") as f_codes, open(
        cluster_ids_path, "wb"
    ) as f_ids:
        offset_codes = offset_ids = (
            0  # byte offsets (not used for now, but place‑holders)
        )
        for cid in tqdm(sorted(clusters), desc="Write clusters"):
            idxs = np.asarray(clusters[cid], dtype=np.int32)
            vec_codes = codes[idxs]

            # write codes (.bvecs)
            for v in vec_codes:
                f_codes.write(struct.pack("I", len(v)))
                f_codes.write(v.tobytes())

            # write original IDs (.ivecs, as single‑element vectors)
            for vid in idxs:
                f_ids.write(struct.pack("I", 1))
                f_ids.write(struct.pack("I", int(vid)))

            cluster_info.append(len(idxs))

    # 4c. Cluster metadata (size only – offsets can be added if needed)
    to_ivecs(cluster_info_path, np.asarray(cluster_info, dtype=np.int32).reshape(-1, 1))

    print("[✓] All done. Memory‑optimized artifacts saved to:")
    for p in (
        rotation_path,
        codebooks_path,
        cluster_codes_path,
        cluster_ids_path,
        cluster_info_path,
    ):
        print(f"    • {p}")


if __name__ == "__main__":
    main()

import argparse
import os
import struct
import numpy as np
import faiss
import collections


def read_fvecs(filename: str, c_contiguous: bool = True) -> np.ndarray:
    """Read *.fvecs file into (n, d) float32 ndarray."""
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0), dtype=np.float32)

    dim = int(fv.view(np.int32)[0])
    fv = fv.reshape(-1, 1 + dim)

    if not np.all(fv.view(np.int32)[:, 0] == dim):
        raise IOError(f"Non‑uniform vector sizes in {filename}")

    fv = fv[:, 1:]
    return fv.copy() if c_contiguous else fv


def extract_centroids(q: faiss.Index) -> np.ndarray:
    """Robustly pull centroids from an IndexFlat quantizer, regardless of
    Faiss binding variant. Falls back to per‑centroid `reconstruct()` if
    *.xb* / *get_xb()* are not usable.
    """
    # A) modern pybind ≥1.7
    if hasattr(q, "xb"):
        return faiss.vector_to_array(q.xb).reshape(q.ntotal, q.d)

    # B) SWIG: try get_xb(); some builds still expose it but its return type
    #    changed over time, so guard with try/except.
    if hasattr(q, "get_xb"):
        try:
            return faiss.vector_to_array(q.get_xb()).reshape(q.ntotal, q.d)
        except (AttributeError, AssertionError, TypeError):
            pass  # fall through to slow path

    # C) universal fallback — call reconstruct() for every centroid
    out = np.empty((q.ntotal, q.d), dtype="float32")
    buf = np.empty(q.d, dtype="float32")
    for i in range(q.ntotal):
        q.reconstruct(i, buf)
        out[i] = buf
    return out


# -----------------------------------------------------------------------------
# Helper to persist OPQ (optional, kept from original template)
# -----------------------------------------------------------------------------


def write_opq_binary(
    filepath: str, codebooks: np.ndarray, rotation: np.ndarray
) -> None:
    if not (codebooks.dtype == rotation.dtype == np.float32):
        raise TypeError("codebooks and rotation must be float32")

    M, k, d_sub = codebooks.shape
    dim = M * d_sub
    if rotation.shape != (dim, dim):
        raise ValueError(f"rotation shape must be ({dim}, {dim})")

    b = int(np.log2(k))
    if 1 << b != k:
        raise ValueError("k must be a power of 2")

    with open(filepath, "wb") as f:
        f.write(struct.pack("iii", dim, M, b))
        f.write(codebooks.tobytes(order="C"))
        f.write(rotation.tobytes(order="C"))


# -----------------------------------------------------------------------------
# Main logic: build IVF coarse quantizer and assign each vector to a centroid
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Assign base vectors to IVF coarse clusters (one per vector).",
    )

    parser.add_argument(
        "--dataset", required=True, help="dataset folder name (under --source)"
    )
    parser.add_argument(
        "--source", default="../../../large_aknn/data", help="root data directory"
    )
    parser.add_argument(
        "--batch", type=int, default=200_000, help="batch size when assigning vectors"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------
    # Locate and load data
    # ------------------------------------------------------------
    path = os.path.join(args.source, args.dataset)
    centroid_f = os.path.join(path, "centroid_4096.fvecs")
    base_f = os.path.join(path, "base.fvecs")

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
    print(f"  base: {xb.shape[0]:,} × {xb.shape[1]}")
    # ------------------------------------------------------------------
    # Assign in batches
    # ------------------------------------------------------------------
    assign_ids = np.empty(xb.shape[0], dtype=np.int32)
    bs = args.batch
    print("Assigning vectors to centroids …", flush=True)
    for i0 in range(0, xb.shape[0], bs):
        x_batch = xb[i0 : i0 + bs]
        _, lbl = quantizer.search(x_batch, 1)  # (dist, labels)
        assign_ids[i0 : i0 + lbl.shape[0]] = lbl.reshape(-1)
        if (i0 // bs) % 10 == 0 or i0 + bs >= xb.shape[0]:
            print(f"  Processed {i0 + lbl.shape[0]:,} / {xb.shape[0]:,}", flush=True)

    # 统计每个cluster分配了哪些向量
    clusters = collections.defaultdict(list)
    for idx, cid in enumerate(assign_ids):
        clusters[cid].append(idx)

    for cid in sorted(clusters):
        print(f"cluster {cid}: {len(clusters[cid])} vectors")

    print("Computing residual vectors …", flush=True)
    residuals = np.empty_like(xb)
    for i in range(xb.shape[0]):
        residuals[i] = xb[i] - centroids[assign_ids[i]]

    
if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
opq_bin2pkl.py — Stand‑alone converter: Quick‑ADC OPQ .bin → Python pickle
==========================================================================
Usage
-----
    python opq_bin2pkl.py --bin opq_64x4.bin            # outputs opq_64x4.pkl
    python opq_bin2pkl.py --bin opq.bin --pkl my.pkl    # custom output name

The input .bin must follow Quick‑ADC’s OPQ layout (little‑endian)::

    int32  dim                         # original vector dimension
    int32  M                           # sub‑quantizer count
    int32  b                           # bits per sub‑quantizer (log2 k)
    float32 codebooks[M * k * (dim/M)] # contiguous
    float32 rotation[dim * dim]        # contiguous

The resulting pickle stores a tuple::

    (codebooks: np.ndarray (M, k, dim/M),
     rotation: np.ndarray (dim, dim))

Compatible with Faiss‑CPU 1.8.0 or later.
"""
import argparse
import pickle
import struct
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------
# Loader for Quick‑ADC OPQ binary
# ---------------------------------------------------------------------

def read_opq_binary(path: Path):
    """Return (codebooks[M,k,d_sub], rotation[dim,dim], dim, M, b)."""
    with path.open("rb") as f:
        header = f.read(12)
        if len(header) != 12:
            raise ValueError("File too small or invalid header")
        dim, M, b = struct.unpack("iii", header)
        k = 1 << b
        d_sub = dim // M

        # Load code‑books
        cb_count = M * k * d_sub
        cb_bytes = cb_count * 4  # float32
        codebooks = np.frombuffer(f.read(cb_bytes), dtype=np.float32)
        if codebooks.size != cb_count:
            raise ValueError("File truncated while reading codebooks")
        codebooks = codebooks.reshape(M, k, d_sub)

        # Load rotation matrix
        rot_count = dim * dim
        rotation = np.frombuffer(f.read(rot_count * 4), dtype=np.float32)
        if rotation.size != rot_count:
            raise ValueError("File truncated while reading rotation matrix")
        rotation = rotation.reshape(dim, dim)

    return codebooks, rotation, dim, M, b

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert Quick‑ADC OPQ .bin to pickle")
    ap.add_argument("--bin", required=True, help="input .bin file")
    ap.add_argument("--pkl", help="output pickle filename (.pkl)")
    args = ap.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.exists():
        raise FileNotFoundError(bin_path)

    codebooks, rotation, dim, M, b = read_opq_binary(bin_path)

    pkl_path = Path(args.pkl) if args.pkl else bin_path.with_suffix(".pkl")
    with pkl_path.open("wb") as f:
        pickle.dump((codebooks, rotation), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Converted {bin_path.name} → {pkl_path.name}\n"
        f"  dim={dim}  M={M}  b={b}\n"
        f"  codebooks={tuple(codebooks.shape)}  rotation={tuple(rotation.shape)}"
    )


if __name__ == "__main__":
    main()

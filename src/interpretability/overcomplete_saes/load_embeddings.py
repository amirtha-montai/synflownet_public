"""
Parse graph embeddings out of an ``embeddings.csv`` into cached arrays.

The CSV stores each graph embedding as a stringified torch tensor
(``"tensor([[-4.7186e-01, 9.2863e-01, ...]])"``, ~3.8 kB per row), which is slow
to re-parse on every run. This module parses once and caches ``.npy`` + SMILES,
so downstream scripts load in milliseconds.

Usage
-----
    python load_embeddings.py --embeddings-csv <path> --out-dir <dir>

Outputs (in ``--out-dir``)
    embeddings.npy   float32 [n_molecules, 256]
    smiles.txt       one SMILES per line, row-aligned with embeddings.npy
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

# graph_embeddings fields can be far larger than the default field limit
csv.field_size_limit(10**9)

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_tensor_string(s: str) -> np.ndarray:
    """Extract the float values from a stringified torch tensor.

    Regex rather than ``ast.literal_eval`` because the strings contain embedded
    newlines and a ``tensor(...)`` wrapper, and because this is ~20x faster.
    """
    return np.asarray(_NUM.findall(s), dtype=np.float32)


def load(embeddings_csv: Path, expected_dim: int = 256):
    """Return ``(embeddings [n, dim], smiles [n])`` for all parseable rows."""
    vecs, smiles, skipped = [], [], 0

    with open(embeddings_csv, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            raw = row.get("graph_embeddings")
            smi = row.get("smiles")
            if not raw or not smi:
                skipped += 1
                continue
            v = parse_tensor_string(raw)
            if v.size != expected_dim:
                # A wrong-width row means the CSV is not what we think it is;
                # skip it but keep count so the caller can see the damage.
                skipped += 1
                continue
            vecs.append(v)
            smiles.append(smi)
            if (i + 1) % 5000 == 0:
                print(f"  parsed {i + 1} rows", file=sys.stderr)

    if not vecs:
        raise SystemExit(f"No parseable embeddings found in {embeddings_csv}")

    X = np.stack(vecs)
    print(f"parsed {X.shape[0]} molecules, dim {X.shape[1]}, skipped {skipped}")
    return X, smiles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embeddings-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--expected-dim", type=int, default=256)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    X, smiles = load(args.embeddings_csv, args.expected_dim)

    np.save(args.out_dir / "embeddings.npy", X)
    (args.out_dir / "smiles.txt").write_text("\n".join(smiles) + "\n")
    print(f"wrote {args.out_dir/'embeddings.npy'} and smiles.txt")


if __name__ == "__main__":
    main()

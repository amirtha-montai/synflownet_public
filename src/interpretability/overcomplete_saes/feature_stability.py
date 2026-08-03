"""
Dictionary stability across SAE training runs.

Tests whether interpreted features are real structure or single-run artifacts.
For each pair of seeds, every decoder column in run A is matched to its most
similar column in run B by cosine similarity (columns are unit-norm, so this is
just a dot product). The distribution of best-match similarities says how much of
the dictionary is reproducible.

Feature *indices* are not comparable across runs — they depend on
initialisation. This is exactly why a published feature ID like "#868" cannot be
carried between runs, and why matching must be done geometrically.

Outputs (in ``--out-dir``)
    T_feature_stability.csv         best-match similarity per (seed pair, feature)
    T_feature_stability_summary.csv one row per seed pair
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def decoder_matrix(ckpt_path: Path) -> np.ndarray:
    """Return unit-norm decoder columns [input_dim, n_features]."""
    ck = torch.load(ckpt_path, map_location="cpu")
    W = ck["state_dict"]["W_dec"].numpy()
    return W / np.clip(np.linalg.norm(W, axis=0, keepdims=True), 1e-8, None)


def null_baseline(input_dim: int, n_features: int, n_reps: int = 3, seed: int = 0):
    """Best-match cosine between *independent random* dictionaries.

    Without this, an observed mean of 0.24 is uninterpretable: in 256 dimensions
    the maximum of 1024 random cosines is already ~0.21, so most of the observed
    similarity is what chance produces.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_reps):
        A = rng.normal(size=(input_dim, n_features))
        B = rng.normal(size=(input_dim, n_features))
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        B /= np.linalg.norm(B, axis=0, keepdims=True)
        out.append(np.abs(A.T @ B).max(axis=1))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sae-dir", required=True, type=Path, help="dir with sae_seed*.pt")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--high-sim", type=float, default=0.9,
                    help="threshold for calling a feature reproduced")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpts = sorted(args.sae_dir.glob("sae_seed*.pt"))
    if len(ckpts) < 2:
        raise SystemExit(f"need >=2 checkpoints, found {len(ckpts)} in {args.sae_dir}")
    print(f"{len(ckpts)} runs: {[c.stem for c in ckpts]}")

    mats = {c.stem.replace("sae_seed", ""): decoder_matrix(c) for c in ckpts}

    rows, summary = [], []
    for a, b in itertools.combinations(mats, 2):
        Wa, Wb = mats[a], mats[b]
        # cosine similarity between every column of Wa and every column of Wb
        S = np.abs(Wa.T @ Wb)               # sign-invariant: a feature and its negation are the same direction
        best = S.max(axis=1)
        rows.extend(
            {"seed_a": a, "seed_b": b, "feature_a": i, "best_cosine": float(v)}
            for i, v in enumerate(best)
        )
        summary.append(
            {
                "seed_a": a,
                "seed_b": b,
                "n_features": int(Wa.shape[1]),
                "mean_best_cosine": float(best.mean()),
                "median_best_cosine": float(np.median(best)),
                "frac_above_thresh": float((best >= args.high_sim).mean()),
                "frac_above_0.7": float((best >= 0.7).mean()),
                "frac_above_0.5": float((best >= 0.5).mean()),
            }
        )
        print(f"  seeds {a} vs {b}: mean best cosine {best.mean():.3f}, "
              f"{(best >= args.high_sim).mean():.1%} above {args.high_sim}")

    d, n = next(iter(mats.values())).shape
    null = null_baseline(d, n)
    obs = pd.DataFrame(rows).assign(source="observed")
    nul = pd.DataFrame({"seed_a": "chance", "seed_b": "chance",
                        "feature_a": np.arange(null.size), "best_cosine": null,
                        "source": "chance"})
    pd.concat([obs, nul], ignore_index=True).to_csv(
        args.out_dir / "T_feature_stability.csv", index=False)

    all_obs = obs.best_cosine.to_numpy()
    thr99 = float(np.quantile(null, 0.99))
    print(f"\nnull (random dictionaries): mean {null.mean():.3f}, 99th pct {thr99:.3f}")
    print(f"observed mean {all_obs.mean():.3f}  -> {all_obs.mean()/null.mean():.2f}x chance")
    for t in (0.5, 0.7, 0.9):
        print(f"  >= {t}: observed {(all_obs>=t).mean():.2%}   null {(null>=t).mean():.2%}")
    summary.append({"seed_a": "NULL", "seed_b": "NULL", "n_features": n,
                    "mean_best_cosine": float(null.mean()),
                    "median_best_cosine": float(np.median(null)),
                    "frac_above_thresh": float((null >= args.high_sim).mean()),
                    "frac_above_0.7": float((null >= 0.7).mean()),
                    "frac_above_0.5": float((null >= 0.5).mean())})
    sdf = pd.DataFrame(summary)
    sdf.to_csv(args.out_dir / "T_feature_stability_summary.csv", index=False)

    print("\nacross all seed pairs:")
    print(f"  mean best-match cosine      {sdf.mean_best_cosine.mean():.3f} "
          f"+/- {sdf.mean_best_cosine.std():.3f}")
    print(f"  fraction reproduced (>={args.high_sim}) {sdf.frac_above_thresh.mean():.1%}")
    print(f"\nwrote {args.out_dir}/T_feature_stability.csv")


if __name__ == "__main__":
    main()

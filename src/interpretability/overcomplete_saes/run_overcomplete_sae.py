"""
Train the overcomplete BatchTopK SAE across seeds and report reconstruction,
sparsity, and dead-feature statistics.

Also trains an undercomplete (256 -> 128) autoencoder on the *identical* split
for a matched comparison. The original paper compared an undercomplete model on
one dataset against an overcomplete model on a different dataset while claiming
"identical splits and seeds"; this script makes that comparison actually true.

Outputs (in ``--out-dir``)
    analysis_report.txt     human-readable summary
    metrics.csv             one row per seed
    metrics_summary.csv     mean +/- std across seeds
    latents.npz             latent + reconstruction from the first seed
    sae_seed<N>.pt          model state dicts
    config.json             exact configuration and RDKit version
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from overcomplete_sae import SAEConfig, feature_stats, nmse, train_sae


class UndercompleteAE(nn.Module):
    """Single-layer 256 -> 128 -> 256 autoencoder with L1, for the matched baseline.

    Deliberately a *single* linear layer each way, unlike the original paper's
    four-layer MLP, so that the only difference from the SAE is the bottleneck
    width and the sparsity mechanism.
    """

    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.enc = nn.Linear(input_dim, hidden_dim)
        self.dec = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        z = F.relu(self.enc(x))
        return self.dec(z), z


def train_undercomplete(Xtr, Xva, seed, epochs=50, batch_size=512, lr=3e-4, l1=1e-3,
                        hidden_dim=128, device="cpu"):
    torch.manual_seed(seed)
    m = UndercompleteAE(Xtr.shape[1], hidden_dim).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        idx = torch.randperm(Xtr.shape[0], device=device)
        for s in range(0, Xtr.shape[0], batch_size):
            xb = Xtr[idx[s : s + batch_size]]
            xh, z = m(xb)
            loss = F.mse_loss(xh, xb) + l1 * z.abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        xh, z = m(Xva)
        return m, {
            "nmse": nmse(Xva, xh),
            "mean_l0": float((z > 0).float().sum(dim=1).mean().item()),
            "frac_dead": float(((z > 0).float().sum(dim=0) == 0).float().mean().item()),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n-features", type=int, default=1024)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-undercomplete", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.cache_dir / "embeddings.npy")
    print(f"embeddings: {X.shape} on {args.device}")

    rows = []
    latents_saved = False
    for seed in args.seeds:
        print(f"\n=== overcomplete SAE, seed {seed} ===")
        cfg = SAEConfig(
            input_dim=X.shape[1],
            n_features=args.n_features,
            k=args.k,
            epochs=args.epochs,
            seed=seed,
        )
        model, hist, scaler, stats = train_sae(X, cfg, device=args.device)

        row = {
            "seed": seed,
            "model": "overcomplete_batchtopk",
            "n_features": cfg.n_features,
            "k": cfg.k,
            "nmse": stats["nmse"],
            "mean_l0": stats["mean_l0"],
            "n_dead": stats["n_dead"],
            "frac_dead": stats["frac_dead"],
            "final_train_mse": hist[-1]["train_mse"],
        }
        rows.append(row)
        print(
            f"  -> NMSE {row['nmse']:.5f}  mean L0 {row['mean_l0']:.2f}"
            f"  dead {row['n_dead']}/{cfg.n_features} ({row['frac_dead']:.1%})"
        )

        torch.save(
            {"state_dict": model.state_dict(), "config": cfg.to_dict(),
             "scaler_mean": scaler[0], "scaler_std": scaler[1]},
            args.out_dir / f"sae_seed{seed}.pt",
        )
        pd.DataFrame(hist).to_csv(args.out_dir / f"history_seed{seed}.csv", index=False)

        # save representations from the first seed for the probes
        if not latents_saved:
            mu, sd = scaler
            Xs = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(args.device)
            with torch.no_grad():
                zs, rs = [], []
                for i in range(0, Xs.shape[0], 4096):
                    xb = Xs[i : i + 4096]
                    r, z = model(xb)
                    zs.append(z.cpu().numpy())
                    rs.append(r.cpu().numpy())
            np.savez_compressed(
                args.out_dir / "latents.npz",
                latent=np.concatenate(zs),
                reconstruction=np.concatenate(rs),
                seed=seed,
            )
            latents_saved = True

        # matched undercomplete baseline on the identical split
        if not args.skip_undercomplete:
            mu, sd = scaler
            Xs_np = ((X - mu) / sd).astype(np.float32)
            tr, va = stats["train_idx"], stats["val_idx"]
            _, ustats = train_undercomplete(
                torch.from_numpy(Xs_np[tr]).to(args.device),
                torch.from_numpy(Xs_np[va]).to(args.device),
                seed=seed, epochs=args.epochs, device=args.device,
            )
            rows.append(
                {
                    "seed": seed, "model": "undercomplete_l1", "n_features": 128,
                    "k": np.nan, **ustats, "n_dead": int(ustats["frac_dead"] * 128),
                    "final_train_mse": np.nan,
                }
            )
            print(f"  undercomplete 128: NMSE {ustats['nmse']:.5f}  L0 {ustats['mean_l0']:.1f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "metrics.csv", index=False)
    summary = df.groupby("model", as_index=False).agg(
        nmse_mean=("nmse", "mean"), nmse_std=("nmse", "std"),
        l0_mean=("mean_l0", "mean"), l0_std=("mean_l0", "std"),
        frac_dead_mean=("frac_dead", "mean"), frac_dead_std=("frac_dead", "std"),
        n_seeds=("seed", "nunique"),
    )
    summary.to_csv(args.out_dir / "metrics_summary.csv", index=False)

    import rdkit
    (args.out_dir / "config.json").write_text(
        json.dumps(
            {"seeds": args.seeds, "n_features": args.n_features, "k": args.k,
             "epochs": args.epochs, "n_molecules": int(X.shape[0]),
             "input_dim": int(X.shape[1]), "device": args.device,
             "rdkit": rdkit.__version__, "torch": torch.__version__},
            indent=2,
        )
    )

    with open(args.out_dir / "analysis_report.txt", "w") as f:
        f.write("OVERCOMPLETE BATCHTOPK SAE — ANALYSIS REPORT\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"Molecules: {X.shape[0]}   embedding dim: {X.shape[1]}\n")
        f.write(f"Dictionary: {args.n_features} features ({args.n_features // X.shape[1]}x overcomplete)\n")
        f.write(f"Sparsity:   BatchTopK k={args.k}  (mean L0 should equal k)\n")
        f.write(f"Seeds:      {args.seeds}\n")
        f.write(f"rdkit {rdkit.__version__}   torch {torch.__version__}\n\n")
        f.write("RESULTS (mean +/- std over seeds)\n" + "-" * 78 + "\n")
        f.write(summary.to_string(index=False) + "\n\n")
        f.write("PER-SEED\n" + "-" * 78 + "\n")
        f.write(df.to_string(index=False) + "\n")
    print("\n" + summary.to_string(index=False))
    print(f"\nwrote {args.out_dir}/analysis_report.txt")


if __name__ == "__main__":
    main()

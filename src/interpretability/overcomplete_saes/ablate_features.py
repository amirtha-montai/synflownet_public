"""
Causal validation: ablate individual SAE features and measure the effect on
property predictions.

Correlational evidence ("feature f activates on lipophilic molecules") is weaker
than interventional evidence ("zeroing f changes the model's decodable
lipophilicity"). This script provides the latter:

    encode h -> z, set z[:, f] = 0, decode -> h_hat_ablated,
    re-apply the *already trained* probes, measure Delta per property.

This is ablation only — zeroing a feature and re-reading the frozen probes. It is
not steering: nothing is amplified and generation is never re-run.

A genuinely monosemantic feature should move its targeted property and leave the
others roughly unchanged. The output is a feature x property intervention matrix
plus a specificity score:

    specificity = |Delta targeted| / mean(|Delta non-targeted|)

Probes are trained once on unablated reconstructions and then frozen, so any
change is attributable to the intervention and not to refitting.

Outputs (in ``--out-dir``)
    T_feature_ablation.csv     one row per (feature, property)
    T_ablation_specificity.csv one row per feature
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import properties as P
from overcomplete_sae import BatchTopKSAE, SAEConfig


def load_sae(ckpt_path: Path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device)
    cfg = SAEConfig(**ck["config"])
    model = BatchTopKSAE(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, ck["scaler_mean"], ck["scaler_std"]


def _frozen_probe(Xtr, ytr, is_clf):
    """Fit a probe once; return a callable that scores any matrix."""
    sc = StandardScaler().fit(Xtr)
    if is_clf:
        if len(np.unique(ytr)) < 2:
            return None
        m = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
        return lambda X, y: (
            float(roc_auc_score(y, m.predict_proba(sc.transform(X))[:, 1]))
            if len(np.unique(y)) > 1 else np.nan
        )
    m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    return lambda X, y: float(r2_score(y, m.predict(sc.transform(X))))


@torch.no_grad()
def encode_decode(model, Xs, ablate=None, scale=None, device="cpu", chunk=4096):
    """Encode, optionally zero one feature, decode. Returns reconstructions."""
    out = []
    for i in range(0, Xs.shape[0], chunk):
        xb = torch.from_numpy(Xs[i : i + chunk]).to(device)
        z = model.encode(xb)
        if ablate is not None:
            z = z.clone()
            z[:, ablate] = 0.0 if scale is None else z[:, ablate] * scale
        out.append(model.decode(z).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--sae-checkpoint", required=True, type=Path)
    ap.add_argument("--feature-summary", required=True, type=Path,
                    help="T_feature_summary.csv from characterize_features.py")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-features", type=int, default=12,
                    help="how many top features to ablate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.cache_dir / "embeddings.npy")
    smiles = (args.cache_dir / "smiles.txt").read_text().splitlines()
    targets, valid = P.compute_targets(smiles)
    X = X[valid]
    print(f"{X.shape[0]} molecules, {len(P.ALL_TARGETS)} properties")

    model, cfg, mu, sd = load_sae(args.sae_checkpoint, args.device)
    Xs = ((X - mu) / sd).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(X.shape[0])
    n_te = int(0.2 * X.shape[0])
    te, tr = perm[:n_te], perm[n_te:]

    # baseline reconstructions, and probes frozen on them
    base = encode_decode(model, Xs, device=args.device)
    probes, base_scores = {}, {}
    for t in P.ALL_TARGETS:
        y = targets[t].to_numpy()
        is_clf = t in P.CLASSIFICATION_TARGETS
        fn = _frozen_probe(base[tr], y[tr], is_clf)
        if fn is None:
            continue
        probes[t] = (fn, is_clf)
        base_scores[t] = fn(base[te], y[te])
    print("baseline (unablated reconstruction) scores:")
    for t, v in base_scores.items():
        print(f"  {t:22s} {v:.4f}")

    fs = pd.read_csv(args.feature_summary)
    feats = fs.sort_values("n_properties", ascending=False)["feature"].head(args.n_features).tolist()
    print(f"\nablating {len(feats)} features: {feats}")

    rows = []
    for f in feats:
        for mode, scale in [("ablate", None)]:
            rec = encode_decode(model, Xs, ablate=f, scale=scale, device=args.device)
            frac_active = float((np.abs(base[:, :] - rec[:, :]).sum(axis=1) > 1e-8).mean())
            for t, (fn, is_clf) in probes.items():
                y = targets[t].to_numpy()
                s = fn(rec[te], y[te])
                rows.append(
                    {
                        "feature": int(f),
                        "intervention": mode,
                        "scale": scale if scale is not None else 0.0,
                        "property": t,
                        "metric": "AUROC" if is_clf else "R2",
                        "baseline": base_scores[t],
                        "ablated": s,
                        "delta": s - base_scores[t],
                        "frac_molecules_affected": frac_active,
                    }
                )
        print(f"  feature {f}: max |delta| = "
              f"{max(abs(r['delta']) for r in rows if r['feature']==f):.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "T_feature_ablation.csv", index=False)

    # specificity: is the largest effect concentrated on one property?
    spec = []
    for f, g in df[df.intervention == "ablate"].groupby("feature"):
        g = g.assign(abs_delta=g["delta"].abs()).sort_values("abs_delta", ascending=False)
        top = g.iloc[0]
        others = g.iloc[1:]["abs_delta"]
        spec.append(
            {
                "feature": int(f),
                "most_affected_property": top["property"],
                "delta_targeted": float(top["delta"]),
                "mean_abs_delta_others": float(others.mean()) if len(others) else np.nan,
                "specificity_ratio": float(top["abs_delta"] / others.mean())
                if len(others) and others.mean() > 1e-9 else np.nan,
                "frac_molecules_affected": float(top["frac_molecules_affected"]),
            }
        )
    sdf = pd.DataFrame(spec).sort_values("specificity_ratio", ascending=False)
    sdf.to_csv(args.out_dir / "T_ablation_specificity.csv", index=False)
    print("\nspecificity (higher = effect concentrated on one property):")
    print(sdf.to_string(index=False))
    print(f"\nwrote {args.out_dir}/T_feature_ablation.csv")


if __name__ == "__main__":
    main()

"""
Describe what individual SAE features fire on, and test it statistically.

For each property probe, features are ranked by |weight|. For each top feature we
compare its highest-activating molecules against molecules where it is silent:

* continuous descriptors -> Mann-Whitney U
* substructure presence   -> Fisher's exact test
* p-values corrected across all tests with Benjamini-Hochberg
* "enrichment" = prevalence gap > ``--min-gap`` points at q < ``--alpha``

A feature is called *monosemantic* if it appears in exactly one property's
top-N list, *polysemantic* if it appears in several. That is a statement about
probe weights, not about the feature's semantics, and is labelled as such.

Outputs (in ``--out-dir``)
    T_feature_characterization.csv   one row per (feature, descriptor|substructure)
    T_feature_summary.csv            one row per feature
    T_monosemanticity.csv            mono/poly split counts
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

import properties as P


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Return BH-adjusted q-values."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    if not ok.any():
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest p downward
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    q[ok] = adj
    return q


def top_features_per_target(weights_npz, targets, top_n=20):
    """Map target -> list of feature indices, ranked by mean |weight| over seeds."""
    per_target = {}
    for target in targets:
        keys = [k for k in weights_npz.files if k.startswith(f"{target}__seed")]
        if not keys:
            continue
        W = np.stack([np.abs(weights_npz[k]) for k in keys])  # [seeds, n_features]
        mean_abs = W.mean(axis=0)
        per_target[target] = np.argsort(mean_abs)[::-1][:top_n]
    return per_target


def signed_weight(weights_npz, target, feature):
    keys = [k for k in weights_npz.files if k.startswith(f"{target}__seed")]
    return float(np.mean([weights_npz[k][feature] for k in keys])) if keys else np.nan


def characterize(
    Z, smiles, features, n_top=30, n_silent=30, seed=0
):
    """Compare high-activating vs silent molecules for each feature."""
    rng = np.random.default_rng(seed)
    desc_all = P.compute_characterization_descriptors(smiles)
    sub_all = P.substructure_matrix(smiles)
    rows = []

    for feat in features:
        acts = Z[:, feat]
        active_idx = np.nonzero(acts > 0)[0]
        silent_idx = np.nonzero(acts == 0)[0]
        if active_idx.size < 5 or silent_idx.size < n_silent:
            continue

        top_idx = active_idx[np.argsort(acts[active_idx])[::-1][:n_top]]
        sil_idx = rng.choice(silent_idx, size=min(n_silent, silent_idx.size), replace=False)
        act_freq = float((acts > 0).mean())

        for d in P.CHARACTERIZATION_DESCRIPTORS:
            a = desc_all[d].to_numpy()[top_idx]
            b = desc_all[d].to_numpy()[sil_idx]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if a.size < 3 or b.size < 3:
                continue
            try:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                p = np.nan
            rows.append(
                {
                    "feature": int(feat),
                    "kind": "descriptor",
                    "name": d,
                    "activating_mean": float(np.mean(a)),
                    "silent_mean": float(np.mean(b)),
                    "difference": float(np.mean(a) - np.mean(b)),
                    "activating_prevalence": np.nan,
                    "silent_prevalence": np.nan,
                    "prevalence_gap_pts": np.nan,
                    "p_value": float(p),
                    "activation_frequency": act_freq,
                }
            )

        for s in sub_all.columns:
            a = sub_all[s].to_numpy()[top_idx]
            b = sub_all[s].to_numpy()[sil_idx]
            table = [[int(a.sum()), int((1 - a).sum())], [int(b.sum()), int((1 - b).sum())]]
            try:
                _, p = fisher_exact(table)
            except ValueError:
                p = np.nan
            pa, pb = float(a.mean() * 100), float(b.mean() * 100)
            rows.append(
                {
                    "feature": int(feat),
                    "kind": "substructure",
                    "name": s,
                    "activating_mean": np.nan,
                    "silent_mean": np.nan,
                    "difference": np.nan,
                    "activating_prevalence": pa,
                    "silent_prevalence": pb,
                    "prevalence_gap_pts": pa - pb,
                    "p_value": float(p),
                    "activation_frequency": act_freq,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["q_value"] = benjamini_hochberg(df["p_value"].to_numpy())
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--latents", required=True, type=Path)
    ap.add_argument("--probe-weights", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--top-n-features", type=int, default=20)
    ap.add_argument("--min-gap", type=float, default=30.0, help="prevalence gap, points")
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    smiles = (args.cache_dir / "smiles.txt").read_text().splitlines()
    Z = np.load(args.latents)["latent"]
    W = np.load(args.probe_weights)

    # align: probes were fit on molecules with valid targets
    _, valid = P.compute_targets(smiles)
    smiles = [s for s, v in zip(smiles, valid) if v]
    if Z.shape[0] != len(smiles):
        Z = Z[valid]
    print(f"latents {Z.shape}, {len(smiles)} molecules")

    per_target = top_features_per_target(W, P.ALL_TARGETS, args.top_n_features)
    all_feats = sorted({int(f) for v in per_target.values() for f in v})
    print(f"{len(all_feats)} unique features across {len(per_target)} top-{args.top_n_features} lists")

    df = characterize(Z, smiles, all_feats)
    if df.empty:
        raise SystemExit("no features could be characterised")
    df["enriched"] = (df["prevalence_gap_pts"].abs() > args.min_gap) & (df["q_value"] < args.alpha)
    df["rdkit_version"] = P.version_info()["rdkit"]
    df.to_csv(args.out_dir / "T_feature_characterization.csv", index=False)

    # per-feature summary: which properties it serves, and its strongest enrichment
    summary = []
    for feat in all_feats:
        serves = [t for t, v in per_target.items() if feat in set(int(x) for x in v)]
        sub = df[(df.feature == feat) & df.enriched & (df.kind == "substructure")]
        sub = sub.reindex(sub["prevalence_gap_pts"].abs().sort_values(ascending=False).index)
        summary.append(
            {
                "feature": feat,
                "n_properties": len(serves),
                "properties": "|".join(serves),
                "semanticity": "monosemantic" if len(serves) == 1 else "polysemantic",
                "activation_frequency": float((Z[:, feat] > 0).mean()),
                "top_enriched_substructures": "|".join(
                    f"{r['name']}({r['prevalence_gap_pts']:+.0f}pts)" for _, r in sub.head(3).iterrows()
                ),
                "n_enriched": int(sub.shape[0]),
                **{f"w_{t}": signed_weight(W, t, feat) for t in P.ALL_TARGETS},
            }
        )
    sdf = pd.DataFrame(summary).sort_values("n_properties", ascending=False)
    sdf.to_csv(args.out_dir / "T_feature_summary.csv", index=False)

    n_mono = int((sdf.semanticity == "monosemantic").sum())
    n_poly = int((sdf.semanticity == "polysemantic").sum())
    tot = max(n_mono + n_poly, 1)
    pd.DataFrame(
        [
            {
                "n_features_in_any_top_list": tot,
                "n_monosemantic": n_mono,
                "pct_monosemantic": 100.0 * n_mono / tot,
                "n_polysemantic": n_poly,
                "pct_polysemantic": 100.0 * n_poly / tot,
                "top_n_used": args.top_n_features,
            }
        ]
    ).to_csv(args.out_dir / "T_monosemanticity.csv", index=False)

    print(f"\nmonosemantic {n_mono}/{tot} ({100*n_mono/tot:.0f}%), polysemantic {n_poly}")
    print("\nmost polysemantic features:")
    print(sdf.head(8)[["feature", "n_properties", "properties",
                       "activation_frequency", "top_enriched_substructures"]].to_string(index=False))
    print("\nlargest negative drug-likeness weights:")
    neg = sdf.nsmallest(5, "w_drug_likeness")
    print(neg[["feature", "w_drug_likeness", "top_enriched_substructures"]].to_string(index=False))
    print(f"\nwrote {args.out_dir}/T_feature_characterization.csv and T_feature_summary.csv")


if __name__ == "__main__":
    main()

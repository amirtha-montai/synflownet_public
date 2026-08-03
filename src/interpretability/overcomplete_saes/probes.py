"""
Linear probes on raw embeddings, SAE latents, and SAE reconstructions —
with the control baselines that make the numbers interpretable.

Probes are *genuinely linear* (Ridge for regression, LogisticRegression for
classification). This matters: the original paper reported R^2 values from a
three-layer MLP while describing them as linear predictors, which conflates
"the information is linearly accessible" with "some network can extract it".

For every (representation, target) pair we report four columns:

    trained       probe on the real representation
    random_label  same probe, labels shuffled -> should collapse to chance
    descriptor    same probe on RDKit counts + ECFP4, no model involved at all
    gap           trained - descriptor

The ``gap`` column is the point of the exercise. A near-perfect score with a
near-perfect descriptor baseline means the property was trivially present in the
input, not that the model learned anything; only a positive gap is evidence the
representation adds value.

Outputs
-------
    <out>/T_probes.csv               one row per (representation, target, seed)
    <out>/T_probes_summary.csv       mean +/- std across seeds
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import properties as P

CHANCE_AUROC = 0.5


def _split(n, seed, val_frac=0.10, test_frac=0.10):
    """Deterministic train/val/test split. Val is reserved for future tuning."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    return perm[n_test + n_val :], perm[n_test : n_test + n_val], perm[:n_test]


def _fit_regression(Xtr, ytr, Xte, yte, alpha=1.0):
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    return float(r2_score(yte, m.predict(sc.transform(Xte)))), m.coef_.ravel()


def _fit_classification(Xtr, ytr, Xte, yte):
    # a target with one class in train or test yields no meaningful AUROC
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan"), np.zeros(Xtr.shape[1])
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xtr), ytr)
    p = m.predict_proba(sc.transform(Xte))[:, 1]
    return float(roc_auc_score(yte, p)), m.coef_.ravel()


def run_probes(
    representations: dict,
    targets: pd.DataFrame,
    descriptor_X: np.ndarray,
    seeds=(0, 1, 2, 3, 4),
    save_weights_for: str = "latent",
):
    """Probe every representation against every target, with controls.

    ``representations`` maps name -> array [n, d]. ``save_weights_for`` names the
    representation whose probe weight vectors are returned, for use by feature
    characterisation.
    """
    rows, weights = [], {}
    n = len(targets)

    for seed in seeds:
        tr, _va, te = _split(n, seed)
        rng = np.random.default_rng(seed)

        for target in P.ALL_TARGETS:
            y = targets[target].to_numpy()
            is_clf = target in P.CLASSIFICATION_TARGETS
            fit = _fit_classification if is_clf else _fit_regression
            chance = CHANCE_AUROC if is_clf else 0.0

            # descriptor baseline: no model, identical probe and split.
            # Two variants, because some targets are *defined* by baseline columns
            # (flexibility == NumRotatableBonds/10), which makes the full-baseline
            # comparison arithmetic rather than informative.
            desc_score, _ = fit(descriptor_X[tr], y[tr], descriptor_X[te], y[te])
            leaky = P.leaky_column_indices(target)
            if leaky:
                keep = [i for i in range(descriptor_X.shape[1]) if i not in set(leaky)]
                Xd = descriptor_X[:, keep]
                desc_nl, _ = fit(Xd[tr], y[tr], Xd[te], y[te])
            else:
                desc_nl = desc_score

            for rep_name, X in representations.items():
                score, coef = fit(X[tr], y[tr], X[te], y[te])

                y_shuf = y.copy()
                rng.shuffle(y_shuf)
                rand_score, _ = fit(X[tr], y_shuf[tr], X[te], y_shuf[te])

                rows.append(
                    {
                        "representation": rep_name,
                        "target": target,
                        "task": "classification" if is_clf else "regression",
                        "metric": "AUROC" if is_clf else "R2",
                        "seed": seed,
                        "trained": score,
                        "random_label": rand_score,
                        "descriptor": desc_score,
                        "descriptor_noleak": desc_nl,
                        "baseline_leaky": bool(leaky),
                        "gap_vs_descriptor": score - desc_score,
                        "gap_vs_descriptor_noleak": score - desc_nl,
                        "chance": chance,
                        "prevalence": float(np.mean(y)) if is_clf else np.nan,
                    }
                )
                if rep_name == save_weights_for:
                    weights.setdefault(target, {})[seed] = coef

    return pd.DataFrame(rows), weights


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["representation", "target", "task", "metric"], as_index=False).agg(
        trained_mean=("trained", "mean"),
        trained_std=("trained", "std"),
        random_label_mean=("random_label", "mean"),
        descriptor_mean=("descriptor", "mean"),
        descriptor_noleak_mean=("descriptor_noleak", "mean"),
        gap_mean=("gap_vs_descriptor", "mean"),
        gap_std=("gap_vs_descriptor", "std"),
        gap_noleak_mean=("gap_vs_descriptor_noleak", "mean"),
        gap_noleak_std=("gap_vs_descriptor_noleak", "std"),
        baseline_leaky=("baseline_leaky", "first"),
        prevalence=("prevalence", "first"),
        n_seeds=("seed", "nunique"),
    )
    # flag targets the controls show to be trivially decodable
    # judged against the leakage-free baseline, so the flag means something
    g["trivially_decodable"] = g["descriptor_noleak_mean"] >= (g["trained_mean"] - 0.02)
    return g.sort_values(["representation", "task", "trained_mean"], ascending=[True, True, False])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, type=Path, help="from load_embeddings.py")
    ap.add_argument("--latents", type=Path, help="latents.npz from run_overcomplete_sae.py")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.cache_dir / "embeddings.npy")
    smiles = (args.cache_dir / "smiles.txt").read_text().splitlines()
    assert len(smiles) == X.shape[0], (len(smiles), X.shape)

    print("computing targets ...")
    targets, valid = P.compute_targets(smiles)
    X = X[valid]
    smiles = [s for s, v in zip(smiles, valid) if v]
    print(f"  {len(targets)} valid molecules ({(~valid).sum()} dropped)")

    print("computing descriptor baseline (counts + ECFP4) ...")
    desc = P.compute_descriptor_baseline(smiles)

    reps = {"raw": X}
    if args.latents and args.latents.exists():
        z = np.load(args.latents)
        for key in ("latent", "reconstruction"):
            if key in z:
                arr = z[key]
                reps[key] = arr[valid] if arr.shape[0] == valid.size else arr
        print(f"  loaded representations: {list(reps)}")

    print(f"running probes over seeds {args.seeds} ...")
    df, weights = run_probes(reps, targets, desc, seeds=tuple(args.seeds))
    df["rdkit_version"] = P.version_info()["rdkit"]

    df.to_csv(args.out_dir / "T_probes.csv", index=False)
    summary = summarize(df)
    summary.to_csv(args.out_dir / "T_probes_summary.csv", index=False)

    if weights:
        np.savez_compressed(
            args.out_dir / "probe_weights_latent.npz",
            **{f"{t}__seed{s}": w for t, d in weights.items() for s, w in d.items()},
        )
    (args.out_dir / "probe_meta.json").write_text(
        json.dumps(
            {
                "n_molecules": int(len(targets)),
                "seeds": args.seeds,
                "representations": list(reps),
                **P.version_info(),
            },
            indent=2,
        )
    )
    print(summary.to_string(index=False))
    print(f"\nwrote {args.out_dir}/T_probes.csv and T_probes_summary.csv")


if __name__ == "__main__":
    main()

"""
Publication figures for the overcomplete-SAE analysis.

Every figure is written to both PDF (vector, for the manuscript) and 300 dpi PNG.
Figures are built only from the CSV tables, never by re-running models, so they
can be regenerated without a GPU.

Design rules applied throughout:
  * categorical hues assigned in fixed order, never cycled (max 3 series here)
  * one y-axis per panel; never a dual-axis chart
  * sequential = one hue light->dark; diverging = blue/red with a neutral gray
    midpoint (never a rainbow, never a hue at the midpoint)
  * thin marks, recessive grid and axes, no top/right spines
  * a legend whenever >= 2 series; direct value labels on bars
  * text in ink colors, never in the series color

Usage
-----
    python make_plots.py --results-dir results_5seed --out-dir figs
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# ---------------------------------------------------------------- palette
# Validated reference values, used verbatim. Only the first three categorical
# slots are used: that subset is documented as clearing the all-pairs CVD and
# normal-vision floors in both light and dark modes.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # blue, orange, aqua
DIV_LO, DIV_MID, DIV_HI = "#2a78d6", "#f0efec", "#d03b3b"   # blue / gray / red
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e6e5e1"
SURFACE = "#ffffff"

DIVERGING = LinearSegmentedColormap.from_list("div", [DIV_LO, DIV_MID, DIV_HI])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", SEQ)

PRETTY = {
    "drug_likeness": "Drug-likeness (QED)",
    "complexity_ringproxy": "Complexity (ring proxy)",
    "lipophilicity": "Lipophilicity",
    "size": "Size",
    "polarity": "Polarity",
    "flexibility": "Flexibility",
    "molecular_weight": "Molecular weight",
    "halogen": "Halogen",
    "aromaticity": "Aromaticity",
    "urea": "Urea",
    "boron": "Boron",
}


def style(ax, xlabel=None, ylabel=None, title=None, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300, bbox_inches="tight",
                    facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# ------------------------------------------------------- F1 decodability
def fig_decodability(summary: pd.DataFrame, out_dir: Path):
    """Embedding vs descriptor baseline vs random-label, per property."""
    d = summary[summary.representation == "raw"].copy()
    d["label"] = d.target.map(PRETTY).fillna(d.target)
    d = d.sort_values("trained_mean", ascending=True)

    y = np.arange(len(d))
    h = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 0.46 * len(d) + 1.6))
    ax.barh(y + h, d.trained_mean, h, color=S1, label="SynFlowNet embedding",
            zorder=3, edgecolor=SURFACE, linewidth=1.0,
            xerr=d.trained_std.fillna(0), error_kw=dict(ecolor=INK3, lw=0.8, capsize=2))
    ax.barh(y, d.descriptor_noleak_mean, h, color=S2,
            label="RDKit descriptors + ECFP4 (no model)", zorder=3,
            edgecolor=SURFACE, linewidth=1.0)
    ax.barh(y - h, d.random_label_mean, h, color=S3, label="Random-label control",
            zorder=3, edgecolor=SURFACE, linewidth=1.0)

    for yi, v in zip(y + h, d.trained_mean):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
    for yi, v in zip(y, d.descriptor_noleak_mean):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)

    ax.set_yticks(y, d.label, fontsize=8.5, color=INK)
    ax.set_xlim(-0.05, 1.15)
    style(ax, xlabel="R² (regression) or AUROC (classification)", grid_axis="x",
          title="Linear decodability against trivial baselines")
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)
    save(fig, out_dir, "F1_decodability_vs_controls")


# ------------------------------------------------------------- F2 the gap
def fig_gap(summary: pd.DataFrame, out_dir: Path):
    """What the embedding adds over cheap descriptors. Diverging about zero."""
    d = summary[summary.representation == "raw"].copy()
    d["label"] = d.target.map(PRETTY).fillna(d.target)
    d = d.sort_values("gap_noleak_mean")
    colors = [DIV_HI if v < 0 else DIV_LO for v in d.gap_noleak_mean]

    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(d) + 1.5))
    y = np.arange(len(d))
    ax.barh(y, d.gap_noleak_mean, 0.6, color=colors, zorder=3,
            edgecolor=SURFACE, linewidth=1.0)
    ax.axvline(0, color=INK2, linewidth=1.0, zorder=4)
    for yi, v in zip(y, d.gap_noleak_mean):
        ax.text(v + (0.006 if v >= 0 else -0.006), yi, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=7.5, color=INK2)
    ax.set_yticks(y, d.label, fontsize=8.5, color=INK)
    style(ax, xlabel="Embedding − descriptor baseline", grid_axis="x",
          title="Where the learned embedding adds information")
    ax.text(0.99, 0.02, "right of zero: embedding wins    left: descriptors win",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=INK3)
    save(fig, out_dir, "F2_gap_vs_descriptor_baseline")


# ------------------------------------------- F3 overcomplete vs undercomplete
def fig_head_to_head(metrics: pd.DataFrame, out_dir: Path):
    """Reconstruction and sparsity, on the identical split. Two panels, one axis each."""
    g = metrics.groupby("model")
    order = [m for m in ["undercomplete_l1", "overcomplete_batchtopk"] if m in g.groups]
    names = {"undercomplete_l1": "Undercomplete\n256→128 (ℓ1)",
             "overcomplete_batchtopk": "Overcomplete\n256→1024 (BatchTopK)"}
    cols = {"undercomplete_l1": S2, "overcomplete_batchtopk": S1}

    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.3))
    for ax, (col, lab, fmt) in zip(
        axes,
        [("nmse", "Reconstruction NMSE (lower better)", "{:.4f}"),
         ("mean_l0", "Mean L0 (active features)", "{:.1f}"),
         ("frac_dead", "Dead features (%)", "{:.2f}%")],
    ):
        x = np.arange(len(order))
        scale = 100.0 if col == "frac_dead" else 1.0
        mu = [g.get_group(m)[col].mean() * scale for m in order]
        sd = [np.nan_to_num(g.get_group(m)[col].std()) * scale for m in order]
        ax.bar(x, mu, 0.52, color=[cols[m] for m in order], zorder=3,
               edgecolor=SURFACE, linewidth=1.2,
               yerr=sd, error_kw=dict(ecolor=INK3, lw=0.9, capsize=3))
        # place the label clear of the error-bar cap
        head = max([m + e for m, e in zip(mu, sd)] + [1e-12])
        for xi, v, e in zip(x, mu, sd):
            ax.text(xi, v + e + 0.045 * head, fmt.format(v), ha="center",
                    va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, [names[m] for m in order], fontsize=7.5, color=INK)
        style(ax, ylabel=lab)
        ax.set_ylim(0, head * 1.30 if head > 0 else 1)
        if col == "mean_l0":
            # k is the design target; show it so a flat line reads as "pinned at k"
            ax.axhline(64, color=INK3, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
            ax.text(ax.get_xlim()[1], 64, " k=64", va="center", fontsize=7.5, color=INK3)
    fig.suptitle("Overcomplete vs undercomplete on the identical split",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out_dir, "F3_overcomplete_vs_undercomplete")


# ------------------------------------------------- F4 information preservation
def fig_preservation(summary: pd.DataFrame, out_dir: Path):
    """raw -> latent -> reconstruction, per property. Slope chart."""
    reps = ["raw", "latent", "reconstruction"]
    have = [r for r in reps if r in set(summary.representation)]
    if len(have) < 2:
        return
    piv = summary.pivot_table(index="target", columns="representation",
                              values="trained_mean")[have]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    x = np.arange(len(have))
    for t, row in piv.iterrows():
        ax.plot(x, row.values, marker="o", markersize=5, linewidth=1.6,
                color=S1, alpha=0.75, zorder=3)
        ax.annotate(PRETTY.get(t, t), (x[-1], row.values[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=7.5, color=INK2, va="center")
    ax.set_xticks(x, ["Raw\nembedding", "SAE\nlatent", "SAE\nreconstruction"][: len(have)],
                  fontsize=8.5, color=INK)
    ax.set_xlim(-0.25, len(have) - 0.25 + 1.1)
    style(ax, ylabel="R² or AUROC",
          title="Information retained through the sparse bottleneck")
    save(fig, out_dir, "F4_information_preservation")


# ----------------------------------------------------- F5 training diagnostics
def fig_training(results_dir: Path, out_dir: Path):
    hs = sorted(results_dir.glob("history_seed*.csv"))
    if not hs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    for i, (col, lab) in enumerate(
        [("val_nmse", "Validation NMSE"),
         ("val_mean_l0", "Mean L0 per epoch"),
         ("n_dead_epoch", "Dead features this epoch")]
    ):
        for h in hs:
            d = pd.read_csv(h)
            # seeds are interchangeable replicates, so identity carries no meaning:
            # one hue, no legend, rather than a ramp misused as a categorical scale
            axes[i].plot(d.epoch, d[col], linewidth=1.2, color=S1, alpha=0.55, zorder=3)
        style(axes[i], xlabel="Epoch", ylabel=lab)
        if col == "val_nmse":
            axes[i].set_yscale("log")
        if col == "val_mean_l0":
            # otherwise matplotlib renders a "+6.4e1" offset and the panel looks broken
            axes[i].set_ylim(0, 80)
            axes[i].axhline(64, color=INK3, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
            axes[i].text(0.98, 0.70, "pinned at k = 64", transform=axes[i].transAxes,
                         ha="right", fontsize=7.5, color=INK3)
            axes[i].ticklabel_format(useOffset=False, axis="y")
    axes[0].text(0.97, 0.93, f"{len(hs)} seeds overlaid", transform=axes[0].transAxes,
                 ha="right", fontsize=7.5, color=INK3)
    fig.suptitle("SAE training diagnostics across seeds", color=INK, fontsize=11,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, out_dir, "F5_training_diagnostics")


# --------------------------------------------------------- F6 feature stability
def fig_stability(results_dir: Path, out_dir: Path):
    f = results_dir / "T_feature_stability.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    obs = d[d.source == "observed"].best_cosine if "source" in d else d.best_cosine
    null = d[d.source == "chance"].best_cosine if "source" in d else None

    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    bins = np.linspace(0, 1, 61)
    if null is not None and len(null):
        # chance drawn as an outline so the overlap reads as overlap, not a stack
        ax.hist(null, bins=bins, weights=np.full(len(null), 100 / len(null)),
                histtype="step", color=S2, linewidth=1.8, zorder=4,
                label="Random dictionaries (chance)")
    ax.hist(obs, bins=bins, weights=np.full(len(obs), 100 / len(obs)),
            color=S1, edgecolor=SURFACE, linewidth=0.4, zorder=3,
            label="Trained SAEs (seed pairs)")
    if null is not None and len(null):
        thr = float(np.quantile(null, 0.99))
        ax.axvline(thr, color=INK2, linewidth=1.1, linestyle=(0, (4, 3)), zorder=5)
        ax.text(thr + 0.012, ax.get_ylim()[1] * 0.92,
                f"99th pct of chance = {thr:.2f}\n"
                f"{(obs >= thr).mean() * 100:.0f}% of features above",
                fontsize=7.5, color=INK2, va="top")
    ax.set_xlim(0.10, 0.92)
    style(ax, xlabel="Best-match cosine similarity to another dictionary",
          ylabel="% of features",
          title="Are individual features reproducible across runs?")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right",
              bbox_to_anchor=(1.0, 0.62))
    save(fig, out_dir, "F6_feature_stability")


# ------------------------------------------------------ F7 intervention matrix
def fig_intervention(results_dir: Path, out_dir: Path):
    f = results_dir / "T_feature_ablation.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d[d.intervention == "ablate"]
    piv = d.pivot_table(index="feature", columns="property", values="delta")
    piv = piv[[c for c in PRETTY if c in piv.columns]]
    lim = float(np.nanmax(np.abs(piv.values))) or 1e-6

    fig, ax = plt.subplots(figsize=(0.62 * piv.shape[1] + 3.2, 0.34 * piv.shape[0] + 2.2))
    im = ax.imshow(piv.values, cmap=DIVERGING,
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim), aspect="auto")
    ax.set_xticks(range(piv.shape[1]),
                  [PRETTY.get(c, c) for c in piv.columns],
                  rotation=40, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(piv.shape[0]), [f"#{i}" for i in piv.index],
                  fontsize=8, color=INK)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v) and abs(v) > 0.2 * lim:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5,
                        color=INK if abs(v) < 0.6 * lim else SURFACE)
    ax.set_title("Effect of ablating each feature on property decoding",
                 color=INK, fontsize=10.5, loc="left", pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Δ in R² / AUROC after ablation", color=INK2, fontsize=8)
    cb.ax.tick_params(colors=INK2, labelsize=7)
    cb.outline.set_visible(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, out_dir, "F7_intervention_matrix")


# ------------------------------------------------------------- F8 semanticity
def fig_semanticity(results_dir: Path, out_dir: Path):
    fs = results_dir / "T_feature_summary.csv"
    if not fs.exists():
        return
    d = pd.read_csv(fs)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))

    counts = d.n_properties.value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, 0.6,
                color=[S1 if k == 1 else S2 for k in counts.index],
                zorder=3, edgecolor=SURFACE, linewidth=1.2)
    for k, v in counts.items():
        axes[0].text(k, v, f" {v}", ha="center", va="bottom", fontsize=8, color=INK2)
    axes[0].set_xticks(counts.index, [str(int(k)) for k in counts.index],
                       fontsize=8.5, color=INK)
    style(axes[0], xlabel="Number of properties a feature serves",
          ylabel="Number of features",
          title="Monosemantic (1) vs polysemantic (≥2)")

    axes[1].hist(d.activation_frequency, bins=30, color=S1,
                 edgecolor=SURFACE, linewidth=0.5, zorder=3)
    style(axes[1], xlabel="Activation frequency", ylabel="Number of features",
          title="How often features fire")
    fig.tight_layout()
    save(fig, out_dir, "F8_feature_semanticity")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "figure.facecolor": SURFACE})

    print(f"building figures from {args.results_dir}")
    sm = args.results_dir / "T_probes_summary.csv"
    if sm.exists():
        summary = pd.read_csv(sm)
        fig_decodability(summary, args.out_dir)
        fig_gap(summary, args.out_dir)
        fig_preservation(summary, args.out_dir)
    mt = args.results_dir / "metrics.csv"
    if mt.exists():
        fig_head_to_head(pd.read_csv(mt), args.out_dir)
    fig_training(args.results_dir, args.out_dir)
    fig_stability(args.results_dir, args.out_dir)
    fig_intervention(args.results_dir, args.out_dir)
    fig_semanticity(args.results_dir, args.out_dir)
    print(f"\nfigures in {args.out_dir}/ (PDF for the manuscript, PNG for preview)")


if __name__ == "__main__":
    main()

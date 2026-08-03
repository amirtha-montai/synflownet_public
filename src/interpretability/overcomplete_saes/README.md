# Overcomplete BatchTopK Sparse Autoencoder

Feature-level interpretability of SynFlowNet graph embeddings, with the control
baselines and multi-seed error bars needed to tell learned structure from
trivially decodable input properties.

Dataset: the **32,054-molecule** set from arXiv 2511.19264 (256-d graph embeddings).

---

## Why this is separate from the undercomplete analysis

The analysis in the parent directory trains a **256 → 128** bottleneck. That is an
*undercomplete* autoencoder — it compresses. Overcompleteness is the premise of the
sparse-dictionary / superposition literature (Elhage et al. 2022; Bricken et al. 2023;
Cunningham et al. 2023): you need **more** features than input dimensions for
individual features to specialise. A 256 → 128 bottleneck cannot support that reading
however sparse its code is, so monosemanticity claims should not be made from it.

This directory trains **256 → 1024** (4× overcomplete) with BatchTopK sparsity, which
is the setting those claims require.

| | Parent directory | Here |
|---|---|---|
| Dictionary | 256 → **128** (undercomplete) | 256 → **1024** (4× overcomplete) |
| Architecture | 4-layer MLP autoencoder, dropout 0.1 | single linear encoder + decoder |
| Sparsity | L1 penalty | **BatchTopK**, mean L0 exactly `k` |
| Decoder norm | unconstrained | columns renormalised to unit L2 each step |
| Probes | 3-layer MLP | **linear** (Ridge / LogisticRegression) |
| Controls | none | random-label, descriptor + ECFP4 baseline |
| Seeds | 1 | 5, reported as mean ± std |

---

## Files

| File | Role |
|---|---|
| `overcomplete_sae.py` | The SAE: BatchTopK, unit-norm decoder, `b_pre` = training mean, dead-feature resampling. Run directly to execute self-tests. |
| `properties.py` | Single source of truth for the 11 targets, the descriptor/ECFP4 baseline, and characterisation SMARTS. |
| `load_embeddings.py` | Parses `embeddings.csv` → cached `embeddings.npy` + `smiles.txt`. |
| `run_overcomplete_sae.py` | Trains across seeds; also trains a matched undercomplete baseline on the *identical* split. |
| `probes.py` | Linear probes on raw / latent / reconstruction, with controls. |
| `characterize_features.py` | What each feature fires on: Mann–Whitney (descriptors), Fisher (substructures), Benjamini–Hochberg. |
| `run_all.sh` | Regenerates every table from the raw CSV. |

## Requirements

```
rdkit == 2023.9.5      # QED and descriptors are version-sensitive
torch == 2.1.2
scikit-learn, scipy, pandas, numpy
```
`run_all.sh` warns if RDKit differs from the pinned version.

## Usage

```bash
./run_all.sh /path/to/embeddings.csv results cpu
```

`embeddings.csv` (116 MB) is not distributed in this repository — it exceeds the
per-file size limit. See `../REPRODUCE.md` for how to obtain or regenerate it. All
downstream steps run from the cached `.npy`, so the parse happens once.

## Outputs

| Artifact | Contents |
|---|---|
| `analysis_report.txt` | Reconstruction, sparsity, dead features; overcomplete vs matched undercomplete |
| `metrics_summary.csv` | NMSE, mean L0, dead fraction — mean ± std over seeds |
| `T_probes_summary.csv` | Per target: `trained`, `random_label`, `descriptor`, `gap_vs_descriptor`, `trivially_decodable` |
| `T_feature_characterization.csv` | Per (feature, descriptor/substructure): effect size, p, BH q, enriched flag |
| `T_feature_summary.csv` | Per feature: properties served, mono/polysemantic, activation frequency, signed probe weights |
| `T_monosemanticity.csv` | Monosemantic / polysemantic split |

---

## Reading the results honestly

**`gap_vs_descriptor` is the column that matters.** A probe scoring 1.000 on
molecular size proves only that heavy-atom count is recoverable from a graph
embedding — which it trivially is. The descriptor baseline uses RDKit counts and
ECFP4 with no access to the model at all; where it matches the embedding probe, the
`trivially_decodable` flag is set and no claim about learned representation is
warranted. Only a positive gap is evidence the model contributes something.

**`random_label` must collapse to chance** (0.0 for R², 0.5 for AUROC). If it does
not, the probe is memorising rather than decoding.

**Feature indices are run-specific.** They depend on initialisation, so they will
not match any previously published numbering, and features from different seeds are
not comparable by index. Match dictionaries by decoder-column cosine similarity
instead.

**"Monosemantic" here is a statement about probe weights**, not about semantics: it
means the feature appears in exactly one property's top-|w| list. It is a useful
proxy, not proof of a single underlying concept.

**On this dataset boron is rare (0.3%).** Boron-context features reported elsewhere
came from a different, boron-rich (43.7%) molecule set and should not be expected to
reappear here. That contrast is itself informative: it shows which features are
properties of the model and which are properties of the training distribution.

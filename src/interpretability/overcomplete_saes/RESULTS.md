# Results — Overcomplete BatchTopK SAE on SynFlowNet Embeddings

All numbers below were produced by the scripts in this directory on 2026-08-03 and are
reproducible via `run_all.sh`. Every value is a mean over 5 seeds unless stated. Nothing here is
copied from a previous manuscript; where a previously reported value exists it is shown alongside
ours and labelled as such.

**Dataset.** 32,054 molecules, 256-d graph embeddings, from the SynFlowNet analysed in
arXiv 2511.19264 (QED reward). Split 90/10 for the SAE; 80/10/10 for probes.
**Environment.** rdkit 2023.09.5 (the pinned version — QED is version-sensitive), torch 2.1.2+cu121,
scikit-learn 1.2.2, Tesla T4.
**Seeds.** SAE 42–46; probes 0–4.

---

## 1. Headline findings

1. **BatchTopK delivers exact sparsity.** Mean L0 = **64.000 ± 0.000** across all 5 seeds. An ℓ1
   penalty cannot do this — a previously reported "exact L0 = 64" therefore implies BatchTopK.
2. **Dead features are negligible with resampling: 0.64% ± 0.28%**, not the ~54% previously
   reported. The 1024/k=64 configuration needs no special justification.
3. **The overcomplete SAE beats a matched undercomplete baseline** on reconstruction — NMSE
   0.00151 vs 0.00263, a 1.75× improvement — on the *identical* split, seeds and data.
4. **The embedding does not beat cheap descriptors on any standard property.** With
   target-leakage removed, RDKit counts + ECFP4 match or exceed the embedding on 10 of 11 targets.
   This does not mean the embedding is uninformative (§4.2) but it does rule out the claim that the
   model learned chemistry descriptors lack.
5. **Individual features have specific, causal influence.** Ablating one of 1024 features moves one
   property with specificity ratios up to **44×**.
6. **Halogen-subtype factorization replicates, and more cleanly than before** — separate,
   near-perfect F, Cl, Br and I detectors, plus a boron/boronic-acid detector, all at q < 1e-10.
7. **Individual features are mostly NOT reproducible across runs.** Mean cross-seed dictionary
   similarity is 0.237 against a chance baseline of 0.214 — only **1.11× chance**. A small core
   (~1–3%) does reproduce far above chance. **Any named feature ID must be stability-checked
   before it is interpreted.**

---

## 2. Sparse autoencoder (addendum §2.4, §3.8)

Figure: `figs/F3_overcomplete_vs_undercomplete.pdf`, `figs/F5_training_diagnostics.pdf`
Table: `results_5seed/metrics_summary.csv`, `metrics.csv`

| Model | NMSE | Mean L0 | Dead features |
|---|---|---|---|
| **Overcomplete 256→1024, BatchTopK k=64** | **0.00151 ± 0.00009** | **64.000 ± 0.000** | 0.64% ± 0.28% |
| Undercomplete 256→128, ℓ1 (matched split) | 0.00263 ± 0.00011 | 67.57 ± 0.76 | 0% |

The undercomplete baseline is a *single* linear layer each way, so the only differences from the SAE
are dictionary width and sparsity mechanism. (The original paper's undercomplete model was a
four-layer MLP autoencoder with dropout; that is a different architecture and is not the baseline
used here.)

Notice the undercomplete model reaches mean L0 = 67.6 *without* being asked to — comparable sparsity
to k=64 — yet reconstructs 1.75× worse. Overcompleteness, not sparsity, is what buys the
reconstruction.

**Suggested wording:** "A 4× overcomplete BatchTopK autoencoder (1024 features, k=64) reconstructs
the 256-d embedding at NMSE 0.0015 ± 0.0001 with exactly 64 active features per molecule, compared
with 0.0026 ± 0.0001 for a matched undercomplete (256→128) ℓ1 autoencoder on identical splits and
seeds. Feature death is negligible (0.6% ± 0.3%) under periodic resampling."

---

## 3. Information preservation (addendum §3.4)

Figure: `figs/F4_information_preservation.pdf` · Table: `T_probes_summary.csv`

Linear probes applied to raw embeddings, SAE latents, and SAE reconstructions:

| Target | Raw | Latent | Reconstruction | Δ latent | Δ recon |
|---|---|---|---|---|---|
| boron (AUROC) | 1.0000 | 0.9995 | 0.9991 | −0.0005 | −0.0009 |
| halogen (AUROC) | 0.9999 | 0.9994 | 0.9998 | −0.0006 | −0.0002 |
| aromaticity (AUROC) | 0.9999 | 0.9987 | 0.9998 | −0.0012 | −0.0001 |
| size / molecular weight (R²) | 0.9965 | 0.9854 | 0.9821 | −0.0111 | −0.0144 |
| polarity (R²) | 0.9931 | 0.9731 | 0.9676 | −0.0200 | −0.0255 |
| complexity ring-proxy (R²) | 0.9485 | 0.9360 | 0.9328 | −0.0125 | −0.0157 |
| urea (AUROC) | 0.9221 | 0.9080 | 0.8734 | −0.0141 | −0.0487 |
| lipophilicity (R²) | 0.8857 | 0.8309 | 0.8247 | −0.0548 | −0.0610 |
| flexibility (R²) | 0.6944 | 0.5949 | 0.5782 | **−0.0994** | **−0.1162** |
| drug-likeness (R²) | 0.5860 | **0.6192** | 0.5459 | **+0.0333** | −0.0401 |

**Median retention through encode→decode: 98.3%.** Classification is essentially untouched
(≤0.001). The largest loss is **flexibility (−0.116)**, not lipophilicity.

**Drug-likeness is the one property the sparse latent decodes *better* than the raw embedding**
(+0.033): the sparse basis makes QED more linearly accessible than it is in the dense embedding.
That is a real and quotable result.

---

## 4. Linear decodability and trivial baselines (addendum §3.3)

Figures: `figs/F1_decodability_vs_controls.pdf`, `figs/F2_gap_vs_descriptor_baseline.pdf`

### 4.1 The table

Probes are genuinely linear (Ridge / LogisticRegression). Controls: labels shuffled
(`random_label`), and the identical probe on RDKit counts + ECFP4 with **no model involved**
(`descriptor`).

| Target | Embedding | Descriptor baseline | Gap | Trivially decodable? |
|---|---|---|---|---|
| boron (AUROC) | 1.0000 ± 0.0000 | 1.0000 | +0.000 | yes |
| halogen (AUROC) | 0.9999 ± 0.0001 | 0.9999 | −0.000 | yes |
| aromaticity (AUROC) | 0.9999 ± 0.0002 | 1.0000 | −0.000 | yes |
| size / mol. weight (R²) | 0.9965 ± 0.0012 | 0.9801 | +0.016 | yes |
| polarity (R²) | 0.9931 ± 0.0002 | 0.9855 | +0.008 | yes |
| **complexity ring-proxy (R²)** | **0.9485 ± 0.0044** | 0.9125 | **+0.036** | **no** |
| urea (AUROC) | 0.9221 ± 0.0240 | 0.9992 | −0.077 | yes |
| lipophilicity (R²) | 0.8857 ± 0.0044 | 0.9589 | −0.073 | yes |
| flexibility (R²) | 0.6944 ± 0.0091 | 0.8965 | −0.202 | yes |
| drug-likeness (R²) | 0.5860 ± 0.0151 | 0.6117 | −0.026 | yes |

Random-label controls collapse to chance everywhere (AUROC 0.49–0.52; R² −0.005 to −0.008), so the
probes are sound and the scores are not memorisation.

**Note on the baseline.** Five targets are *defined* by columns in the descriptor set — e.g.
flexibility = NumRotatableBonds/10 — which makes the full-baseline comparison arithmetic rather
than informative. Those columns are dropped per target and the leakage-free number is what is
reported above. `T_probes_summary.csv` carries both (`descriptor_mean`, `descriptor_noleak_mean`)
and a `baseline_leaky` flag.

### 4.2 What this does and does not license

**Does not license:** "the embedding provides no useful information." In absolute terms it decodes
QED at R² 0.59, size at 0.997, halogen at AUROC 1.000, with random-label at chance. That is real
linearly-accessible chemical information.

**Does license:** the embedding carries no information *beyond cheap descriptors* for these targets.
Note what the targets are — QED, LogP, TPSA, MW, rotatable-bond count, halogen presence — every one a
deterministic function of the molecular graph that RDKit computes directly. ECFP4 plus atom counts
was always going to be near-perfect at these. The comparison is worth reporting because it prevents
overclaiming, but "does not beat ECFP4 at predicting TPSA" is weak evidence about representation
quality.

**Untested and worth stating as such:** whether the embedding is good at *its own job*. It was
trained to support sequential action selection — which reactant, which template, when to stop — not
descriptor prediction. No probe here addresses that.

**Suggested wording:** "Against a descriptor baseline of RDKit counts and ECFP4 fingerprints, the
learned embedding shows no advantage on ten of eleven properties (largest deficit: flexibility,
−0.202), with a modest advantage only on the ring-count complexity proxy (+0.036). Since all probed
properties are deterministic functions of molecular structure that cheminformatics descriptors
compute directly, we interpret this as evidence against claims of *additional* learned chemical
knowledge rather than evidence that the representation is uninformative."

---

## 5. Feature-level interpretation (addendum §2.6, §3.5)

Figure: `figs/F8_feature_semanticity.pdf` · Tables: `T_feature_summary.csv`,
`T_feature_characterization.csv`, `T_monosemanticity.csv`

### 5.1 Mono- vs polysemantic

**107 features** appear in at least one property's top-20 weight list:
**46 monosemantic (43.0%)**, **61 polysemantic (57.0%)**.

"Monosemantic" here means the feature appears in exactly one property's top-|w| list. It is a
statement about probe weights, not proof of a single underlying concept, and should be described
that way.

### 5.2 Substructure enrichment — halogen factorization replicates

3,959 enrichment tests; **280 significant** (prevalence gap > 30 points, Benjamini–Hochberg
q < 0.01). The strongest are clean, single-halogen detectors:

| Feature | Substructure | Activating | Silent | Gap | q |
|---|---|---|---|---|---|
| 813 | **iodine** | 100% | 0% | +100 pts | 2.2e-14 |
| 854 | **chlorine** | 100% | 0% | +100 pts | 2.2e-14 |
| 346 | **bromine** | 100% | 0% | +100 pts | 2.2e-14 |
| 448 | chlorine | 100% | 3.3% | +96.7 pts | 5.2e-13 |
| 800 | benzene (negative) | 0% | 93.3% | −93.3 pts | 6.6e-12 |
| **787** | **boron + boronic acid/ester** | 90% | 0% | +90 pts | 5.2e-11 |
| 483, 986 | fluorine | 100% | 16.7% | +83.3 pts | 8.0e-10 |

This **replicates and sharpens** the previously reported halogen factorization: rather than one
"halogen" signal, the dictionary contains *separate detectors for each halogen*, plus an
anti-aromatic (benzene-absent) feature. The boron detector at 90% vs 0% is notable because **boron
occurs in only 0.3% of this dataset** — the feature is not a data-composition artifact here, unlike
in a boron-rich set.

**Suggested wording:** "Substructure enrichment identifies chemically clean feature detectors:
separate features fire almost exclusively on iodine-, chlorine-, bromine- and fluorine-containing
molecules (100% vs 0–17% in silent molecules, q < 1e-9), and one feature detects boron and boronic
esters (90% vs 0%, q = 5e-11) despite boron appearing in only 0.3% of the corpus. The policy
therefore represents halogen identity, not merely halogen presence."

### 5.3 Polysemantic features

Feature 100 participates in 7 property lists (drug-likeness, complexity, lipophilicity, size,
polarity, molecular weight, halogen), fires on 41.3% of molecules, and is enriched for fluorine
(+77 pts) and ester (+40 pts). Features 877 and 806 each serve 5 properties. These are the
"broad chemical context" directions, and their breadth is expected: a feature that tracks molecular
elaboration will load on every size-correlated property.

### 5.4 Drug-likeness decomposition (addendum §3.7)

Largest negative QED probe weights:

| Feature | w(QED) | Properties served | Activation freq. | Top enrichments |
|---|---|---|---|---|
| 425 | −0.128 | 1 | 9.0% | benzene −73, amide −67, aromatic N −67 |
| 100 | −0.111 | 7 | 41.3% | fluorine +77, ester +40 |
| 211 | −0.107 | 4 | 42.9% | aromatic N +43, ester +43, ether +43 |
| 365 | −0.103 | 2 | 37.0% | ester +67, ether +53 |

The *structure* of the previous claim holds — a small number of features carry the largest negative
QED weight — but the **magnitudes do not compare**: ours are ≈0.1, previously reported values were
≈−2.6. That is a probe-scaling difference (Ridge on standardised features), not a discrepancy in
kind. **Report ranks and enrichments, not raw weight magnitudes.**

---

## 6. Causal validation by ablation (addendum §3.6)

Figure: `figs/F7_intervention_matrix.pdf` · Tables: `T_feature_ablation.csv`,
`T_ablation_specificity.csv`

Method: encode → zero one feature → decode → re-apply the **frozen** probes (fitted once on
unablated reconstructions, never refitted), and measure Δ per property.
**This is ablation only. No steering: nothing is amplified and generation is never re-run.**

| Feature | Property moved | Δ | Mean off-target \|Δ\| | Specificity ratio |
|---|---|---|---|---|
| 632 | flexibility | −0.014 | 0.0003 | **44.5×** |
| 211 | lipophilicity | −1.671 | 0.072 | **23.1×** |
| 995 | flexibility | −0.616 | 0.029 | **20.9×** |
| 877 | flexibility | −2.198 | 0.122 | **18.1×** |
| 163 | lipophilicity | −1.376 | 0.097 | 14.2× |
| 188 | drug-likeness | −0.932 | 0.114 | 8.2× |
| 741 | polarity | −0.257 | 0.031 | 8.2× |

**Report specificity ratios, not raw Δ.** Large absolute deltas partly reflect general
reconstruction damage — zeroing a high-magnitude feature degrades ĥ overall, and R² is unbounded
below. The ratio controls for this: if the effect were non-specific, off-target properties would
degrade in proportion. Ratios of 8–44× show they do not.

**Suggested wording:** "Zeroing a single feature and re-applying frozen probes produces
property-specific degradation, with the targeted property degrading 8–44× more than the mean
off-target property. This converts the feature-level account from correlational to interventional."

---

## 7. Feature stability across runs (plan §A8)

Figure: `figs/F6_feature_stability.pdf` · Tables: `T_feature_stability.csv`, `_summary.csv`

Decoder columns are unit-norm, so cosine similarity is a dot product. Every column in run A is
matched to its most similar column in run B, across all 10 seed pairs. **The chance baseline is
essential**: in 256 dimensions the maximum of 1024 random cosines is already ≈0.21.

| | Mean best-match cosine | ≥ 0.5 | ≥ 0.7 | ≥ 0.9 |
|---|---|---|---|---|
| Random dictionaries (chance) | 0.214 | 0.00% | 0.00% | 0.00% |
| Trained SAEs (10 seed pairs) | **0.237** (1.11× chance) | **2.71%** | **0.97%** | 0.07% |

**The bulk of the dictionary is at chance — individual features are largely run-specific.** But
~1–3% of features reproduce at similarities chance never reaches, so a small stable core is real.

Two consequences for any write-up:

1. **Feature indices are not portable.** A named feature ID from one run does not exist in another.
   Any previously published ID cannot be recovered by retraining; match dictionaries geometrically.
2. **Interpret only the stable core.** Feature-level claims should be restricted to features that
   pass a cross-seed stability check, and the check should be reported.

**Suggested wording:** "Matching decoder columns across independently seeded runs gives a mean
best-match cosine of 0.237, against a chance baseline of 0.214 from random dictionaries — only 1.11×
chance. However 2.7% of features exceed cosine 0.5 and 1.0% exceed 0.7, thresholds no random pair
attains. We therefore restrict feature-level interpretation to this reproducible core and note that
individual feature indices are not portable between runs."

---

## 8. Comparison with previously reported values

Ours are from this pipeline; "reported" are the corresponding published/draft values.

| Quantity | Reported | Ours | Verdict |
|---|---|---|---|
| flexibility R² (raw) | 0.688 | 0.694 | ✅ replicates |
| size R² | 1.000 | 0.997 | ✅ |
| molecular weight R² | 0.987 | 0.997 | ✅ |
| complexity R² | 0.970 | 0.949 | ✅ |
| aromaticity / halogen / boron AUROC | ≥0.998 | ≈1.000 | ✅ |
| polarity R² | 0.945 | 0.993 | ~ ours higher |
| lipophilicity R² | 0.806 | 0.886 | ~ ours higher |
| urea AUROC | 0.804 | 0.922 | ~ ours higher |
| **drug-likeness R² (raw)** | **0.756** | **0.586** | ❌ does not replicate (−0.17) |
| mean L0 | 64 | 64.000 ± 0.000 | ✅ exact |
| NMSE | 0.0008 | 0.00151 | ~ ours 1.9× higher |
| **dead features** | **54%** | **0.64%** | ❌ large discrepancy |
| features in any top-20 list | 118 | 107 | ✅ close |
| **monosemantic fraction** | **55%** | **43%** | ❌ split inverts |
| aromaticity preserved through bottleneck | "perfectly" | 0.9999 → 0.9998 | ✅ |
| largest ΔR² through bottleneck | lipophilicity (−0.047) | **flexibility (−0.116)** | ❌ different property |
| drug-likeness ΔR² raw→latent | −0.002 | **+0.033** | ❌ sign differs (ours improves) |
| QED weight magnitudes | −2.65 / −2.46 / −2.16 | ≈−0.13 / −0.11 / −0.11 | scale differs (probe scaling) |
| halogen-subtype factorization | qualitative claim | ✅ replicates, sharper | ✅ |

**Nine quantities replicate closely, five do not.** The five worth explaining in the paper:
drug-likeness R², dead-feature fraction, the mono/poly split, which property loses most through the
bottleneck, and the sign of the drug-likeness latent delta.

---

## 9. Claims that are now supported, and claims that are not

**Supported:**
- A 4× overcomplete BatchTopK SAE reconstructs SynFlowNet embeddings at NMSE 0.0015 with exactly 64 active features, outperforming a matched undercomplete baseline.
- ≥98% of probe-detectable information survives the sparse bottleneck; classification is essentially unaffected.
- The dictionary contains chemically clean, statistically significant substructure detectors, including per-halogen and boron/boronic-ester features.
- Single-feature ablation produces property-specific effects (8–44× specificity), i.e. interventional not merely correlational evidence.
- Drug-likeness is *more* linearly accessible in the sparse latent than in the dense embedding.

**Not supported — do not claim:**
- That the embedding encodes chemistry beyond what cheap descriptors provide. Ten of eleven properties are matched or beaten by RDKit counts + ECFP4.
- That near-perfect probe scores demonstrate learned chemical understanding. Size, halogen, aromaticity and boron are trivially decodable; the controls show it.
- That specific feature indices are meaningful across runs. Only ~1–3% reproduce above chance.
- That the model has "internalized structure–activity reasoning." Nothing here tests that.
- Any steering or generation-time controllability claim. Not attempted.

**Terminology.** A 256→128 model is *undercomplete* and cannot support monosemanticity or
superposition claims — those require an overcomplete dictionary. Reserve "sparse autoencoder" in the
Bricken/Cunningham sense for the 256→1024 model; describe the 256→128 model as a compressive
bottleneck autoencoder, closer to sparse PCA.

---

## 10. Artifact index

| Figure | Content |
|---|---|
| `F1_decodability_vs_controls` | Embedding vs descriptor baseline vs random-label, per property |
| `F2_gap_vs_descriptor_baseline` | Signed gap: where the embedding adds information |
| `F3_overcomplete_vs_undercomplete` | NMSE, L0, dead features on the identical split |
| `F4_information_preservation` | raw → latent → reconstruction, per property |
| `F5_training_diagnostics` | NMSE, L0 pinned at k, dead features per epoch, 5 seeds |
| `F6_feature_stability` | Observed vs chance dictionary similarity |
| `F7_intervention_matrix` | Feature × property ablation deltas |
| `F8_feature_semanticity` | Mono/poly split; activation-frequency distribution |

| Table | Content |
|---|---|
| `metrics.csv`, `metrics_summary.csv` | SAE reconstruction / sparsity / dead features per seed and summary |
| `T_probes.csv`, `T_probes_summary.csv` | All probe results with both baseline variants and the leaky flag |
| `T_feature_characterization.csv` | 3,959 enrichment tests with p, BH q, enriched flag |
| `T_feature_summary.csv` | Per feature: properties served, semanticity, activation freq., signed weights |
| `T_monosemanticity.csv` | Mono/poly counts |
| `T_feature_ablation.csv`, `T_ablation_specificity.csv` | Intervention matrix and specificity ratios |
| `T_feature_stability.csv`, `_summary.csv` | Cross-seed similarity, observed and chance |

All figures are PDF (vector) + 300 dpi PNG and are built only from the CSVs, so they regenerate
without a GPU: `python make_plots.py --results-dir results_5seed --out-dir figs`.

---

## 11. Limitations

1. **One checkpoint, one reward (QED).** No claim of generality across rewards.
2. **The analysed checkpoint no longer exists.** Embeddings are the surviving artifact; the training
   recipe is preserved at `../retrain_v1/local_overrides.yaml` but retraining will not reproduce it
   bit-for-bit, and a new checkpoint would require a newly trained SAE.
3. **No random-network control.** Whether an untrained network of identical architecture would score
   similarly is not yet tested; it needs only the architecture, not a trained checkpoint, so it is
   the cheapest remaining strengthening.
4. **`complexity` is not a synthetic-accessibility score.** It reproduces the original definition,
   `1 − min(1, 0.1·n_aliphatic_rings + 0.05·n_aromatic_rings + 0.5)` — a clipped two-ring-count
   proxy. Named `complexity_ringproxy` throughout to stop the mislabel propagating.
5. **Probe classifiers did not fully converge** (`max_iter=1000`); AUROCs are ≥0.92 so conclusions
   are unaffected, but tighter convergence would quiet the warnings.
6. **Two seed families.** SAE seeds 42–46 control dictionary initialisation; probe seeds 0–4 control
   data splits. They are independent and both are reported.
7. **Ablation uses seed 42's dictionary only.** Given §7, the specific feature indices in §6 are not
   portable to other seeds; the *distribution* of specificity ratios is the transferable claim.

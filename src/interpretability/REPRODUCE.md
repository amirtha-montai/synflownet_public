# Reproducing arXiv 2511.19264v1

**"Interpreting GFlowNets for Drug Discovery: Extracting Actionable Insights for Medicinal Chemistry"**

This file records the exact inputs, versions, and expected outputs for the paper's results, so
every published number can be checked. Scripts live in this directory.

---

## 1. Which dataset the paper uses

The paper uses the **32,054-molecule** set in `results_qed_run/`, split **28,848 / 3,206** (90/10),
with **6 reward signals**. Boron content of this set is **0.3%**.

`results_7669/` is a **separate, later experiment** on a different checkpoint and a different
molecule population (30,676 molecules, **43.7% boron**). Its numbers are not comparable to this
paper's and must not be mixed with them.

---

## 2. Environment

```
rdkit == 2023.9.5      # pin exactly: QED and descriptor values are version-sensitive
torch == 2.1.2
numpy == 1.26.4
python 3.10
```
Pinned in `requirements/main-3.10.txt` at the repository root. Note `pyproject.toml` currently
leaves `rdkit` unpinned — prefer the lockfile.

---

## 3. The trained model

`results_qed_run/config_interpretability_qed.json` references a checkpoint path that is **no longer
available**. The full training recipe is preserved in
[`training_config_qed_arxiv_2511.19264.yaml`](training_config_qed_arxiv_2511.19264.yaml):

```
reward: qed                    seed: 0                num_training_steps: 1000
num_layers: 4                  num_emb: 128           git_hash: d134838
templates_filename:            hb.txt
building_blocks_filename:      enamine_bbs.txt
precomputed_bb_masks_filename: precomputed_bb_masks_enamine_bbs.pkl
illegal_action_logreward: -75.0
```

All three referenced assets are in this repository:
- `src/synflownet/data/templates/hb.txt`
- `src/synflownet/data/building_blocks/enamine_bbs.txt` (4,996 building blocks)
- `src/synflownet/data/building_blocks/precomputed_bb_masks_enamine_bbs.pkl`

Retraining will not bit-match the original (GPU nondeterminism), so treat regenerated embeddings as
a re-derivation rather than a reproduction.

**Two distinct seeds — do not conflate them.** SynFlowNet training used **seed 0** (above). The
SAE and probe splits use `train_test_split(random_state=42)` in `sparse_autoencoder.py`.

---

## 4. Reproducing the analysis without the checkpoint

Everything downstream of the frozen embeddings reproduces exactly. The embeddings matrix
(`results_qed_run/embeddings.csv`, 116 MB, 256-d per molecule) is **not distributed here** because
it exceeds the per-file size limit. Obtain it from the authors, or regenerate it with
`embedding_analysis.py` after retraining per §3.

```bash
python run_sparse_autoencoder.py --config myconfig.json
```

`run_sparse_autoencoder.py` hard-codes an embeddings path near the top of the file — point it at
your local `results_qed_run/embeddings.csv` first.

Config that reproduces the paper's compressive SAE:
```json
{"data": {"embedding_file": "results_qed_run/embeddings.csv", "test_size": 0.1},
 "autoencoder": {"hidden_dim": 128, "epochs": 200, "batch_size": 128,
                 "learning_rate": 0.001, "sparsity_weight": 0.01,
                 "sparsity_target": 0.05, "sparsity_method": "l1", "dropout": 0.1},
 "reward_predictor": {"epochs": 100, "learning_rate": 0.001, "dropout": 0.2},
 "output": {"results_dir": "./out", "save_models": true, "save_plots": true}}
```
Note `prepare_data()` defaults to `test_size=0.2`; the paper used the config value **0.1**.

### Expected output

Compare `out/analysis_report.txt` against the archived
`results_qed_run/sparse_autoencoder_dim_128/analysis_report.txt`:

| Reward signal | Test R² |
|---|---|
| polarity | 0.918 |
| complexity | 0.750 |
| size | 0.711 |
| lipophilicity | 0.664 |
| flexibility | 0.502 |
| drug_likeness | 0.251 |

Also: mean activation sparsity **0.105**; Factor 11 ↔ size **r = 0.757**; Factor 86 ↔ polarity
**−0.570**; Factor 118 ↔ polarity **0.540**.

Motif probes (`motif_probe_trainer.py`) → `results_qed_run/motif_probe_results_summary.csv`:
40 motifs, mean test AUC **0.9521**.

---

## 5. Implementation notes that differ from the paper text

Recorded here so readers are not misled while a corrected version is prepared.

1. **SAE architecture.** The paper describes `256→128→256` with `z = ReLU(Wh+b)`. The
   implementation (`sparse_autoencoder.py`, `SparseAutoencoder.__init__`) is a **4-layer MLP
   autoencoder**: `Linear(256,256) → ReLU → Dropout(0.1) → Linear(256,128) → ReLU` encoding, and
   `Linear(128,256) → ReLU → Dropout(0.1) → Linear(256,256)` decoding.

2. **The reward predictors are not linear.** `RewardPredictor` is a 3-layer MLP
   (`latent → latent/2 → latent/4 → 1`, ReLU, dropout 0.2). All R² values in §4 are MLP-probe
   values, not linear-probe values.

3. **`complexity` is not a synthetic-accessibility score.** `_compute_sa_score` returns
   `min(1.0, NumAliphaticRings×0.1 + NumAromaticRings×0.05 + 0.5)` — a clipped two-ring-count
   proxy, as its docstring states — and `complexity = 1 − sa_score`. The R² of 0.750 measures
   recovery of that proxy.

4. **Near-perfect motif AUROCs lack controls.** Five of forty motifs score ≥ 0.9999
   (`halogen_Cl` = 1.0000). No descriptor, random-label, or random-network baseline was run, so
   these should be read as *possibly trivially decodable from the input* rather than as evidence of
   learned chemical concepts.

---

## 6. File map

| Path | Role |
|---|---|
| `embedding_analysis.py` | extract graph embeddings from a checkpoint |
| `extract_generated_smiles.py` | pull SMILES/trajectories from the run database |
| `sparse_autoencoder.py` | compressive SAE + reward predictors |
| `run_sparse_autoencoder.py` | driver + report generator |
| `motif_probe_trainer.py` | motif classification probes |
| `motif_correlation_analysis.py` | motif–factor correlation analysis |
| `plot_correleation_sparse_features.py`, `plot_reactions.py` | figures |
| `saliency_analysis.py` | Integrated Gradients + counterfactual edits |
| `results_qed_run/` | archived outputs for this paper |
| `training_config_qed_arxiv_2511.19264.yaml` | training recipe for the analysed model |

# SynFlowNet Interpretability Module

This module provides tools for interpreting and analyzing graph embeddings from trained SynFlowNet models.

## Pipeline Workflow

### 1. Embedding Analysis
**File**: `embedding_analysis.py`

Extract graph embeddings from trained SynFlowNet models.

**Usage**:
```bash
python embedding_analysis.py
```

**Outputs**:
- `embeddings.pkl` - Serialized embedding data
- `embeddings.csv` - CSV format embeddings with SMILES

### 2. Motif Probe Trainer
**File**: `motif_probe_trainer.py`

Train classification probes to predict chemical motifs from embeddings.

**Features**:
- 50+ chemical motif patterns (SMARTS)
- Binary classification for motif presence
- AUROC-based evaluation

### 3. Motif Correlation Analysis
**File**: `motif_correlation_analysis.py`

Analyze relationships between chemical motifs in ground truth vs model representations.

**Usage**:
```bash
python motif_correlation_analysis.py
```

**Outputs**:
- `ground_truth_motif_correlations.png` - Ground truth correlation heatmap
- `extracted_motif_correlations.png` - Model-extracted correlation heatmap
- `motif_correlation_differences.png` - Difference analysis heatmap

### 4. Sparse Autoencoder
**File**: `run_sparse_autoencoder.py`

Train sparse autoencoders to discover interpretable chemical factors.

**Usage**:
```bash
python run_sparse_autoencoder.py [--config CONFIG_FILE] [--quick]
```

**Outputs**:
- `sparse_autoencoder_results.pkl` - Full results and trained models
- `analysis_report.txt` - Analysis summary
- Training plots and visualizations

### 5. Plot Correlation
**File**: `plot_corrleation_sparse_features.py`

Generate correlation plots and visualizations for sparse features.

## Quick Start

Run the pipeline in order:

1. **Extract Embeddings**:
   ```bash
   python embedding_analysis.py
   ```

2. **Train Motif Probes**:
   ```bash
   python motif_probe_trainer.py
   ```

3. **Analyze Motif Correlations**:
   ```bash
   python motif_correlation_analysis.py
   ```

4. **Run Sparse Autoencoder**:
   ```bash
   python run_sparse_autoencoder.py --quick
   ```

5. **Generate Plots**:
   ```bash
   python plot_corrleation_sparse_features.py
   ```

## Requirements

Install the synflownet repo and train a model to completion. Use checkpoints here.
The file `unique_smiles.csv` can be extracted with 
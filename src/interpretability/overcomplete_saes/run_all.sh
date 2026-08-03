#!/usr/bin/env bash
# Regenerate every overcomplete-SAE table from the raw embeddings CSV.
#
#   ./run_all.sh <embeddings.csv> [out_dir] [device]
#
# Reproduces the arXiv 2511.19264 dataset (32,054 molecules, 256-d graph
# embeddings). Requires rdkit==2023.9.5 and torch==2.1.2 — QED and descriptor
# values are RDKit-version-sensitive, so a different version will shift numbers.
set -euo pipefail

EMB_CSV="${1:?usage: run_all.sh <embeddings.csv> [out_dir] [device]}"
OUT="${2:-results}"
DEVICE="${3:-cpu}"
SEEDS="${SEEDS:-42 43 44 45 46}"
PROBE_SEEDS="${PROBE_SEEDS:-0 1 2 3 4}"
PY="${PYTHON:-python}"

cd "$(dirname "$0")"
echo "== 0/4  environment"
$PY - <<'EOF'
import rdkit, torch, sklearn, numpy
print(f"   rdkit {rdkit.__version__}  torch {torch.__version__}  sklearn {sklearn.__version__}")
if rdkit.__version__ != "2023.09.5":
    print(f"   WARNING: rdkit is {rdkit.__version__}, not the pinned 2023.09.5 —"
          " QED/descriptor values may differ from the published numbers")
EOF

echo "== 1/4  self-tests"
$PY overcomplete_sae.py

echo "== 2/4  parse embeddings -> cache/"
if [[ ! -f cache/embeddings.npy ]]; then
  $PY load_embeddings.py --embeddings-csv "$EMB_CSV" --out-dir cache
else
  echo "   cache/embeddings.npy exists, skipping"
fi

echo "== 3/4  train SAE over seeds $SEEDS (+ matched undercomplete baseline)"
$PY run_overcomplete_sae.py \
  --cache-dir cache --out-dir "$OUT" \
  --seeds $SEEDS --device "$DEVICE"

echo "== 4/4  probes with controls, then feature characterization"
$PY probes.py \
  --cache-dir cache --latents "$OUT/latents.npz" \
  --out-dir "$OUT" --seeds $PROBE_SEEDS

$PY characterize_features.py \
  --cache-dir cache --latents "$OUT/latents.npz" \
  --probe-weights "$OUT/probe_weights_latent.npz" --out-dir "$OUT"

echo
echo "done. artifacts in $OUT/:"
echo "  analysis_report.txt              SAE reconstruction / sparsity / dead features"
echo "  metrics_summary.csv              overcomplete vs matched undercomplete, mean±std"
echo "  T_probes_summary.csv             probe scores with random-label + descriptor controls"
echo "  T_feature_characterization.csv   per-feature descriptor and substructure tests"
echo "  T_feature_summary.csv            one row per feature, with probe weights"
echo "  T_monosemanticity.csv            monosemantic / polysemantic split"

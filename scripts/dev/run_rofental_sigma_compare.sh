#!/usr/bin/env bash
set -euo pipefail

# Convenience helper: compare uncertainty_layer vs formula runs for Rofental copies.
# Defaults assume the local layout used during development.

REPO=${REPO:-/home/franz/workspace/repos/openamundsen_da}
DATA_ROOT=${DATA_ROOT:-/home/franz/workspace/dev_projects}
PROJECT_UNC=${PROJECT_UNC:-$DATA_ROOT/rofental_uncertainty_layer/projects/project_2022_2023}
PROJECT_FORM=${PROJECT_FORM:-$DATA_ROOT/rofental_formula/projects/project_2022_2023}
OUT_DIR=${OUT_DIR:-$DATA_ROOT/rofental_sigma_compare}
IMAGE=${IMAGE:-openamundsen-da-ci:local}

mkdir -p "$OUT_DIR"

mkdir -p "$OUT_DIR"

docker run --rm \
  -v "$REPO":/workspace \
  -v "$DATA_ROOT":/data \
  -w /workspace \
  "$IMAGE" \
  python /workspace/scripts/dev/compare_sigma_modes.py \
    --project-uncertainty "${PROJECT_UNC/$DATA_ROOT/\/data}" \
    --project-formula "${PROJECT_FORM/$DATA_ROOT/\/data}" \
    --output-csv "${OUT_DIR/$DATA_ROOT/\/data}/sigma_ess_comparison.csv" \
    --output-png "${OUT_DIR/$DATA_ROOT/\/data}/sigma_ess_comparison.png"

echo "Wrote: $OUT_DIR/sigma_ess_comparison.csv"
echo "Wrote: $OUT_DIR/sigma_ess_comparison.png"

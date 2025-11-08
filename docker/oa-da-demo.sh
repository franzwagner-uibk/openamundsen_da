#!/usr/bin/env bash
set -euo pipefail

# Simple end-to-end demo inside the container. Assumes the repo is mounted at
# /workspace and the example project is mounted at /data.

REPO=${REPO:-/workspace}
DATA=${DATA:-/data}
STEP=${STEP:-$DATA/propagation/season_2017-2018/step_00_init}
AOI=${AOI:-$DATA/env/GMBA_Inventory_L8_15422.gpkg}
DATE=${DATE:-2018-01-10}

echo "[demo] Installing repo in editable mode: $REPO"
pip install -e "$REPO" --no-deps

echo "[demo] Assimilating SCF for $DATE"
python -m openamundsen_da.methods.pf.assimilate_scf \
  --project-dir "$DATA" \
  --step-dir    "$STEP" \
  --ensemble    prior \
  --date        "$DATE" \
  --aoi         "$AOI"

datestr=${DATE//-/}

echo "[demo] Plotting weights (SVG backend)"
python -m openamundsen_da.methods.pf.plot_weights \
  "$STEP/assim/weights_scf_${datestr}.csv" \
  --output "$STEP/assim/weights_scf_${datestr}.svg" \
  --backend SVG

echo "[demo] Plotting SCF summary (SVG backend)"
python -m openamundsen_da.observer.plot_scf_summary \
  "$DATA/obs/season_2017-2018/scf_summary.csv" \
  --output "$DATA/obs/season_2017-2018/scf_summary.svg" \
  --backend SVG

echo "[demo] Done. Outputs in: $STEP/assim and $DATA/obs/season_2017-2018"


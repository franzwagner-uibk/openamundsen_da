#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_IMAGE="${CI_IMAGE:-openamundsen-da-ci:local}"
MAX_WORKERS="${OA_DA_TEST_MAX_WORKERS:-4}"
ARTIFACT_DIR="${CI_ARTIFACT_DIR:-}"

TMP_ROOT="$(mktemp -d -t oada-ci-XXXXXX)"
PROJECT_DIR="${TMP_ROOT}/rofental_ci"
SEASON_NAME="season_ci_2022_2023"
SEASON_DIR="/data/propagation/${SEASON_NAME}"
HOST_LOG_FILE="${PROJECT_DIR}/ci_integration.log"
CONTAINER_LOG_FILE="/data/ci_integration.log"

cleanup() {
  local rc=$?
  if [[ -n "${ARTIFACT_DIR}" ]]; then
    mkdir -p "${ARTIFACT_DIR}"
    if [[ -f "${HOST_LOG_FILE}" ]]; then
      cp -f "${HOST_LOG_FILE}" "${ARTIFACT_DIR}/ci_integration.log"
    fi
    if [[ -d "${PROJECT_DIR}/propagation/${SEASON_NAME}" ]]; then
      mkdir -p "${ARTIFACT_DIR}/propagation"
      cp -a "${PROJECT_DIR}/propagation/${SEASON_NAME}" "${ARTIFACT_DIR}/propagation/"
    fi
  fi
  rm -rf "${TMP_ROOT}"
  trap - EXIT
  exit "${rc}"
}
trap cleanup EXIT

cp -a "${ROOT_DIR}/examples/rofental" "${PROJECT_DIR}"
touch "${HOST_LOG_FILE}"
exec > >(tee -a "${HOST_LOG_FILE}") 2>&1

compose_run() {
  REPO="${ROOT_DIR}" \
  PROJ="${PROJECT_DIR}" \
  IMAGE="${CI_IMAGE}" \
  env UID="$(id -u)" GID="$(id -g)" \
  docker compose run --rm oa "$@"
}

echo "[integration] Preparing trimmed CI season in ${PROJECT_DIR}"

compose_run python - <<'PY'
from pathlib import Path
import yaml

project_dir = Path("/data")
project_yml = project_dir / "project.yml"
with project_yml.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

da = cfg.setdefault("data_assimilation", {})
prior = da.setdefault("prior_forcing", {})
prior["ensemble_size"] = 4

with project_yml.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

season_dir = project_dir / "propagation" / "season_ci_2022_2023"
season_dir.mkdir(parents=True, exist_ok=True)
season_cfg = {
    "start_date": "2023-03-12",
    "end_date": "2023-03-27 21:00:00",
    "data_assimilation": {
        "assimilation_events": [
            {
                "date": "2023-03-17",
                "variable": "scf",
                "product": "SNOWCOVER",
            },
            {
                "date": "2023-03-24",
                "variable": "wet_snow",
                "product": "S1",
            },
        ]
    },
}
with (season_dir / "season.yml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(season_cfg, f, sort_keys=False)
PY

compose_run python -m openamundsen_da.pipeline.season_skeleton \
  --project-dir /data \
  --season-dir "${SEASON_DIR}" \
  --overwrite \
  --log-level INFO

compose_run oa-da-scf \
  --season-dir "${SEASON_DIR}" \
  --summary-csv /data/obs/season_2022_2023/scf_summary.csv \
  --overwrite \
  --log-level INFO

compose_run oa-da-wetsnow-season \
  --season-dir "${SEASON_DIR}" \
  --summary-csv /data/obs/season_2022_2023/wet_snow_summary.csv \
  --overwrite \
  --log-level INFO

echo "[integration] Running season pipeline (max-workers=${MAX_WORKERS})"
compose_run python -m openamundsen_da.pipeline.season \
  --project-dir /data \
  --season-dir "${SEASON_DIR}" \
  --max-workers "${MAX_WORKERS}" \
  --overwrite \
  --log-level INFO

compose_run python /workspace/scripts/ci/validate_trimmed_season.py \
  --season-dir "${SEASON_DIR}" \
  --log-file "${CONTAINER_LOG_FILE}"

echo "[integration] PASS"

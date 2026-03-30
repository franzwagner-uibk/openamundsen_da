#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_IMAGE="${CI_IMAGE:-openamundsen-da-ci:local}"
MAX_WORKERS="${OA_DA_TEST_MAX_WORKERS:-4}"
ARTIFACT_DIR="${CI_ARTIFACT_DIR:-}"

TMP_ROOT="$(mktemp -d -t oada-ci-XXXXXX)"
PROJECT_DIR="${TMP_ROOT}/rofental_ci"
PROJECT_NAME=""
PROJECT_PATH=""
SOURCE_PROJECT_NAME=""
HOST_LOG_FILE="${PROJECT_DIR}/ci_integration.log"
REPO_MOUNT="${ROOT_DIR}"
PROJ_MOUNT="${PROJECT_DIR}"

# Prevent Git-Bash path mangling of container-style arguments like /data or /workspace.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    export MSYS_NO_PATHCONV=1
    export MSYS2_ARG_CONV_EXCL="*"
    REPO_MOUNT="$(cygpath -w "${ROOT_DIR}")"
    PROJ_MOUNT="$(cygpath -w "${PROJECT_DIR}")"
    ;;
esac

cleanup() {
  local rc=$?
  if [[ -n "${ARTIFACT_DIR}" ]]; then
    mkdir -p "${ARTIFACT_DIR}"
    if [[ -f "${HOST_LOG_FILE}" ]]; then
      cp -f "${HOST_LOG_FILE}" "${ARTIFACT_DIR}/ci_integration.log"
    fi
    if [[ -n "${PROJECT_NAME}" && -d "${PROJECT_DIR}/projects/${PROJECT_NAME}" ]]; then
      mkdir -p "${ARTIFACT_DIR}/projects"
      cp -a "${PROJECT_DIR}/projects/${PROJECT_NAME}" "${ARTIFACT_DIR}/projects/"
    fi
  fi
  rm -rf "${TMP_ROOT}"
  trap - EXIT
  exit "${rc}"
}
trap cleanup EXIT

cp -a "${ROOT_DIR}/examples/rofental" "${PROJECT_DIR}"
SOURCE_PROJECT_DIR="$(find "${PROJECT_DIR}/projects" -mindepth 1 -maxdepth 1 -type d -name 'project_*' | sort | head -n 1)"
if [[ -z "${SOURCE_PROJECT_DIR}" ]]; then
  echo "[integration] ERROR: could not discover source project under ${PROJECT_DIR}/projects"
  exit 1
fi
SOURCE_PROJECT_NAME="$(basename "${SOURCE_PROJECT_DIR}")"
PROJECT_NAME="${SOURCE_PROJECT_NAME}"
PROJECT_PATH="/data/projects/${PROJECT_NAME}"

touch "${HOST_LOG_FILE}"
exec > >(tee -a "${HOST_LOG_FILE}") 2>&1

compose_run() {
  REPO="${REPO_MOUNT}" \
  PROJ="${PROJ_MOUNT}" \
  IMAGE="${CI_IMAGE}" \
  env UID="$(id -u)" GID="$(id -g)" \
  docker compose run --rm oa "$@"
}

summary_host_source() {
  local filename="$1"
  local candidates=(
    "${PROJECT_DIR}/obs/summaries/${SOURCE_PROJECT_NAME}/${filename}"
    "${PROJECT_DIR}/obs/${SOURCE_PROJECT_NAME}/${filename}"
    "${PROJECT_DIR}/obs/summaries/all_data/${filename}"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

container_path_for_host() {
  local host_path="$1"
  printf '/data/%s\n' "${host_path#"${PROJECT_DIR}/"}"
}

uncertainty_companions_missing() {
  local obs_key="$1"
  local source_project_name="$2"
  compose_run python - "${obs_key}" "${source_project_name}" <<'PY'
from pathlib import Path
import sys
import yaml

obs_key = str(sys.argv[1]).strip()
source_project = str(sys.argv[2]).strip()
project_yml = Path("/data/projects") / source_project / f"{source_project}.yml"
with project_yml.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

unc_root = ((cfg.get("data_assimilation") or {}).get("uncertainty") or {})
if obs_key == "scf":
    block = (unc_root.get("scf") or {})
    obs_dir = (((cfg.get("obs") or {}).get("snowcover") or {}).get("dir") or "obs/snowcover")
elif obs_key == "wet_snow":
    block = (unc_root.get("wet_snow") or {})
    obs_dir = (((cfg.get("obs") or {}).get("wetsnow") or {}).get("dir") or "obs/wetsnow")
else:
    raise SystemExit(2)

enabled = bool(block.get("enabled", False))
if not enabled:
    raise SystemExit(1)

obs_path = Path("/data") / str(obs_dir)
if not obs_path.exists():
    raise SystemExit(1)

tifs = sorted(obs_path.glob("*.tif")) + sorted(obs_path.glob("*.tiff"))
tifs = [p for p in tifs if not p.stem.lower().endswith("_uncertainty")]
if not tifs:
    raise SystemExit(1)

for src in tifs:
    unc = src.parent / f"{src.stem}_uncertainty.tif"
    if not unc.exists():
        raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "[integration] Preparing full example project in ${PROJECT_DIR}"

if ! summary_host_source "scf_summary.csv" >/dev/null; then
  echo "[integration] SCF summary missing in example; generating from raw rasters"
  if uncertainty_companions_missing "scf" "${SOURCE_PROJECT_NAME}"; then
    echo "[integration] SCF uncertainty enabled and companion rasters missing; generating companions first"
    compose_run python -m openamundsen_da.observer.scf_uncertainty \
      --setup-dir /data \
      --project-label "${SOURCE_PROJECT_NAME}" \
      --overwrite
  fi
  compose_run oa-da-snowcover \
    --input-dir /data/obs/snowcover \
    --project-label "${SOURCE_PROJECT_NAME}" \
    --setup-dir /data \
    --overwrite
fi

if ! summary_host_source "wet_snow_summary.csv" >/dev/null; then
  echo "[integration] Wet-snow summary missing in example; generating from raw rasters"
  if uncertainty_companions_missing "wet_snow" "${SOURCE_PROJECT_NAME}"; then
    echo "[integration] Wet-snow uncertainty enabled and companion rasters missing; generating companions first"
    compose_run python -m openamundsen_da.observer.wetsnow_uncertainty \
      --setup-dir /data \
      --project-label "${SOURCE_PROJECT_NAME}" \
      --overwrite
  fi
  compose_run oa-da-wetsnow \
    --input-dir /data/obs/wetsnow \
    --project-label "${SOURCE_PROJECT_NAME}" \
    --setup-dir /data \
    --overwrite
fi

STATION_METADATA_HOST="${PROJECT_DIR}/obs/stations/stations_da_metadata.csv"
if [[ ! -f "${STATION_METADATA_HOST}" ]]; then
  echo "[integration] ERROR: station metadata CSV is required for the promoted example:"
  echo "  - ${STATION_METADATA_HOST}"
  exit 1
fi

SCF_SUMMARY_HOST_SOURCE="$(summary_host_source "scf_summary.csv" || true)"
WET_SUMMARY_HOST_SOURCE="$(summary_host_source "wet_snow_summary.csv" || true)"
if [[ -z "${SCF_SUMMARY_HOST_SOURCE}" || -z "${WET_SUMMARY_HOST_SOURCE}" ]]; then
  echo "[integration] ERROR: expected summary CSVs were not found after preprocessing fallback"
  exit 1
fi

SCF_SUMMARY_CSV="$(container_path_for_host "${SCF_SUMMARY_HOST_SOURCE}")"
WET_SUMMARY_CSV="$(container_path_for_host "${WET_SUMMARY_HOST_SOURCE}")"
echo "[integration] Using SCF summary: ${SCF_SUMMARY_CSV}"
echo "[integration] Using wet-snow summary: ${WET_SUMMARY_CSV}"

compose_run python -m openamundsen_da.pipeline.project_skeleton \
  --setup-dir /data \
  --project-dir "${PROJECT_PATH}" \
  --overwrite \
  --log-level INFO

compose_run oa-da-scf \
  --project-dir "${PROJECT_PATH}" \
  --summary-csv "${SCF_SUMMARY_CSV}" \
  --overwrite \
  --log-level INFO

compose_run oa-da-wetsnow-project \
  --project-dir "${PROJECT_PATH}" \
  --summary-csv "${WET_SUMMARY_CSV}" \
  --overwrite \
  --log-level INFO

echo "[integration] Running project pipeline (max-workers=${MAX_WORKERS})"
compose_run python -m openamundsen_da.pipeline.project \
  --setup-dir /data \
  --project-dir "${PROJECT_PATH}" \
  --max-workers "${MAX_WORKERS}" \
  --overwrite \
  --log-level INFO

CONTAINER_LOG_FILE="$(compose_run python - "${PROJECT_NAME}" <<'PY' | tr -d '\r' | tail -n 1
from pathlib import Path
import sys

project_name = str(sys.argv[1]).strip()
project_dir = Path("/data/projects") / project_name
if not project_dir.is_dir():
    raise SystemExit(f"Missing project directory: {project_dir}")
candidates = sorted(project_dir.glob(f"{project_name}.log"))
if not candidates:
    candidates = sorted(project_dir.glob("project_*.log"))
if not candidates:
    candidates = sorted(project_dir.glob("*.log"))
if not candidates:
    raise SystemExit(f"No project log found under {project_dir}")
print(candidates[-1].as_posix())
PY
)"
echo "[integration] Using project log: ${CONTAINER_LOG_FILE}"

compose_run python /workspace/scripts/ci/validate_trimmed_project.py \
  --project-dir "${PROJECT_PATH}" \
  --log-file "${CONTAINER_LOG_FILE}"

echo "[integration] PASS"

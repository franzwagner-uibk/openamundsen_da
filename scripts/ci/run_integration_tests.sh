#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_IMAGE="${CI_IMAGE:-openamundsen-da-ci:local}"
MAX_WORKERS="${OA_DA_TEST_MAX_WORKERS:-4}"
ARTIFACT_DIR="${CI_ARTIFACT_DIR:-}"

TMP_ROOT="$(mktemp -d -t oada-ci-XXXXXX)"
PROJECT_DIR="${TMP_ROOT}/rofental_ci"
PROJECT_NAME="project_ci_2022_2023"
PROJECT_PATH="/data/projects/${PROJECT_NAME}"
HOST_LOG_FILE="${PROJECT_DIR}/ci_integration.log"
CONTAINER_LOG_FILE="/data/ci_integration.log"
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
    if [[ -d "${PROJECT_DIR}/projects/${PROJECT_NAME}" ]]; then
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
touch "${HOST_LOG_FILE}"
exec > >(tee -a "${HOST_LOG_FILE}") 2>&1

compose_run() {
  REPO="${REPO_MOUNT}" \
  PROJ="${PROJ_MOUNT}" \
  IMAGE="${CI_IMAGE}" \
  env UID="$(id -u)" GID="$(id -g)" \
  docker compose run --rm oa "$@"
}

echo "[integration] Preparing trimmed CI project in ${PROJECT_DIR}"

compose_run python - <<'PY'
from pathlib import Path
import yaml

setup_dir = Path("/data")
source_project_yml = setup_dir / "projects" / "project_2022_2023" / "project_2022_2023.yml"
with source_project_yml.open("r", encoding="utf-8") as f:
    source_project_cfg = yaml.safe_load(f) or {}

project_dir = setup_dir / "projects" / "project_ci_2022_2023"
project_dir.mkdir(parents=True, exist_ok=True)
project_cfg = dict(source_project_cfg)
project_cfg["start_date"] = "2023-03-12"
project_cfg["end_date"] = "2023-03-27 21:00:00"

da_cfg = dict(project_cfg.get("data_assimilation") or {})
prior_cfg = dict(da_cfg.get("prior_forcing") or {})
prior_cfg["ensemble_size"] = 4
da_cfg["prior_forcing"] = prior_cfg
da_cfg["assimilation_events"] = [
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
project_cfg["data_assimilation"] = da_cfg

with (project_dir / "project_ci_2022_2023.yml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(project_cfg, f, sort_keys=False)
PY

compose_run python -m openamundsen_da.pipeline.project_skeleton \
  --setup-dir /data \
  --project-dir "${PROJECT_PATH}" \
  --overwrite \
  --log-level INFO

compose_run oa-da-scf \
  --project-dir "${PROJECT_PATH}" \
  --summary-csv /data/obs/project_2022_2023/scf_summary.csv \
  --overwrite \
  --log-level INFO

compose_run oa-da-wetsnow-project \
  --project-dir "${PROJECT_PATH}" \
  --summary-csv /data/obs/project_2022_2023/wet_snow_summary.csv \
  --overwrite \
  --log-level INFO

echo "[integration] Running project pipeline (max-workers=${MAX_WORKERS})"
compose_run python -m openamundsen_da.pipeline.project \
  --setup-dir /data \
  --project-dir "${PROJECT_PATH}" \
  --max-workers "${MAX_WORKERS}" \
  --overwrite \
  --log-level INFO

compose_run python /workspace/scripts/ci/validate_trimmed_project.py \
  --project-dir "${PROJECT_PATH}" \
  --log-file "${CONTAINER_LOG_FILE}"

echo "[integration] PASS"



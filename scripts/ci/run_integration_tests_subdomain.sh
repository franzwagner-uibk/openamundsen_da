#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_IMAGE="${CI_IMAGE:-openamundsen-da-ci:local}"
MAX_WORKERS="${OA_DA_SUBDOMAIN_TEST_MAX_WORKERS:-3}"
INNER_WORKERS="${OA_DA_SUBDOMAIN_TEST_INNER_WORKERS:-2}"
ARTIFACT_DIR="${CI_ARTIFACT_DIR:-}"

TMP_ROOT="$(mktemp -d -t oada-subdomain-ci-XXXXXX)"
BASE_SETUP_DIR="${TMP_ROOT}/rofental"
SETUP_DIR="${TMP_ROOT}/rofental_subdomains"
SETUP_PATH="/data/rofental_subdomains"
PROJECT_NAME="project_ci_2022_2023"
PROJECT_PATH="${SETUP_PATH}/projects/${PROJECT_NAME}"
PROJECT_HOST_DIR="${SETUP_DIR}/projects/${PROJECT_NAME}"
HOST_LOG_FILE="${SETUP_DIR}/ci_integration_subdomain.log"
CONTAINER_LOG_FILE="${PROJECT_PATH}/subdomain_run.log"
REPO_MOUNT="${ROOT_DIR}"
PROJ_MOUNT="${TMP_ROOT}"

# Prevent Git-Bash path mangling of container-style arguments like /data or /workspace.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    export MSYS_NO_PATHCONV=1
    export MSYS2_ARG_CONV_EXCL="*"
    REPO_MOUNT="$(cygpath -w "${ROOT_DIR}")"
    PROJ_MOUNT="$(cygpath -w "${TMP_ROOT}")"
    ;;
esac

cleanup() {
  local rc=$?
  if [[ -n "${ARTIFACT_DIR}" ]]; then
    mkdir -p "${ARTIFACT_DIR}"
    if [[ -f "${HOST_LOG_FILE}" ]]; then
      cp -f "${HOST_LOG_FILE}" "${ARTIFACT_DIR}/ci_integration_subdomain.log" 2>/dev/null || true
    fi
    if [[ -d "${PROJECT_HOST_DIR}/subdomains" ]]; then
      mkdir -p "${ARTIFACT_DIR}/projects/${PROJECT_NAME}"
      cp -aL "${PROJECT_HOST_DIR}/subdomains" "${ARTIFACT_DIR}/projects/${PROJECT_NAME}/" 2>/dev/null || true
    fi
    if [[ -d "${SETUP_DIR}/projects/${PROJECT_NAME}" ]]; then
      mkdir -p "${ARTIFACT_DIR}/projects"
      cp -aL "${SETUP_DIR}/projects/${PROJECT_NAME}" "${ARTIFACT_DIR}/projects/" 2>/dev/null || true
    fi
  fi
  rm -rf "${TMP_ROOT}"
  trap - EXIT
  exit "${rc}"
}
trap cleanup EXIT

cp -a "${ROOT_DIR}/examples/rofental" "${BASE_SETUP_DIR}"
cp -a "${ROOT_DIR}/examples/rofental_subdomains" "${SETUP_DIR}"
touch "${HOST_LOG_FILE}"
exec > >(tee -a "${HOST_LOG_FILE}") 2>&1

compose_run() {
  REPO="${REPO_MOUNT}" \
  PROJ="${PROJ_MOUNT}" \
  IMAGE="${CI_IMAGE}" \
  env UID="$(id -u)" GID="$(id -g)" \
  docker compose run --rm oa "$@"
}

echo "[subdomain-integration] Preparing trimmed CI project in ${SETUP_DIR}"

compose_run python - <<'PY'
from pathlib import Path
import yaml

setup_dir = Path("/data/rofental_subdomains")
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
prior_cfg["ensemble_size"] = 3
da_cfg["prior_forcing"] = prior_cfg
da_cfg["assimilation_events"] = [
    {
        "date": "2023-03-17",
        "variable": "scf",
        "product": "SNOWCOVER",
    },
]
project_cfg["data_assimilation"] = da_cfg

with (project_dir / "project_ci_2022_2023.yml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(project_cfg, f, sort_keys=False)
PY

echo "[subdomain-integration] Running sub-domain pipeline (max-workers=${MAX_WORKERS}, inner=${INNER_WORKERS})"
compose_run python -m openamundsen_da.subdomain.cli pipeline \
  --setup-dir "${SETUP_PATH}" \
  --project-dir "${PROJECT_PATH}" \
  --regions "${SETUP_PATH}/env/subdomains.gpkg" \
  --max-workers "${MAX_WORKERS}" \
  --inner-max-workers "${INNER_WORKERS}" \
  --overwrite \
  --log-level INFO

compose_run python /workspace/scripts/ci/validate_trimmed_subdomain.py \
  --subdomain-root "${PROJECT_PATH}/subdomains" \
  --log-file "${CONTAINER_LOG_FILE}"

echo "[subdomain-integration] PASS"

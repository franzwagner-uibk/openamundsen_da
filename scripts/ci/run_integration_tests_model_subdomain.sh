#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_IMAGE="${CI_IMAGE:-openamundsen-da-ci:local}"
MAX_WORKERS="${OA_DA_MODEL_SUBDOMAIN_TEST_MAX_WORKERS:-3}"
ARTIFACT_DIR="${CI_ARTIFACT_DIR:-}"

TMP_ROOT="$(mktemp -d -t oada-model-subdomain-ci-XXXXXX)"
SETUP_DIR="${TMP_ROOT}/subdomains"
SETUP_PATH="/data/subdomains"
MODEL_ROOT_HOST="${SETUP_DIR}/subdomains/model"
MODEL_ROOT_PATH="${SETUP_PATH}/subdomains/model"
HOST_LOG_FILE="${SETUP_DIR}/ci_integration_model_subdomain.log"
CONTAINER_LOG_FILE="${SETUP_PATH}/subdomains/model_pipeline.log"
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
      cp -f "${HOST_LOG_FILE}" "${ARTIFACT_DIR}/ci_integration_model_subdomain.log" 2>/dev/null || true
    fi
    if [[ -d "${MODEL_ROOT_HOST}" ]]; then
      mkdir -p "${ARTIFACT_DIR}/subdomains"
      cp -aL "${MODEL_ROOT_HOST}" "${ARTIFACT_DIR}/subdomains/model" 2>/dev/null || true
    fi
  fi
  rm -rf "${TMP_ROOT}"
  trap - EXIT
  exit "${rc}"
}
trap cleanup EXIT

cp -a "${ROOT_DIR}/examples/subdomains" "${SETUP_DIR}"
touch "${HOST_LOG_FILE}"
exec > >(tee -a "${HOST_LOG_FILE}") 2>&1

compose_run() {
  REPO="${REPO_MOUNT}" \
  PROJ="${PROJ_MOUNT}" \
  IMAGE="${CI_IMAGE}" \
  env UID="$(id -u)" GID="$(id -g)" \
  docker compose run --rm oa "$@"
}

echo "[model-subdomain-integration] Preparing trimmed setup in ${SETUP_DIR}"

compose_run python - <<'PY'
from pathlib import Path
import yaml

setup_yml = Path("/data/subdomains/subdomains.yml")
with setup_yml.open("r", encoding="utf-8-sig") as f:
    cfg = yaml.safe_load(f) or {}

cfg["start_date"] = "2023-03-17"
cfg["end_date"] = "2023-03-18 21:00:00"

with setup_yml.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

echo "[model-subdomain-integration] Running model sub-domain pipeline (max-workers=${MAX_WORKERS})"
compose_run python -m openamundsen_da.subdomain.cli model-pipeline \
  --setup-dir "${SETUP_PATH}" \
  --regions "${SETUP_PATH}/env/subdomains.gpkg" \
  --station-buffer-km 10 \
  --max-workers "${MAX_WORKERS}" \
  --overwrite \
  --log-level INFO

compose_run python /workspace/scripts/ci/validate_trimmed_model_subdomain.py \
  --subdomain-root "${MODEL_ROOT_PATH}" \
  --log-file "${CONTAINER_LOG_FILE}"

echo "[model-subdomain-integration] PASS"

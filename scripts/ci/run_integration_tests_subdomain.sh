#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_IMAGE="${CI_IMAGE:-openamundsen-da-ci:local}"
MAX_WORKERS="${OA_DA_SUBDOMAIN_TEST_MAX_WORKERS:-3}"
INNER_WORKERS="${OA_DA_SUBDOMAIN_TEST_INNER_WORKERS:-2}"
ARTIFACT_DIR="${CI_ARTIFACT_DIR:-}"

TMP_ROOT="$(mktemp -d -t oada-subdomain-ci-XXXXXX)"
SETUP_DIR="${TMP_ROOT}/subdomains"
SETUP_PATH="/data/subdomains"
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

echo "[subdomain-integration] Preparing trimmed CI project in ${SETUP_DIR}"

compose_run python - <<'PY'
from pathlib import Path
from datetime import datetime, timedelta
import re
import yaml

setup_dir = Path("/data/subdomains")
source_project_yml = setup_dir / "projects" / "project_2022_2023" / "project_2022_2023.yml"
with source_project_yml.open("r", encoding="utf-8") as f:
    source_project_cfg = yaml.safe_load(f) or {}

project_dir = setup_dir / "projects" / "project_ci_2022_2023"
project_dir.mkdir(parents=True, exist_ok=True)
project_cfg = dict(source_project_cfg)

da_cfg = dict(project_cfg.get("data_assimilation") or {})
source_events = da_cfg.get("assimilation_events") or []
scf_events = [
    dict(event)
    for event in source_events
    if isinstance(event, dict) and str(event.get("variable", "")).strip().lower() == "scf"
]
station_events = [
    dict(event)
    for event in source_events
    if isinstance(event, dict) and str(event.get("variable", "")).strip().lower() == "station_hs"
]
if not scf_events:
    raise RuntimeError("No SCF assimilation event found in shipped sub-domain example")
if not station_events:
    raise RuntimeError("No station_hs assimilation event found in shipped sub-domain example")

def event_day(event):
    return datetime.fromisoformat(str(event["date"])).date()

station_event = sorted(station_events, key=event_day)[len(station_events) // 2]
station_day = event_day(station_event)
scf_event = min(scf_events, key=lambda event: abs((event_day(event) - station_day).days))
scf_day = event_day(scf_event)
selected_events = sorted([station_event, scf_event], key=lambda event: (str(event["date"]), str(event["variable"])))
start_day = min(station_day, scf_day) - timedelta(days=5)
end_day = max(station_day, scf_day) + timedelta(days=10)
project_cfg["start_date"] = start_day.isoformat()
project_cfg["end_date"] = f"{end_day.isoformat()} 21:00:00"

prior_cfg = dict(da_cfg.get("prior_forcing") or {})
prior_cfg["ensemble_size"] = 3
da_cfg["prior_forcing"] = prior_cfg
da_cfg["assimilation_events"] = selected_events
project_cfg["data_assimilation"] = da_cfg

with (project_dir / "project_ci_2022_2023.yml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(project_cfg, f, sort_keys=False)

source_maps_cfg = source_project_yml.parent / "maps.yml"
if source_maps_cfg.is_file():
    with source_maps_cfg.open("r", encoding="utf-8") as f:
        maps_cfg = yaml.safe_load(f) or {}

    event_date = str(scf_event["date"])
    event_title_date = event_date.replace("-", "/")
    rewritten_maps = {}
    for name, spec in (maps_cfg.get("maps") or {}).items():
        spec = dict(spec or {})
        defaults = dict(spec.get("defaults") or {})
        defaults["date"] = event_date
        spec["defaults"] = defaults

        title = spec.get("title")
        if isinstance(title, str):
            spec["title"] = re.sub(r"\d{4}[/-]\d{2}[/-]\d{2}", event_title_date, title)

        rewritten_name = re.sub(r"\d{4}-\d{2}-\d{2}", event_date, str(name))
        rewritten_maps[rewritten_name] = spec

    maps_cfg["maps"] = rewritten_maps
    with (project_dir / "maps.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(maps_cfg, f, sort_keys=False)
PY

echo "[subdomain-integration] Running sub-domain pipeline (max-workers=${MAX_WORKERS}, inner=${INNER_WORKERS})"
compose_run python -m openamundsen_da.subdomain.cli pipeline \
  --setup-dir "${SETUP_PATH}" \
  --project-dir "${PROJECT_PATH}" \
  --regions "${SETUP_PATH}/env/subdomains.gpkg" \
  --station-buffer-km 10 \
  --max-workers "${MAX_WORKERS}" \
  --inner-max-workers "${INNER_WORKERS}" \
  --overwrite \
  --log-level INFO

compose_run python /workspace/scripts/ci/validate_trimmed_subdomain.py \
  --subdomain-root "${PROJECT_PATH}/subdomains" \
  --log-file "${CONTAINER_LOG_FILE}"

echo "[subdomain-integration] PASS"

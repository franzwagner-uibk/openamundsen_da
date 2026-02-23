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
SOURCE_PROJECT_DIR="$(find "${PROJECT_DIR}/projects" -mindepth 1 -maxdepth 1 -type d -name 'project_*' | sort | head -n 1)"
if [[ -z "${SOURCE_PROJECT_DIR}" ]]; then
  echo "[integration] ERROR: could not discover source project under ${PROJECT_DIR}/projects"
  exit 1
fi
SOURCE_PROJECT_NAME="$(basename "${SOURCE_PROJECT_DIR}")"
PROJECT_NAME="project_ci_${SOURCE_PROJECT_NAME#project_}"
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

echo "[integration] Preparing trimmed CI project in ${PROJECT_DIR}"

SCF_SUMMARY_SOURCE_NEW="${PROJECT_DIR}/obs/${SOURCE_PROJECT_NAME}/scf_summary.csv"
SCF_SUMMARY_SOURCE_OLD="${PROJECT_DIR}/obs/summaries/${SOURCE_PROJECT_NAME}/scf_summary.csv"
SCF_SUMMARY_SOURCE_ALL="${PROJECT_DIR}/obs/summaries/all_data/scf_summary.csv"
WET_SUMMARY_SOURCE_NEW="${PROJECT_DIR}/obs/${SOURCE_PROJECT_NAME}/wet_snow_summary.csv"
WET_SUMMARY_SOURCE_OLD="${PROJECT_DIR}/obs/summaries/${SOURCE_PROJECT_NAME}/wet_snow_summary.csv"
WET_SUMMARY_SOURCE_ALL="${PROJECT_DIR}/obs/summaries/all_data/wet_snow_summary.csv"

if [[ ! -f "${SCF_SUMMARY_SOURCE_NEW}" && ! -f "${SCF_SUMMARY_SOURCE_OLD}" && ! -f "${SCF_SUMMARY_SOURCE_ALL}" ]]; then
  echo "[integration] SCF summary missing in example; generating from raw rasters"
  compose_run oa-da-snowcover \
    --input-dir /data/obs/snowcover \
    --project-label "${SOURCE_PROJECT_NAME}" \
    --setup-dir /data \
    --overwrite
fi

if [[ ! -f "${WET_SUMMARY_SOURCE_NEW}" && ! -f "${WET_SUMMARY_SOURCE_OLD}" && ! -f "${WET_SUMMARY_SOURCE_ALL}" ]]; then
  echo "[integration] Wet-snow summary missing in example; generating from raw rasters"
  compose_run oa-da-wetsnow \
    --input-dir /data/obs/wetsnow \
    --project-label "${SOURCE_PROJECT_NAME}" \
    --setup-dir /data \
    --overwrite
fi

compose_run python - <<'PY'
import csv
from datetime import datetime, timedelta
from pathlib import Path
import yaml


def _parse_date(value: str) -> datetime:
    return datetime.fromisoformat(str(value)[:10])


def _find_summary_csv(setup_dir: Path, source_project_name: str, filename: str) -> Path:
    candidates = [
        setup_dir / "obs" / source_project_name / filename,
        setup_dir / "obs" / "summaries" / source_project_name / filename,
        setup_dir / "obs" / "summaries" / "all_data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(
        f"Missing required summary CSV '{filename}' in expected locations: "
        + ", ".join(str(p) for p in candidates)
    )


def _read_summary_dates(path: Path) -> set[str]:
    dates: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            date_val = str((row or {}).get("date", "")).strip()
            if date_val:
                dates.add(date_val[:10])
    return dates


setup_dir = Path("/data")
source_project_dirs = sorted(p for p in (setup_dir / "projects").glob("project_*") if p.is_dir() and not p.name.startswith("project_ci_"))
if not source_project_dirs:
    raise SystemExit(f"No source project found under {setup_dir / 'projects'}")
source_project_name = source_project_dirs[0].name
project_name = f"project_ci_{source_project_name.removeprefix('project_')}"
source_project_yml = setup_dir / "projects" / source_project_name / f"{source_project_name}.yml"
with source_project_yml.open("r", encoding="utf-8") as f:
    source_project_cfg = yaml.safe_load(f) or {}

project_dir = setup_dir / "projects" / project_name
project_dir.mkdir(parents=True, exist_ok=True)
project_cfg = dict(source_project_cfg)

da_cfg = dict(project_cfg.get("data_assimilation") or {})
prior_cfg = dict(da_cfg.get("prior_forcing") or {})
prior_cfg["ensemble_size"] = 4
da_cfg["prior_forcing"] = prior_cfg

scf_summary = _find_summary_csv(setup_dir, source_project_name, "scf_summary.csv")
wet_summary = _find_summary_csv(setup_dir, source_project_name, "wet_snow_summary.csv")
scf_available = _read_summary_dates(scf_summary)
wet_available = _read_summary_dates(wet_summary)
if not scf_available or not wet_available:
    raise SystemExit("SCF/Wet-snow summary CSV contains no usable dates")

events = list(da_cfg.get("assimilation_events") or [])
scf_events: list[dict] = []
wet_events: list[dict] = []
for event in events:
    variable = str((event or {}).get("variable", "")).strip().lower()
    date_raw = str((event or {}).get("date", "")).strip()
    if not date_raw:
        continue
    date_key = date_raw[:10]
    product = str((event or {}).get("product", "")).strip()
    if variable == "scf" and date_key in scf_available:
        scf_events.append({"date": date_key, "product": product})
    elif variable == "wet_snow" and date_key in wet_available:
        wet_events.append({"date": date_key, "product": product})

if not scf_events or not wet_events:
    raise SystemExit(
        "No overlapping SCF/Wet-snow events found between project assimilation_events and summary CSV dates"
    )

best_pair = None
best_gap_days = None
for scf_event in scf_events:
    scf_dt = _parse_date(scf_event["date"])
    for wet_event in wet_events:
        wet_dt = _parse_date(wet_event["date"])
        gap_days = (wet_dt - scf_dt).days
        if gap_days < 0:
            continue
        if best_gap_days is None or gap_days < best_gap_days:
            best_gap_days = gap_days
            best_pair = (scf_event, wet_event)

if best_pair is None:
    scf_event = min(scf_events, key=lambda e: e["date"])
    wet_event = min(wet_events, key=lambda e: e["date"])
else:
    scf_event, wet_event = best_pair

scf_dt = _parse_date(scf_event["date"])
wet_dt = _parse_date(wet_event["date"])
window_start = min(scf_dt, wet_dt) - timedelta(days=7)
window_end = max(scf_dt, wet_dt) + timedelta(days=7)

source_start_raw = str(source_project_cfg.get("start_date", "")).strip()
source_end_raw = str(source_project_cfg.get("end_date", "")).strip()
source_start = _parse_date(source_start_raw) if source_start_raw else window_start
source_end = _parse_date(source_end_raw) if source_end_raw else window_end

trim_start = max(window_start, source_start)
trim_end = min(window_end, source_end)
if trim_end < trim_start:
    raise SystemExit("Computed CI trim window is invalid (end before start)")

project_cfg["start_date"] = trim_start.strftime("%Y-%m-%d")
project_cfg["end_date"] = trim_end.strftime("%Y-%m-%d 21:00:00")
da_cfg["assimilation_events"] = [
    {
        "date": scf_event["date"],
        "variable": "scf",
        "product": scf_event["product"],
    },
    {
        "date": wet_event["date"],
        "variable": "wet_snow",
        "product": wet_event["product"],
    },
]
project_cfg["data_assimilation"] = da_cfg

with (project_dir / f"{project_name}.yml").open("w", encoding="utf-8") as f:
    yaml.safe_dump(project_cfg, f, sort_keys=False)

print(f"[integration/python] selected scf={scf_event['date']} wet_snow={wet_event['date']}")
print(f"[integration/python] trim window {project_cfg['start_date']} -> {project_cfg['end_date']}")
PY

compose_run python -m openamundsen_da.pipeline.project_skeleton \
  --setup-dir /data \
  --project-dir "${PROJECT_PATH}" \
  --overwrite \
  --log-level INFO

SCF_SUMMARY_CANDIDATE_NEW="${PROJECT_DIR}/obs/${SOURCE_PROJECT_NAME}/scf_summary.csv"
SCF_SUMMARY_CANDIDATE_OLD="${PROJECT_DIR}/obs/summaries/${SOURCE_PROJECT_NAME}/scf_summary.csv"
WET_SUMMARY_CANDIDATE_NEW="${PROJECT_DIR}/obs/${SOURCE_PROJECT_NAME}/wet_snow_summary.csv"
WET_SUMMARY_CANDIDATE_OLD="${PROJECT_DIR}/obs/summaries/${SOURCE_PROJECT_NAME}/wet_snow_summary.csv"

if [[ -f "${SCF_SUMMARY_CANDIDATE_NEW}" ]]; then
  SCF_SUMMARY_HOST_SOURCE="${SCF_SUMMARY_CANDIDATE_NEW}"
elif [[ -f "${SCF_SUMMARY_CANDIDATE_OLD}" ]]; then
  SCF_SUMMARY_HOST_SOURCE="${SCF_SUMMARY_CANDIDATE_OLD}"
else
  echo "[integration] ERROR: SCF summary CSV not found in expected locations:"
  echo "  - ${SCF_SUMMARY_CANDIDATE_NEW}"
  echo "  - ${SCF_SUMMARY_CANDIDATE_OLD}"
  exit 1
fi

if [[ -f "${WET_SUMMARY_CANDIDATE_NEW}" ]]; then
  WET_SUMMARY_HOST_SOURCE="${WET_SUMMARY_CANDIDATE_NEW}"
elif [[ -f "${WET_SUMMARY_CANDIDATE_OLD}" ]]; then
  WET_SUMMARY_HOST_SOURCE="${WET_SUMMARY_CANDIDATE_OLD}"
else
  echo "[integration] ERROR: Wet-snow summary CSV not found in expected locations:"
  echo "  - ${WET_SUMMARY_CANDIDATE_NEW}"
  echo "  - ${WET_SUMMARY_CANDIDATE_OLD}"
  exit 1
fi

# Mirror selected summaries under obs/<ci-project>/ for plotting defaults.
CI_OBS_DIR="${PROJECT_DIR}/obs/${PROJECT_NAME}"
mkdir -p "${CI_OBS_DIR}"
cp -f "${SCF_SUMMARY_HOST_SOURCE}" "${CI_OBS_DIR}/scf_summary.csv"
cp -f "${WET_SUMMARY_HOST_SOURCE}" "${CI_OBS_DIR}/wet_snow_summary.csv"
SCF_SUMMARY_CSV="/data/obs/${PROJECT_NAME}/scf_summary.csv"
WET_SUMMARY_CSV="/data/obs/${PROJECT_NAME}/wet_snow_summary.csv"
echo "[integration] Using SCF summary: ${SCF_SUMMARY_CSV}"
echo "[integration] Using wet-snow summary: ${WET_SUMMARY_CSV}"

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

CONTAINER_LOG_FILE="$(compose_run python - <<'PY' | tr -d '\r' | tail -n 1
from pathlib import Path

project_dirs = sorted(p for p in Path("/data/projects").glob("project_ci_*") if p.is_dir())
if not project_dirs:
    raise SystemExit("No CI project directory found under /data/projects")
project_dir = project_dirs[0]
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



#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_PYTHON="${OA_DA_NATIVE_PYTHON:-python3}"
EXPECTED_VERSION="${OA_DA_EXPECTED_VERSION:-}"
MATPLOTLIB_REQUIREMENT="${OA_DA_NATIVE_MATPLOTLIB_REQUIREMENT:-matplotlib>=3.10}"
DEPENDENCY_MODE="${OA_DA_NATIVE_DEPENDENCY_MODE:-locked}"
PIP_VERSION="${OA_DA_NATIVE_PIP_VERSION:-26.1.2}"
CONSTRAINTS_FILE="${OA_DA_NATIVE_CONSTRAINTS:-${ROOT_DIR}/constraints/native-ci-py312.txt}"
WHEEL_PATH="${WHEEL_PATH:-}"
ARTIFACT_DIR="${CI_ARTIFACT_DIR:-}"

if [[ -z "${WHEEL_PATH}" ]]; then
  mapfile -t wheels < <(find "${ROOT_DIR}/dist" -maxdepth 1 -type f -name 'openamundsen_da-*.whl' | sort)
  if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "[native-integration] ERROR: expected one wheel under ${ROOT_DIR}/dist, found ${#wheels[@]}" >&2
    exit 1
  fi
  WHEEL_PATH="${wheels[0]}"
fi
WHEEL_PATH="$(realpath "${WHEEL_PATH}")"
if [[ ! -f "${WHEEL_PATH}" ]]; then
  echo "[native-integration] ERROR: wheel not found: ${WHEEL_PATH}" >&2
  exit 1
fi
if ! command -v "${BASE_PYTHON}" >/dev/null 2>&1; then
  echo "[native-integration] ERROR: Python executable not found: ${BASE_PYTHON}" >&2
  exit 1
fi

install_args=(--disable-pip-version-check)
case "${DEPENDENCY_MODE}" in
  locked)
    if [[ ! -f "${CONSTRAINTS_FILE}" ]]; then
      echo "[native-integration] ERROR: constraints file not found: ${CONSTRAINTS_FILE}" >&2
      exit 1
    fi
    install_args+=(--constraint "${CONSTRAINTS_FILE}")
    ;;
  latest)
    ;;
  *)
    echo "[native-integration] ERROR: unsupported dependency mode '${DEPENDENCY_MODE}' (expected locked or latest)" >&2
    exit 1
    ;;
esac

VENV_PARENT="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
mkdir -p "${VENV_PARENT}"
VENV_DIR="$(mktemp -d -p "${VENV_PARENT}" oada-native-ci-XXXXXX)"

cleanup() {
  local rc=$?
  rm -rf "${VENV_DIR}"
  trap - EXIT
  exit "${rc}"
}
trap cleanup EXIT

echo "[native-integration] Creating isolated environment with ${BASE_PYTHON}"
if ! "${BASE_PYTHON}" -m venv "${VENV_DIR}"; then
  echo "[native-integration] ensurepip unavailable; bootstrapping pip into a clean venv"
  "${BASE_PYTHON}" -m venv --clear --without-pip "${VENV_DIR}"
fi
PYTHON="${VENV_DIR}/bin/python"
CLI="${VENV_DIR}/bin/openamundsen-da"
"${BASE_PYTHON}" -m pip --python "${VENV_DIR}" install \
  --disable-pip-version-check "pip==${PIP_VERSION}"

echo "[native-integration] Installing exact wheel (dependency mode: ${DEPENDENCY_MODE})"
"${PYTHON}" -m pip install "${install_args[@]}" "${WHEEL_PATH}" "${MATPLOTLIB_REQUIREMENT}"
"${PYTHON}" -m pip check
"${PYTHON}" -m pip freeze
if [[ -n "${ARTIFACT_DIR}" ]]; then
  mkdir -p "${ARTIFACT_DIR}"
  "${PYTHON}" -m pip freeze > "${ARTIFACT_DIR}/native-environment.txt"
fi

validator_args=("${WHEEL_PATH}")
if [[ -n "${EXPECTED_VERSION}" ]]; then
  validator_args+=(--expected-version "${EXPECTED_VERSION}")
fi
"${PYTHON}" "${ROOT_DIR}/scripts/ci/validate_installed_wheel.py" "${validator_args[@]}"

(
  cd "${VENV_DIR}"
  env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 \
    "${PYTHON}" - "${VENV_DIR}" "${EXPECTED_VERSION}" <<'PY'
from importlib import metadata
import json
from pathlib import Path
import sys

import matplotlib
import openamundsen_da
from packaging.version import Version

venv_dir = Path(sys.argv[1]).resolve()
expected_version = str(sys.argv[2]).strip()
package_origin = Path(openamundsen_da.__file__).resolve()
package_version = metadata.version("openamundsen-da")

if not package_origin.is_relative_to(venv_dir):
    raise RuntimeError(f"Package imported outside native test environment: {package_origin}")
if expected_version and package_version != expected_version:
    raise RuntimeError(f"Installed version {package_version!r} does not match expected {expected_version!r}")
if Version(matplotlib.__version__) < Version("3.10"):
    raise RuntimeError(f"Native compatibility run requires Matplotlib >=3.10, got {matplotlib.__version__}")

print(
    json.dumps(
        {
            "matplotlib": matplotlib.__version__,
            "openamundsen_da": package_version,
            "origin": str(package_origin),
            "python": sys.version.split()[0],
        },
        sort_keys=True,
    )
)
PY
)

export MPLBACKEND="${MPLBACKEND:-Agg}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

OA_DA_TEST_RUNTIME=native \
OA_DA_TEST_PROJECT_DRIVER=api \
OA_DA_TEST_PYTHON="${PYTHON}" \
OA_DA_TEST_CLI="${CLI}" \
bash "${ROOT_DIR}/scripts/ci/run_integration_tests.sh"

echo "[native-integration] PASS"

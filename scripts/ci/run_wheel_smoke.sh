#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_IMAGE="${CI_IMAGE:-openamundsen-da-ci:local}"
REPO_MOUNT="${ROOT_DIR}"
PROJ_MOUNT="${ROOT_DIR}/examples/rofental"
WHEEL_PATH="${WHEEL_PATH:-}"

if [[ -z "${WHEEL_PATH}" ]]; then
  mapfile -t wheels < <(find "${ROOT_DIR}/dist" -maxdepth 1 -type f -name 'openamundsen_da-*.whl' | sort)
  if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "Expected one wheel under ${ROOT_DIR}/dist, found ${#wheels[@]}" >&2
    exit 1
  fi
  WHEEL_PATH="${wheels[0]}"
fi

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    export MSYS_NO_PATHCONV=1
    export MSYS2_ARG_CONV_EXCL="*"
    REPO_MOUNT="$(cygpath -w "${ROOT_DIR}")"
    PROJ_MOUNT="$(cygpath -w "${ROOT_DIR}/examples/rofental")"
    ;;
esac

REPO="${REPO_MOUNT}" \
PROJ="${PROJ_MOUNT}" \
IMAGE="${CI_IMAGE}" \
env UID="$(id -u)" GID="$(id -g)" \
docker compose -f "${ROOT_DIR}/compose.yml" -f "${ROOT_DIR}/compose.ci.yml" run --rm oa \
  python /workspace/scripts/ci/validate_installed_wheel.py "/workspace/dist/$(basename "${WHEEL_PATH}")"

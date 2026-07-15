#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_IMAGE="${CI_IMAGE:-openamundsen-da-ci:local}"
REPO_MOUNT="${ROOT_DIR}"
PROJ_MOUNT="${ROOT_DIR}/examples/rofental"

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
docker compose run --rm oa \
  python /workspace/scripts/ci/validate_installed_wheel.py /workspace

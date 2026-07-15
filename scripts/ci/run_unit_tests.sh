#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_IMAGE="${CI_IMAGE:-openamundsen-da-ci:local}"
REPO_MOUNT="${ROOT_DIR}"
PROJ_MOUNT="${ROOT_DIR}/examples/rofental"

# Prevent Git-Bash path mangling of container-style arguments like /workspace.
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
  pytest -q /workspace/tests/unit

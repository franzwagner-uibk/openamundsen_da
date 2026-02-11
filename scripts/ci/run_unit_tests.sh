#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CI_IMAGE="${CI_IMAGE:-openamundsen-da-ci:local}"

REPO="${ROOT_DIR}" \
PROJ="${ROOT_DIR}/examples/rofental" \
IMAGE="${CI_IMAGE}" \
env UID="$(id -u)" GID="$(id -g)" \
docker compose run --rm oa \
  pytest -q /workspace/tests/unit

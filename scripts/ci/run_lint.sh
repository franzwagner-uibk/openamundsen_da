#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "${ROOT_DIR}:/work" \
  -w /work \
  ghcr.io/astral-sh/ruff:0.8.4 \
  check --no-cache --select E9,F63,F7,F82,F401,F841,ERA001 openamundsen_da scripts tests

#!/usr/bin/env bash
set -euo pipefail

# Reliable + live docs preview for WSL when the repo lives on /mnt/c (or other
# slow/notification-unreliable mounts). We mirror docs/ into /tmp and run
# Jekyll from the mirror, while polling-syncing repo changes into that mirror.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DOCS_SRC="${REPO_ROOT}/docs"

PORT="${PORT:-4001}"
LIVERELOAD_PORT="${LIVERELOAD_PORT:-35731}"
WORK_ROOT="${OA_DA_DOCS_LIVE_ROOT:-/tmp/oa-da-docs-live-sync}"
DOCS_MIRROR="${WORK_ROOT}/docs"
SYNC_LOG_PREFIX="[docs-live-sync]"

if [[ ! -d "${DOCS_SRC}" ]]; then
  echo "${SYNC_LOG_PREFIX} source docs directory not found: ${DOCS_SRC}" >&2
  exit 1
fi

mkdir -p "${WORK_ROOT}"

sync_once() {
  rsync -a --delete \
    --exclude "_site" \
    --exclude ".jekyll-cache" \
    --exclude ".sass-cache" \
    --exclude "tmp" \
    "${DOCS_SRC}/" "${DOCS_MIRROR}/"
}

cleanup() {
  if [[ -n "${SYNC_PID:-}" ]]; then
    kill "${SYNC_PID}" 2>/dev/null || true
    wait "${SYNC_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "${SYNC_LOG_PREFIX} source: ${DOCS_SRC}"
echo "${SYNC_LOG_PREFIX} mirror: ${DOCS_MIRROR}"
echo "${SYNC_LOG_PREFIX} url:    http://127.0.0.1:${PORT}/"
echo "${SYNC_LOG_PREFIX} live:   ws://127.0.0.1:${LIVERELOAD_PORT}/"

sync_once

(
  while true; do
    sync_once
    sleep 0.5
  done
) &
SYNC_PID=$!

cd "${DOCS_MIRROR}"
TMPDIR=/tmp TMP=/tmp TEMP=/tmp \
bundle exec jekyll serve \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --livereload \
  --livereload-port "${LIVERELOAD_PORT}" \
  --force_polling \
  --config _config.yml,_config_dev.yml,_config_wsl_reliable.yml

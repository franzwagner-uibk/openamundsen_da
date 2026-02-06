#!/usr/bin/env bash
set -euo pipefail

# If /data is bind-mounted, remember its current owner so we can restore it after the run.
DATA_UID=""
DATA_GID=""
if [ -d /data ]; then
  DATA_UID="$(stat -c '%u' /data 2>/dev/null || true)"
  DATA_GID="$(stat -c '%g' /data 2>/dev/null || true)"
fi

restore_data_owner() {
  if [ -d /data ] && [ -n "${DATA_UID}" ] && [ -n "${DATA_GID}" ]; then
    chown -R "${DATA_UID}:${DATA_GID}" /data 2>/dev/null || true
  fi
}
trap restore_data_owner EXIT

# Remove stale mamba lock to avoid hard failures when a previous run crashed.
lock_dir="/cache/xdg/mamba/proc"
lock_file="${lock_dir}/proc.lock"
mkdir -p "$lock_dir" 2>/dev/null || true
if ! ( : > "$lock_file" ) 2>/dev/null; then
  export XDG_CACHE_HOME="/tmp/xdg"
  lock_dir="${XDG_CACHE_HOME}/mamba/proc"
  lock_file="${lock_dir}/proc.lock"
  mkdir -p "$lock_dir"
fi
rm -f "$lock_file"

status=0
if [ "${1:-}" = "micromamba" ]; then
  "$@"
  status=$?
else
  micromamba run -n openamundsen_da "$@"
  status=$?
fi

exit "$status"

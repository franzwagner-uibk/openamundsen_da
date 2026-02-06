#!/usr/bin/env bash
set -euo pipefail

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

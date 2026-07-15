#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIST_DIR="${1:-${ROOT_DIR}/dist}"
PYTHON="${PYTHON:-python3}"

if [[ -z "${DIST_DIR}" || "${DIST_DIR}" == "/" || "${DIST_DIR}" == "${ROOT_DIR}" ]]; then
  echo "Refusing unsafe distribution output directory: ${DIST_DIR}" >&2
  exit 2
fi

rm -rf "${DIST_DIR}"
rm -rf "${ROOT_DIR}/build" "${ROOT_DIR}/openamundsen_da.egg-info"
rm -f "${ROOT_DIR}/openamundsen_da/_version.py"
mkdir -p "${DIST_DIR}"
"${PYTHON}" -m build --outdir "${DIST_DIR}" "${ROOT_DIR}"
"${PYTHON}" -m twine check "${DIST_DIR}"/*
"${PYTHON}" "${ROOT_DIR}/scripts/ci/validate_distribution.py" \
  "${DIST_DIR}" \
  --source-dir "${ROOT_DIR}"

"${PYTHON}" - "${DIST_DIR}" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys

dist_dir = Path(sys.argv[1])
artifacts = sorted((*dist_dir.glob("*.whl"), *dist_dir.glob("*.tar.gz")))
lines = [f"{sha256(path.read_bytes()).hexdigest()}  {path.name}" for path in artifacts]
(dist_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

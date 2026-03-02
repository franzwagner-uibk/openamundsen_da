#!/usr/bin/env bash
set -euo pipefail

# Local-only docs guard for common Jekyll-stalling Markdown mistakes.

DOCS_DIR="${1:-docs}"

if [[ ! -d "${DOCS_DIR}" ]]; then
  echo "[docs-guard] docs directory not found: ${DOCS_DIR}" >&2
  exit 2
fi

status=0
TMP_MATCH="/tmp/oa-da-docs-guard-rg.out"

check_file() {
  local f="$1"

  # Catch escaped ".md" inside Jekyll {% link %} tags (breaks link resolution).
  if rg -nH '\{% link [^%}]*\\\.md %\}' "$f" >"${TMP_MATCH}" 2>/dev/null; then
    while IFS= read -r line; do
      [[ -n "${line}" ]] || continue
      echo "${line}: escaped \\.md inside {% link %} breaks Jekyll link resolution"
    done <"${TMP_MATCH}"
    status=1
  fi

  local first_line
  first_line="$(head -n 1 "$f" || true)"
  if [[ "${first_line}" != "---" ]]; then
    return 0
  fi

  awk '
    BEGIN { in_fm=1; closed=0; found_bad=0 }
    NR==1 { next }
    in_fm {
      if ($0 == "---") { closed=1; in_fm=0; next }
      if ($0 ~ /^-{4,}$/) {
        printf "%s:%d: suspicious front-matter separator \"%s\" (expected exactly ---)\n", FILENAME, NR, $0
        found_bad=1
      }
      if ($0 ~ /^[[:space:]]*[A-Za-z0-9_-]+\\_[A-Za-z0-9_-]*[[:space:]]*:/) {
        printf "%s:%d: escaped underscore in YAML key (e.g. nav\\_order). Use plain key names.\n", FILENAME, NR
        found_bad=1
      }
    }
    END {
      if (!closed) {
        printf "%s:1: front matter starts with --- but no closing --- was found\n", FILENAME
        found_bad=1
      }
      exit(found_bad ? 1 : 0)
    }
  ' "$f" || status=1
}

while IFS= read -r f; do
  check_file "$f"
done < <(
  find "$DOCS_DIR" -type f -name '*.md' \
    ! -path '*/_site/*' \
    ! -path '*/.jekyll-cache/*' \
    | sort
)

rm -f "${TMP_MATCH}" >/dev/null 2>&1 || true

if [[ ${status} -eq 0 ]]; then
  echo "[docs-guard] OK"
else
  echo "[docs-guard] FAILED - fix the file(s) above; live sync stays on last good snapshot" >&2
fi

exit "${status}"

#!/usr/bin/env bash
set -euo pipefail

# Local-only docs preview service.
# - Mirrors docs/ into /tmp and runs Jekyll from the mirror (fast on WSL)
# - Fast guard catches common Markdown/Jekyll syntax traps
# - Strict Jekyll preflight build catches deeper Liquid/Jekyll build errors
# - On failure, the served site is replaced by a detailed browser error page

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_DEFAULT}}"
DOCS_SRC="${DOCS_SRC:-${REPO_ROOT}/docs}"
DOCS_GUARD="${SCRIPT_DIR}/jekyll_md_guard_local.sh"

PORT="${PORT:-4001}"
LIVERELOAD_PORT="${LIVERELOAD_PORT:-35731}"
WORK_ROOT="${OA_DA_DOCS_LIVE_ROOT:-/tmp/oa-da-docs-live-sync}"
DOCS_STAGE="${WORK_ROOT}/docs_stage"
DOCS_MIRROR="${WORK_ROOT}/docs"
SITE_DEST="${OA_DA_DOCS_SITE_DEST:-/tmp/oa-da-docs-site}"
PREFLIGHT_DEST="${WORK_ROOT}/preflight_site"

GUARD_LAST_ERR="${WORK_ROOT}/last_guard_error.log"
JEKYLL_LAST_ERR="${WORK_ROOT}/last_jekyll_preflight_error.log"
LAST_ERROR_KIND_FILE="${WORK_ROOT}/last_error_kind.txt"
LAST_ERROR_LOG_PATH_FILE="${WORK_ROOT}/last_error_log_path.txt"
STATE_FILE="${WORK_ROOT}/sync_state.txt"
LAST_SOURCE_SIG_FILE="${WORK_ROOT}/last_source_sig.txt"
ERROR_PAGE_HTML="${WORK_ROOT}/build_blocked.html"
SYNC_LOG_PREFIX="[docs-live-sync]"
JEKYLL_IMAGE="${OA_DA_DOCS_JEKYLL_IMAGE:-jekyll/jekyll:4}"

detect_jekyll_runner() {
  local mode
  mode="${OA_DA_DOCS_JEKYLL_RUNNER:-auto}"
  case "${mode}" in
    local)
      if ! command -v bundle >/dev/null 2>&1; then
        echo "${SYNC_LOG_PREFIX} OA_DA_DOCS_JEKYLL_RUNNER=local but 'bundle' is not installed in WSL" >&2
        exit 1
      fi
      printf 'local'
      ;;
    docker)
      if ! command -v docker >/dev/null 2>&1; then
        echo "${SYNC_LOG_PREFIX} OA_DA_DOCS_JEKYLL_RUNNER=docker but 'docker' is not available in WSL" >&2
        exit 1
      fi
      printf 'docker'
      ;;
    auto)
      if command -v bundle >/dev/null 2>&1; then
        printf 'local'
      elif command -v docker >/dev/null 2>&1; then
        printf 'docker'
      else
        echo "${SYNC_LOG_PREFIX} neither local Jekyll (bundle) nor Docker is available" >&2
        echo "${SYNC_LOG_PREFIX} install Ruby+Bundler or enable Docker Desktop WSL integration" >&2
        exit 1
      fi
      ;;
    *)
      echo "${SYNC_LOG_PREFIX} invalid OA_DA_DOCS_JEKYLL_RUNNER='${mode}' (use auto|local|docker)" >&2
      exit 1
      ;;
  esac
}

JEKYLL_RUNNER="$(detect_jekyll_runner)"

run_jekyll_preflight() {
  if [[ "${JEKYLL_RUNNER}" == "local" ]]; then
    (
      cd "${DOCS_STAGE}" && \
      TMPDIR=/tmp TMP=/tmp TEMP=/tmp \
      bundle exec jekyll build \
        --trace \
        --config _config.yml,_config_dev.yml,_config_wsl_reliable.yml \
        --destination "${PREFLIGHT_DEST}"
    )
    return
  fi

  mkdir -p "${WORK_ROOT}/bundle-cache"
  docker run --rm \
    -u "$(id -u):$(id -g)" \
    -e BUNDLE_PATH=/work/bundle-cache \
    -v "${WORK_ROOT}:/work" \
    -w /work/docs_stage \
    "${JEKYLL_IMAGE}" \
    sh -lc "bundle install && bundle exec jekyll build --trace --config _config.yml,_config_dev.yml,_config_wsl_reliable.yml --destination /work/preflight_site"
}

run_jekyll_serve() {
  if [[ "${JEKYLL_RUNNER}" == "local" ]]; then
    TMPDIR=/tmp TMP=/tmp TEMP=/tmp \
    bundle exec jekyll serve \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --livereload \
      --livereload-port "${LIVERELOAD_PORT}" \
      --force_polling \
      --config _config.yml,_config_dev.yml,_config_wsl_reliable.yml
    return
  fi

  mkdir -p "${WORK_ROOT}/bundle-cache"
  docker run --rm \
    -u "$(id -u):$(id -g)" \
    -e BUNDLE_PATH=/work/bundle-cache \
    -p "127.0.0.1:${PORT}:${PORT}" \
    -p "127.0.0.1:${LIVERELOAD_PORT}:${LIVERELOAD_PORT}" \
    -v "${WORK_ROOT}:/work" \
    -w /work/docs \
    "${JEKYLL_IMAGE}" \
    sh -lc "bundle install && bundle exec jekyll serve --host 0.0.0.0 --port ${PORT} --livereload --livereload-port ${LIVERELOAD_PORT} --force_polling --config _config.yml,_config_dev.yml,_config_wsl_reliable.yml"
}

if [[ ! -d "${DOCS_SRC}" ]]; then
  echo "${SYNC_LOG_PREFIX} source docs directory not found: ${DOCS_SRC}" >&2
  exit 1
fi
if [[ ! -x "${DOCS_GUARD}" ]]; then
  echo "${SYNC_LOG_PREFIX} guard script not found/executable: ${DOCS_GUARD}" >&2
  exit 1
fi

mkdir -p "${WORK_ROOT}"
# Reset transient state on start so existing errors are emitted/published again.
rm -f "${STATE_FILE}" "${LAST_SOURCE_SIG_FILE}"

sync_dir() {
  local src="$1"
  local dest="$2"
  rsync -a --delete \
    --exclude "_site" \
    --exclude ".jekyll-cache" \
    --exclude ".sass-cache" \
    --exclude "tmp" \
    "${src}/" "${dest}/"
}

source_signature() {
  find "${DOCS_SRC}" -type f \
    ! -path '*/_site/*' \
    ! -path '*/.jekyll-cache/*' \
    -printf '%P|%T@|%s\n' \
    | LC_ALL=C sort \
    | sha256sum \
    | awk '{print $1}'
}

read_state() {
  if [[ -f "${STATE_FILE}" ]]; then
    cat "${STATE_FILE}"
  else
    printf 'init'
  fi
}

write_state() {
  printf '%s' "$1" > "${STATE_FILE}"
}

set_error_meta() {
  local kind="$1"
  local path="$2"
  printf '%s' "${kind}" > "${LAST_ERROR_KIND_FILE}"
  printf '%s' "${path}" > "${LAST_ERROR_LOG_PATH_FILE}"
}

clear_error_meta() {
  : > "${LAST_ERROR_KIND_FILE}"
  : > "${LAST_ERROR_LOG_PATH_FILE}"
}

current_error_kind() {
  if [[ -f "${LAST_ERROR_KIND_FILE}" ]]; then
    cat "${LAST_ERROR_KIND_FILE}"
  else
    printf 'unknown'
  fi
}

current_error_log_path() {
  if [[ -f "${LAST_ERROR_LOG_PATH_FILE}" ]]; then
    cat "${LAST_ERROR_LOG_PATH_FILE}"
  else
    printf '%s' "${GUARD_LAST_ERR}"
  fi
}

html_escape_stream() {
  sed -E 's/\x1B\\[[0-9;]*[[:alpha:]]//g' \
    | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}

first_error_line() {
  local log_path="$1"
  if [[ ! -f "${log_path}" ]]; then
    return 0
  fi
  grep -m1 -E 'YAML Exception|Liquid Exception|SyntaxError|Traceback|Error:|Exception|:[0-9]+:' "${log_path}" \
    || head -n 1 "${log_path}" \
    || true
}

write_error_page() {
  local ts error_kind error_log first_line stage_label stage_desc stage_class
  ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  error_kind="$(current_error_kind)"
  error_log="$(current_error_log_path)"
  first_line="$(first_error_line "${error_log}")"

  case "${error_kind}" in
    guard)
      stage_label="Fast Guard"
      stage_desc="A local markdown/Jekyll trap check failed before syncing changes."
      stage_class="#7f1d1d"
      ;;
    jekyll-preflight)
      stage_label="Jekyll Preflight"
      stage_desc="A strict Jekyll build check failed (Liquid/Jekyll/YAML/render error)."
      stage_class="#7c2d12"
      ;;
    *)
      stage_label="Unknown"
      stage_desc="The local docs preview reported an unknown error stage."
      stage_class="#334155"
      ;;
  esac

  mkdir -p "${WORK_ROOT}"
  {
    cat <<'HTML_HEAD'
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="1">
  <title>Docs Build Blocked</title>
  <style>
    :root { color-scheme: dark; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: #e5e7eb;
      background: radial-gradient(circle at 15% 15%, #1f2937 0%, #0b1020 45%, #06080f 100%);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .wrap { max-width: 1180px; margin: 0 auto; padding: 28px; }
    .panel {
      background: rgba(17, 24, 39, 0.88);
      border: 1px solid rgba(148, 163, 184, 0.16);
      border-radius: 14px;
      box-shadow: 0 20px 45px rgba(0,0,0,.35);
      padding: 18px;
      margin-bottom: 16px;
    }
    .row { display: grid; grid-template-columns: 1fr; gap: 14px; }
    @media (min-width: 980px) { .row { grid-template-columns: 1.1fr .9fr; } }
    .badge {
      display:inline-block; padding: 6px 10px; border-radius: 999px;
      font-size: 12px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
      background:#7f1d1d; color:#fee2e2; border:1px solid rgba(254,202,202,.25);
    }
    .title { margin: 12px 0 8px; font-size: 28px; line-height: 1.15; }
    .muted { color: #cbd5e1; line-height: 1.45; }
    .kvs { display:grid; grid-template-columns: 12rem 1fr; gap: 8px 12px; margin-top: 10px; }
    .k { color: #93c5fd; }
    .v { color: #f8fafc; word-break: break-word; }
    .stage {
      display:inline-block; margin-top:10px; padding:6px 10px; border-radius:8px;
      color:#fff; font-weight:700; border:1px solid rgba(255,255,255,.14);
    }
    .summary {
      background: rgba(127, 29, 29, .18);
      border: 1px solid rgba(248, 113, 113, .28);
      border-radius: 10px; padding: 12px; color: #fee2e2;
    }
    .tips ul { margin: 8px 0 0 18px; }
    .tips li { margin: 4px 0; }
    pre {
      margin: 0; white-space: pre-wrap; word-break: break-word; overflow: auto;
      background: #050914; color: #dbeafe; border: 1px solid rgba(59,130,246,.18);
      border-radius: 10px; padding: 14px; line-height: 1.38;
      max-height: 52vh;
    }
    code { background: rgba(148,163,184,.14); padding: 2px 5px; border-radius: 6px; }
    h2 { margin: 0 0 10px; font-size: 18px; }
    .footer { color:#94a3b8; font-size:12px; margin-top: 8px; }
    .autorefresh { margin-top: 10px; color: #bfdbfe; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="panel">
      <span class="badge">Build Blocked</span>
      <div class="title">Local docs preview is blocked by a source error</div>
      <div class="muted">The preview server is still running, but the site HTML is replaced with this error page until the problem is fixed and saved.</div>
HTML_HEAD
    printf '      <span class="stage" style="background:%s">Stage: %s</span>\n' "${stage_class}" "${stage_label}"
    printf '      <div class="muted" style="margin-top:10px">%s</div>\n' "${stage_desc}"
    printf '      <div class="autorefresh"><strong>Auto-refresh:</strong> This page reloads every 1 second and will switch back automatically after the source error is fixed and saved.</div>\n'
    cat <<'HTML_KV'
      <div class="kvs">
        <div class="k">Time</div><div class="v">__TIME__</div>
        <div class="k">Repo docs</div><div class="v">__DOCS_SRC__</div>
        <div class="k">Error log</div><div class="v"><code>__ERR_LOG__</code></div>
        <div class="k">Mirror</div><div class="v"><code>__DOCS_MIRROR__</code></div>
        <div class="k">Served site</div><div class="v"><code>__SITE_DEST__</code></div>
      </div>
HTML_KV
    cat <<'HTML_ROW_A'
    </div>

    <div class="row">
      <div class="panel">
        <h2>First Error</h2>
        <div class="summary">
HTML_ROW_A
    if [[ -n "${first_line}" ]]; then
      printf '%s\n' "${first_line}" | html_escape_stream
    else
      printf 'No error summary line found. Check the full log below.\n'
    fi
    cat <<'HTML_ROW_B'
        </div>
      </div>
      <div class="panel tips">
        <h2>What To Do</h2>
        <div class="muted">Fix the source file and save again. The checker will auto-recover and the docs page will come back without restarting the server.</div>
        <ul>
          <li>YAML front matter must open and close with exactly <code>---</code></li>
          <li>Do not escape front-matter keys (use <code>nav_order</code>, not <code>nav\_order</code>)</li>
          <li>Do not escape Jekyll link paths (use <code>{% link x.md %}</code>, not <code>{% link x\.md %}</code>)</li>
          <li>If this is a Jekyll/Liquid error, read the preflight log below for the exact stack/message</li>
        </ul>
      </div>
    </div>

    <div class="panel">
      <h2>Captured Error Output</h2>
      <pre>
HTML_ROW_B
    if [[ -f "${error_log}" ]]; then
      html_escape_stream < "${error_log}"
    else
      printf 'No error log file found at: %s\n' "${error_log}" | html_escape_stream
    fi
    cat <<'HTML_END'
      </pre>
      <div class="footer">Local-only preview helper. Fix + save to auto-recover.</div>
    </div>
  </div>
</body>
</html>
HTML_END
  } > "${ERROR_PAGE_HTML}.tmp"

  # Fill placeholders after escaping to keep the HTML template readable.
  sed \
    -e "s|__TIME__|${ts//|/\\|}|g" \
    -e "s|__DOCS_SRC__|${DOCS_SRC//|/\\|}|g" \
    -e "s|__ERR_LOG__|${error_log//|/\\|}|g" \
    -e "s|__DOCS_MIRROR__|${DOCS_MIRROR//|/\\|}|g" \
    -e "s|__SITE_DEST__|${SITE_DEST//|/\\|}|g" \
    "${ERROR_PAGE_HTML}.tmp" > "${ERROR_PAGE_HTML}"
  rm -f "${ERROR_PAGE_HTML}.tmp"
}

publish_error_page() {
  local quiet="${1:-0}"
  local replaced_count=0 dir_index_count=0
  write_error_page
  mkdir -p "${SITE_DEST}"
  while IFS= read -r -d '' d; do
    cp "${ERROR_PAGE_HTML}" "${d}/index.html"
    dir_index_count=$((dir_index_count + 1))
  done < <(find "${SITE_DEST}" -type d -print0 2>/dev/null || true)
  while IFS= read -r -d '' f; do
    [[ "${f}" == "${SITE_DEST}/index.html" ]] && continue
    cp "${ERROR_PAGE_HTML}" "${f}"
    replaced_count=$((replaced_count + 1))
  done < <(find "${SITE_DEST}" -type f -name '*.html' -print0 2>/dev/null || true)
  if [[ "${quiet}" != "1" ]]; then
    echo "${SYNC_LOG_PREFIX} published build-blocked error page (dir index files: ${dir_index_count}, replaced html files: ${replaced_count})"
  fi
}

preflight_jekyll_build() {
  local log_tmp
  log_tmp="$(mktemp "${WORK_ROOT}/jekyll-preflight.XXXXXX.log")"
  if run_jekyll_preflight >"${log_tmp}" 2>&1; then
    rm -f "${log_tmp}"
    return 0
  fi
  mv "${log_tmp}" "${JEKYLL_LAST_ERR}"
  return 1
}

sync_cycle() {
  local sig prev_sig state
  local guard_tmp guard_emit jekyll_emit

  sig="$(source_signature)"
  prev_sig="$(cat "${LAST_SOURCE_SIG_FILE}" 2>/dev/null || true)"
  state="$(read_state)"

  if [[ -n "${prev_sig}" && "${sig}" == "${prev_sig}" ]]; then
    if [[ "${state}" == failed_* ]]; then
      publish_error_page 1
    fi
    return 0
  fi

  guard_tmp="$(mktemp "${WORK_ROOT}/guard.XXXXXX.log")"
  if ! bash "${DOCS_GUARD}" "${DOCS_SRC}" >"${guard_tmp}" 2>&1; then
    guard_emit=1
    if [[ -f "${GUARD_LAST_ERR}" ]] && cmp -s "${guard_tmp}" "${GUARD_LAST_ERR}"; then
      guard_emit=0
    fi
    if [[ "${state}" != failed_guard ]]; then
      echo "${SYNC_LOG_PREFIX} guard blocked sync; publishing error page"
      echo "${SYNC_LOG_PREFIX} details: ${GUARD_LAST_ERR}"
    fi
    cp "${guard_tmp}" "${GUARD_LAST_ERR}"
    set_error_meta "guard" "${GUARD_LAST_ERR}"
    publish_error_page "$([[ ${guard_emit} -eq 1 ]] && echo 0 || echo 1)"
    if [[ ${guard_emit} -eq 1 ]]; then
      cat "${guard_tmp}" >&2
    fi
    write_state "failed_guard"
    printf '%s' "${sig}" > "${LAST_SOURCE_SIG_FILE}"
    rm -f "${guard_tmp}"
    return 0
  fi
  rm -f "${guard_tmp}"

  sync_dir "${DOCS_SRC}" "${DOCS_STAGE}"
  if ! preflight_jekyll_build; then
    jekyll_emit=1
    if [[ -f "${JEKYLL_LAST_ERR}" ]] && [[ -f "$(current_error_log_path)" ]] && [[ "$(current_error_kind)" == "jekyll-preflight" ]] && cmp -s "${JEKYLL_LAST_ERR}" "$(current_error_log_path)"; then
      jekyll_emit=0
    fi
    if [[ "${state}" != failed_jekyll ]]; then
      echo "${SYNC_LOG_PREFIX} jekyll preflight failed; publishing error page"
      echo "${SYNC_LOG_PREFIX} details: ${JEKYLL_LAST_ERR}"
    fi
    set_error_meta "jekyll-preflight" "${JEKYLL_LAST_ERR}"
    publish_error_page "$([[ ${jekyll_emit} -eq 1 ]] && echo 0 || echo 1)"
    if [[ ${jekyll_emit} -eq 1 ]]; then
      cat "${JEKYLL_LAST_ERR}" >&2
    fi
    write_state "failed_jekyll"
    printf '%s' "${sig}" > "${LAST_SOURCE_SIG_FILE}"
    return 0
  fi

  sync_dir "${DOCS_STAGE}" "${DOCS_MIRROR}"
  if [[ "${state}" == failed_* ]]; then
    echo "${SYNC_LOG_PREFIX} guard recovered: syncing changes again"
    # We replace served HTML files with an error page during failures. If the source
    # tree recovers to a previously-valid state, rsync may make no mirror changes.
    # Touch a real source file in the mirror once to force Jekyll serve to rebuild.
    if [[ -f "${DOCS_MIRROR}/index.md" ]]; then
      touch "${DOCS_MIRROR}/index.md"
    fi
  fi
  clear_error_meta
  write_state "ok"
  printf '%s' "${sig}" > "${LAST_SOURCE_SIG_FILE}"
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
echo "${SYNC_LOG_PREFIX} stage:  ${DOCS_STAGE}"
echo "${SYNC_LOG_PREFIX} site:   ${SITE_DEST}"
echo "${SYNC_LOG_PREFIX} url:    http://127.0.0.1:${PORT}/"
echo "${SYNC_LOG_PREFIX} live:   ws://127.0.0.1:${LIVERELOAD_PORT}/"
echo "${SYNC_LOG_PREFIX} guard:  ${DOCS_GUARD}"
echo "${SYNC_LOG_PREFIX} state:  ${STATE_FILE}"
echo "${SYNC_LOG_PREFIX} jekyll runner: ${JEKYLL_RUNNER}"
if [[ "${JEKYLL_RUNNER}" == "docker" ]]; then
  echo "${SYNC_LOG_PREFIX} jekyll image: ${JEKYLL_IMAGE}"
fi

sync_cycle

if [[ ! -f "${DOCS_MIRROR}/_config.yml" ]]; then
  echo "${SYNC_LOG_PREFIX} no valid mirrored docs snapshot available (checks blocked initial sync)" >&2
  echo "${SYNC_LOG_PREFIX} fix docs errors, then rerun. See logs under ${WORK_ROOT}" >&2
  exit 1
fi

(
  while true; do
    sync_cycle
    sleep 0.5
  done
) &
SYNC_PID=$!

cd "${DOCS_MIRROR}"
run_jekyll_serve

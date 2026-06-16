# openAMUNDSEN-DA v1.0 Release Code Review - 2026-06-15

## Scope

This review covers the full `openamundsen_da` repository as a v1.0 release audit:

- Python package code, scripts, tests and CI wrappers
- Docker image inputs, Compose workflow and package metadata
- shipped examples and documentation contracts
- repo-local `AGENTS.md` rules for setup/project/step terminology, config ownership, fail-fast behavior and Docker-first validation

The review uses `docs/reference/repo-code-review-2026-04-17.md` as a baseline, then rechecks the current tree. The current inventory is:

- 874 tracked files
- 199 tracked Python files
- 43,452 tracked production package LOC under `openamundsen_da/**/*.py`
- largest production files: `methods/viz/maps/panel_renderers.py` (3044 LOC), `methods/viz/plots/result_overview.py` (2533 LOC), `methods/viz/plots/assimilation/weights.py` (1734 LOC), `methods/wet_snow/area.py` (1574 LOC), `methods/viz/reports/project_collection_pdf.py` (1205 LOC)

Standards references used for the release-engineering pass:

- PyPA `pyproject.toml` guide: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- Dockerfile best practices: https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
- Ruff rule reference, including `PLR2004`: https://docs.astral.sh/ruff/rules/magic-value-comparison/

## Baseline And Changes Applied

Baseline before edits:

- `scripts/ci/run_lint.sh`: passed with the old fatal-only Ruff gate
- `scripts/ci/run_unit_tests.sh`: `586 passed, 1 skipped, 137 warnings, 2 subtests passed`
- Expanded static probe before cleanup: `307 PLR2004`, `45 ARG001`, `13 ARG002`, `2 ERA001`, `2 F841`, `1 F401`

Changes applied in this branch:

- Release metadata: `pyproject.toml`, package `__version__` and `io.__version__` now report `1.0.0`; package metadata now includes README, MIT license expression, authors, URLs, keywords and release classifiers.
- Build hygiene: removed UTF-8 BOMs from tracked text/config files, normalized `examples/rofental/rofental.yml` to LF, fixed mojibake in shipped configs/docs, and excluded `docs/_site/` from Docker context.
- Lint gate: `scripts/ci/run_lint.sh` now enforces `F401`, `F841` and `ERA001` in addition to fatal Ruff checks.
- Pre-v1 cleanup: removed transitional aliases `cli_setup_daily`, `cli_model_setup`, `_read_resampling_from_setup`, the `methods.pf.plot_weights` re-export, the unused `oa-da-resample --project-dir` option, top-level likelihood/resampling fallback reads, and migration-era cleanup of historical project/map/benchmark plot locations.
- Docs/tests: updated resampling docs, setup/project config ownership, Cloudflare Pages docs, terminology drift and changelog entries for v1.0.

Post-cleanup expanded probe:

- `F401`, `F841` and `ERA001`: 0 findings
- Remaining advisory findings: `307 PLR2004`, `46 ARG001`, `13 ARG002`

## Review Matrix

| Area | Review result | Action |
| --- | --- | --- |
| `core/`, `io/`, config/path helpers | Path and config ownership is mostly centralized. `core.constants`, `io.paths` and validation helpers are actively reused. | Enforced stale-code lint. Keep adding constants here only for cross-package contracts. |
| `pipeline/` | Main orchestration is stable but `project.py` remains large and mixes orchestration, plotting hooks, diagnostics and cleanup decisions. | Removed historical plot-root cleanup. Future split should extract post-run artifact hooks, not alter CLI behavior. |
| `observer/` | Observation preprocessing remains strict and product-aware. Summary fallback behavior is documented and therefore treated as retained v1 compatibility. | No behavior removal. Future work: decide whether documented legacy summary defaults should become deprecation warnings. |
| `methods/pf/` | Assimilation/resampling now reads DA config from project-owned `data_assimilation` only. SCF wrapper remains used by pipeline/tests and is a current internal API. | Removed old aliases and top-level config fallbacks. |
| `methods/wet_snow/` | Wet-snow area code still has high LOC and mixed CLI/helper responsibilities, but core behavior is covered. | Removed unused compatibility wrapper/alias. Future split should separate CLI, class loading and daily project orchestration. |
| `methods/viz/` | Plot/map ownership has improved since April, but the largest files and many magic-number findings are still visualization layout/style thresholds. | Removed stale import. Keep `PLR2004` advisory until theme/layout constants are curated. |
| `benchmark/` | Benchmark tables and plots are now under current result paths. | Removed cleanup of historical benchmark plot locations and adjusted tests to assert canonical outputs only. |
| `subdomain/` | Subdomain wrapper honors `run_mode` and CI validators. Compatibility with subdomain CI remains a hard contract. | No risky refactor in this branch. |
| `scripts/ci/` | CI wrappers are Docker-first and aligned with repo rules. | Expanded lint gate to clean stale-code classes. |
| Docker/package metadata | Docker workflow is usable, but the Dockerfile is still a single-stage image that copies the repo and installs editable source. | Fixed build context and metadata. Deeper image hardening remains future work. |
| Shipped examples/docs | Rofental/subdomain configs remain CI baselines. Docs had stale deployment and ownership wording. | Fixed docs drift and config encoding artifacts. |

## Remaining Findings

### 1. High: visualization and reporting modules remain too large for easy v1 maintenance

The top five production modules account for much of the future review risk. The largest files are mostly visualization/reporting code:

- `methods/viz/maps/panel_renderers.py` - 3044 LOC
- `methods/viz/plots/result_overview.py` - 2533 LOC
- `methods/viz/plots/assimilation/weights.py` - 1734 LOC
- `methods/wet_snow/area.py` - 1574 LOC
- `methods/viz/reports/project_collection_pdf.py` - 1205 LOC

Recommendation: split these by behavior only when doing adjacent feature work. Do not change project-map panel sizing or global layout defaults without explicit approval.

### 2. Medium: magic numbers are real but not ready for blanket enforcement

`PLR2004` still reports 307 findings. Many are test fixture values or deliberate visualization thresholds, but several production clusters should become named constants over time:

- percent/fraction conversions such as `100.0`
- raster class sentinels such as `255`
- map layout and legend thresholds
- wet-snow-line aspect bands such as `45`, `135`, `225` and `315`

Recommendation: keep `PLR2004` advisory for now. Promote constants only where the number has domain meaning or is shared across files.

### 3. Medium: unused-argument warnings need a policy, not blind deletion

`ARG001` and `ARG002` still report 59 findings. Many are callback signatures, Matplotlib legend handlers, protocol-compatible helpers or test monkeypatch hooks.

Recommendation: adopt a naming policy for intentionally unused callback arguments, then enable `ARG` rules incrementally.

### 4. Medium: some documented compatibility remains a v1 product decision

The following compatibility behavior is still documented or tied to current workflows and was not removed:

- observation summary fallback paths under `<setup>/obs/<project>/` and `<setup>/obs/summaries/<project>/`
- project-map `below_items` support
- omitted compact output-grid config writing the full default variable/metric set
- GeoTIFF-first, NetCDF-second discovery in path helpers

Recommendation: either keep these as supported v1 contracts, or add explicit deprecation warnings in a separate branch.

### 5. Medium: Docker image hardening remains incomplete

The Dockerfile still uses a single-stage micromamba image, copies the full repository into `/workspace`, installs the package editable, and runs an entrypoint that may recursively restore `/data` ownership.

Recommendation: plan a separate Docker hardening pass after v1 behavior is frozen: non-editable install for release images, narrower build context, explicit runtime user expectations and image-size review.

### 6. Low: test warnings are noisy

The unit suite passes, but still emits warnings from third-party deprecations and many open Matplotlib figures in plot tests.

Recommendation: close figures in high-volume plotting tests and add warning filters only for known third-party noise.

## Validation

Commands run on this branch:

- Baseline before edits:
  - `scripts/ci/run_lint.sh` - passed
  - `scripts/ci/run_unit_tests.sh` - `586 passed, 1 skipped, 137 warnings, 2 subtests passed`
- Post-edit lint:
  - `scripts/ci/run_lint.sh` - passed
- Post-edit full unit wrapper:
  - `scripts/ci/run_unit_tests.sh` - `586 passed, 1 skipped, 137 warnings, 2 subtests passed`
- Targeted post-edit tests:
  - `test_plot_weights.py`
  - `test_benchmark_rendering.py`
  - `test_assimilate_uncertainty.py`
  - `test_project_cli.py`
  - `test_console_scripts.py`
  - `test_rofental_example_config.py`
  - `test_subdomain_example_config.py`
  - `test_run_mode.py`
  - result: `105 passed`
- Editable package install with `HOME=/tmp`: succeeded as `openamundsen-da-1.0.0`
- Docker build: `docker build -t openamundsen-da-v1-review:local .` - passed
- CLI/package smoke checks:
  - `oa-da-resample --help` - passed and no longer lists `--project-dir`
  - `python -c "import openamundsen_da; print(openamundsen_da.__version__)"` - `1.0.0`
- Docker integration wrappers:
  - `scripts/ci/run_integration_tests.sh` with `CI_IMAGE=openamundsen-da-v1-review:local`, `OA_DA_TEST_MAX_WORKERS=8`, `OA_DA_TEST_SETUP_RESOLUTION=500`, `OA_DA_TEST_ENSEMBLE_SIZE=2` - `[integration] PASS`
  - `scripts/ci/run_integration_tests_subdomain.sh` with `CI_IMAGE=openamundsen-da-v1-review:local`, `OA_DA_SUBDOMAIN_TEST_MAX_WORKERS=8`, `OA_DA_SUBDOMAIN_TEST_INNER_WORKERS=4` - `[subdomain-integration] PASS`
  - `scripts/ci/run_integration_tests_model_subdomain.sh` with `CI_IMAGE=openamundsen-da-v1-review:local`, `OA_DA_MODEL_SUBDOMAIN_TEST_MAX_WORKERS=8` - `[model-subdomain-integration] PASS`

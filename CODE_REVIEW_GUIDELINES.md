# General Coding & Review Guidelines

## 1. Code Review Objectives

When reviewing or developing a new module:

- Cleanliness: remove unused variables, imports, and functions.
- Modularity: move reusable or generic logic into helper modules.
- Consistency: align structure, formatting, and signatures with the repo.
- Compactness: simplify without sacrificing clarity or robustness.
- Integration: ensure the module fits naturally within `openamundsen_da` and openAMUNDSEN.

### Review Questions

- Can any part be refactored or simplified?
- Are there duplicated or redundant sections?
- Should any logic be centralized (e.g., helpers in `util`, `viz`, or `io.paths`)?
- Does the module follow our structure and formatting conventions?
- Is configuration handled consistently and defined externally where possible?
- Is there any functionality/CLI flag/option that is unnecessary given the framework/template and workflow? Prefer sensible defaults.
- Consider dropping inputs (e.g., paths or flags) that are already predefined by the process.
- is there any legacy code that is not used anymore/ dates to an older version of the code and can be removed?

List of helper modules (repo-relative paths):

#### Core and IO

- openamundsen_da/core/constants.py
- openamundsen_da/core/config.py
- openamundsen_da/core/env.py
- openamundsen_da/io/paths.py (step helpers: steps_root, list_step_dirs/list_steps_sorted)

#### Utilities

- openamundsen_da/util/ts.py
- openamundsen_da/util/stats.py
- openamundsen_da/util/aoi.py
- openamundsen_da/util/da_events.py
- openamundsen_da/util/validation.py
- openamundsen_da/util/perf_monitor.py
- openamundsen_da/util/parallel.py

#### Methods and viz helpers

- openamundsen_da/methods/daily_aoi_series.py
- openamundsen_da/methods/wet_snow/area.py
- openamundsen_da/methods/wet_snow/classify.py
- openamundsen_da/methods/viz/_style.py
- openamundsen_da/methods/viz/_utils.py

---

## 2. Code Design Principles

- Write compact, readable, and modular code.
- Ensure all variables, constants, and functions are used.
- Avoid duplicate code - consolidate shared logic in helper modules.
- Keep code robust, extensible, and maintainable for future additions.
- Use type hints and explicit function signatures.
- Follow openAMUNDSEN style conventions for naming, structure, and error handling.
- Prioritize clarity over cleverness - the code should be self-explanatory.

---

## 3. Logging

- Use `loguru` for all logging.
- Apply the standard format defined in `core/constants.py` (LOGURU_FORMAT):

```python
from openamundsen_da.core.constants import LOGURU_FORMAT
import sys
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", colorize=True, enqueue=True, format=LOGURU_FORMAT)
```

---

## 4. Repo-Specific Conventions (quick reference)

- Prefer existing helpers over re-implementing:
  - IO/paths: `list_member_dirs`, `find_member_daily_raster`, `abspath_relative_to`
  - Stats: `effective_sample_size`, `normalize_log_weights`, `sigmoid`, `envelope`, `compute_obs_sigma`
  - Viz: `draw_assimilation_vlines`, `dedupe_legend`
  - DA orchestration: `load_assimilation_events`, `compute_step_daily_series_for_all_members`, `start_perf_monitor`
- Keep modules small and cohesive: split unrelated concerns into helper modules (e.g., io/paths, util/parallel, util/da_events) rather than growing monoliths; prefer thin orchestration that delegates to helpers.
- Season layout:
  - Steps live under `season_dir/steps/step_*` (no top-level `step_*`).
- Assimilation configuration:
  - H(x) configuration (method/variable/params) is read from `project.yml` under `data_assimilation.h_of_x` (or top-level `h_of_x`); step YAML overrides are ignored.
  - Assimilation events come from `season.yml` via `data_assimilation.assimilation_events` (variable/product per date); use `util.da_events.load_assimilation_events`.
- Open loop handling:
  - The launcher always runs `open_loop` alongside `member_*` to produce a continuous reference; assimilation and resampling operate on members only.
- Plotting defaults:
  - Ensemble plots show members (and open loop when present); ensemble mean and bands are intentionally omitted.
- Performance monitoring:
  - The season pipeline can run the background monitor in `util.perf_monitor`; extend it instead of adding new ad hoc metrics.

---

## 5. Documentation

- Module headers: add a module-level docstring stating purpose, inputs/outputs, assumptions, and important side effects.
- Function docstrings: describe parameters (with types), return values, errors, and behavior. Prefer Google- or NumPy-style.
- Inline comments: annotate critical steps, invariants, and non-obvious decisions; avoid narrating the obvious.
- README updates: when adding workflows/commands, extend `README.md` at the repo root in the existing style and keep sections aligned with the repo's workflow/framework.
- Encoding: use ASCII-safe characters in docs and comments to avoid rendering issues across environments.

---

## 6. CLI, PowerShell, and Docker

- PowerShell continuation: use the backtick ` for line continuation; do not use `\`.
- One-arg-per-line style for long commands, with trailing backticks for clarity in docs.
- Docker/Docker Compose examples: prefer `docker compose run` snippets and show one CLI parameter per line using PowerShell backticks.
- Provide runnable examples for key scripts to ensure a consistent execution path across environments.

Example (PowerShell formatting):

```
docker compose run `
  --rm `
  app `
  python -m openamundsen_da.pipeline.season `
  --project my-project `
  --season 2017-2018 `
  --log-level INFO
```

---

## 7. Dependencies and Configuration

- Prefer the Python standard library when feasible; avoid adding third-party dependencies without strong justification.
- Reuse libraries already present in `openamundsen` or `openamundsen_da` to minimize environment drift.
- Centralize configuration in conf files; prefer `project.yml` for project-wide settings and keep step-specific overrides minimal.
- Leverage existing repo helpers (`core/config.py`, `core/env.py`, `core/constants.py`, `io/paths.py`, `util/stats.py`, etc.) rather than reimplementing functionality.

---

## 8. Testing and Regression Maintenance

- Treat test updates as part of the feature/fix, not as a later cleanup task.
- For every behavior change, review whether unit tests and integration validation rules must be adapted.
- Keep fast CI tests representative of real workflows while avoiding unnecessary runtime growth.
- If a new warning pattern is expected and benign, explicitly document and whitelist it in integration validation.
- If outputs/files/plots change, update the integration validator checks in the same PR.

### Rofental CI Scenario Policy

- The CI integration scenario is always based on `examples/rofental` (copied into a temp project by `scripts/ci/run_integration_tests.sh`).
- Keep `examples/rofental` and CI setup in sync:
  - when data-assimilation logic changes, verify the example project still reflects expected config/data layout;
  - when output structure changes, update validator expectations accordingly;
  - when new observables/events are added (e.g., wet snow), add corresponding test events and checks.
- Do not maintain a separate hidden CI-only project; the example project is the canonical test baseline for regression checks.

### Review Questions (Testing)

- Does this change modify observable behavior, outputs, logging, or failure modes?
- Which existing tests cover this path, and are they still valid?
- Do we need a new unit test for core logic?
- Does the trimmed integration scenario need changed dates/events/config?
- Does `scripts/ci/validate_trimmed_season.py` need updated required outputs or warning handling?
- Is `tests/README.md` still accurate after this change?

---

## 9. Ignore Files Hygiene (`.gitignore` / `.dockerignore`)

- For every PR, explicitly check whether new generated files, caches, logs, or artifacts should be ignored.
- Keep `.gitignore` aligned with local/dev/CI by excluding files that should never be versioned (temporary outputs, caches, artifacts).
- Keep `.dockerignore` aligned with build performance by excluding files not needed for image build context (caches, artifacts, local data, docs build output, nested repos).
- When adding new CI artifacts or tooling caches, update both ignore files if relevant.
- If a file should be tracked in Git but not shipped to Docker build context, update only `.dockerignore`.
- If a file should be neither tracked nor shipped, update both `.gitignore` and `.dockerignore`.

### Review Questions (Ignore Hygiene)

- Does this change introduce new generated files or directories?
- Should any new output be ignored in Git?
- Should any new output be excluded from Docker build context?
- Are `.gitignore` and `.dockerignore` still consistent with current CI/test tooling?
- Could missing ignore rules cause runner permission issues, dirty worktrees, or oversized Docker contexts?

---

## 10. Mandatory Merge Gate (Critical)

- Do not merge if CI is not green (`Ruff Lint` + `Unit and Integration Tests`).
- Do not merge behavior changes without corresponding test updates (unit and/or integration validator).
- Do not merge interface changes (CLI args, config keys, output file names/paths) without:
  - updating docs (`README.md`, `tests/README.md`) and
  - clearly noting the compatibility impact in PR/commit message.

### Review Questions (Mandatory)

- Are all required CI checks green on the branch to be merged?
- Are tests and docs updated for any behavior/interface change?
- Does this change introduce a breaking workflow/config/output change, and is that explicitly documented?

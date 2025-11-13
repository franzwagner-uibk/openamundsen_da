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

List of helper modules (repo‑relative paths):

#### Core and IO

- openamundsen_da/core/constants.py
- openamundsen_da/core/config.py
- openamundsen_da/core/env.py
- openamundsen_da/io/paths.py

#### Utilities

- openamundsen_da/util/ts.py
- openamundsen_da/util/stats.py
- openamundsen_da/util/aoi.py

#### Viz helpers

- openamundsen_da/methods/viz/\_style.py
- openamundsen_da/methods/viz/\_utils.py

---

## 2. Code Design Principles

- Write compact, readable, and modular code.
- Ensure all variables, constants, and functions are used.
- Avoid duplicate code — consolidate shared logic in helper modules.
- Keep code robust, extensible, and maintainable for future additions.
- Use type hints and explicit function signatures.
- Follow openAMUNDSEN‑style conventions for naming, structure, and error handling.
- Prioritize clarity over cleverness — the code should be self‑explanatory.

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

## 4. Repo‑Specific Conventions (quick reference)

- Prefer existing helpers over re‑implementing:
  - IO/paths: `list_member_dirs`, `find_member_daily_raster`, `abspath_relative_to`
  - Stats: `effective_sample_size`, `normalize_log_weights`, `sigmoid`, `envelope`
  - Viz: `draw_assimilation_vlines`, `dedupe_legend`
- Assimilation configuration precedence:
  - H(x) configuration (method/variable/params) is read from `project.yml` under `data_assimilation.h_of_x`; the step YAML `h_of_x` is a fallback.
- Open loop handling:
  - The launcher always runs `open_loop` alongside `member_*` to produce a continuous reference; assimilation and resampling operate on members only.
 - Plotting defaults:
   - Ensemble plots show members (and open loop when present); ensemble mean and bands are intentionally omitted.

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

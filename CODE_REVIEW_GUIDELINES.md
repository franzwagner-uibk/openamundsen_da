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
- Is there any functionality/CLI flag/option that is not necessary and can be assumed by default following the framework/ ptoject template/structure and workflow and therefore dropped? Think aboput input paths/ flags/ options that dont need to bechosen because they are pre-defined by the process

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

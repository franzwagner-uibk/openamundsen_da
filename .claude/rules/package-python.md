---
paths:
  - "openamundsen_da/**/*.py"
---

# Python Package Rules

- Preserve public CLI, config, path, and output contracts unless the task explicitly changes them.
- Reuse existing `core`, `io`, and `util` helpers before adding new config or path logic.
- Prefer fail-fast validation for dates, products, grids, paths, and observation assumptions.
- Keep package boundaries explicit: orchestration in `pipeline`, preprocessing in `observer`, scientific logic in `methods`, distributed orchestration in `subdomain`.

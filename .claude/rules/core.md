---
paths:
  - "openamundsen_da/core/**/*.py"
  - "openamundsen_da/io/**/*.py"
  - "openamundsen_da/util/**/*.py"
---

# Core Runtime Rules

- Preserve deterministic config merge, environment setup, path discovery, manifests, and runner behavior.
- Avoid alternate discovery heuristics or silent defaults when config should make intent explicit.
- If command or discovery behavior changes, update tests, docs, and CI wrappers together.

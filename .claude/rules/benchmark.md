---
paths:
  - "openamundsen_da/benchmark/**/*.py"
---

# Benchmark Rules

- Benchmark outputs are a scientific contract; keep filenames, table semantics, and score meaning stable unless intentionally changed.
- Preserve explicit open-loop, prior, and posterior comparisons where supported.
- If benchmark outputs move or change shape, update validators, docs, tests, and shipped examples together.

# Benchmark Agent Notes

Inherit the parent `openamundsen_da/AGENTS.md`. This file adds benchmark-specific rules.

- Benchmark outputs under `results/benchmark/` are a user-facing scientific contract; keep filenames, table shapes, and summary semantics stable unless intentionally changed.
- Preserve explicit open-loop, prior, and posterior comparisons where they exist; do not collapse benchmark logic to posterior-mean-only storytelling.
- Maintain consistency across extract, aggregate, render, and reporting layers; do not change one stage in isolation.
- Do not broaden or reinterpret benchmark metrics silently; scientific meaning must stay explicit, especially for sigma-aware scores and independence labels.
- If benchmark outputs or figure names change, update tests, docs, validators, and shipped examples in the same work.
- Preserve the headline performance figure path under `results/plots/assim/scores/` unless the task explicitly redefines it.

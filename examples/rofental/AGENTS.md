# Rofental Example Notes

Inherit `examples/AGENTS.md`. This file adds local rules for the canonical single-domain baseline.

- `examples/rofental` is the primary shipped single-domain regression setup used across docs and CI.
- Treat `projects/project_2022_2023/` as the canonical project contract unless the task explicitly changes that baseline.
- Do not use this shipped tree for paper-specific or exploratory runs; copy it to `dev_examples/` first.
- If config, plots, maps, or outputs change here, update `scripts/ci/validate_trimmed_project.py`, affected tests, and docs together.
- Keep Rofental changes reproducible and conservative because it is both the tutorial surface and the CI regression anchor.
- Keep local changes minimal and reproducible; avoid incidental churn in large observation datasets.

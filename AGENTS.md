# openAMUNDSEN-DA Agent Guide

This repository uses layered instructions on purpose: Codex reads the `AGENTS.md` chain, `CLAUDE.md` imports this root file and adds `.claude/rules/`, and Cursor should rely on this root file plus `.cursor/rules/*.mdc`. Keep shared project facts here and keep tool-specific scoping in the tool-specific rule directories.

- Treat `README.md`, `docs/reference/package-structure.md`, `docs/running.md`, `docs/guides/configuration.md`, `tests/README.md`, `examples/README.md`, and `.github/workflows/ci.yml` as the authoritative workflow and contract sources.
- openAMUNDSEN-DA is currently an operational ROI-scale particle-filter layer on top of openAMUNDSEN. Do not describe current ROI weighting as spatial localization unless the task explicitly implements localization.
- Treat EO and station observations as product-specific evidence with uncertainty and representativeness limits, not as universal ground truth.
- Work on a feature branch only. Never develop on `main`.
- Prefer Docker-first execution via `compose.yml` and the documented `docker compose run --rm oa ...` commands.
- Default heavy local runs to `--max-workers 24` unless the task or user says otherwise.
- Use repo terminology exactly: `setup`, `project`, `step`, `member`, `run`. Do not rename these to `season` or `scenario`.
- Keep config ownership strict: setup YAML is pure openAMUNDSEN/shared data, project YAML owns DA config and `assimilation_events`, step YAML owns step-window overrides only.
- Prefer explicit required config and fail-fast validation over hidden fallback behavior.
- Treat shipped examples as product surface and CI baselines, not scratch data. Use a copied `dev_examples/` tree for experiments.
- Any user-visible behavior, CLI, config, workflow, output, benchmark, or shipped-example change must keep docs, tests, CI validators, and examples aligned.
- Do not edit generated output as source, especially `docs/_site/` and run-generated example result trees.
- The worktree may already contain unrelated user changes. Do not revert or rewrite them unless the task explicitly requires it.
- Token efficiency matters: read the nearest relevant rule file first, prefer targeted file reads over broad searches, and avoid scanning large data trees under `examples/*/obs`, `examples/*/grids`, or `results/` unless the task truly needs them.

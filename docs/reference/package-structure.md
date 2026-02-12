---
layout: default
title: Package Structure
parent: Reference
nav_order: 1
---

# Package Structure

High-level map of the Python package.

## Top-level packages
- `openamundsen_da.core` - config merge, env helpers, runner/launcher, prior forcing, constants.
- `openamundsen_da.pipeline` - setup orchestration (`setup`), skeleton scaffolding (`setup_skeleton`), cleanup (`cleanup`).
- `openamundsen_da.observer` - observation preprocessing and summary tools.
- `openamundsen_da.methods` - assimilation methods, H(x), wet-snow tools, visualization.
- `openamundsen_da.io` - path and file discovery helpers.
- `openamundsen_da.util` - stats, time series, validation, landcover, ROI, parallel helpers.
- `openamundsen_da.batch` - subregion batch workflow (`oa-da-batch`).

## CLI entry points
Declared in `pyproject.toml` as `oa-da-*` scripts.

## Conventions
- `project.yml` contains project-wide openAMUNDSEN settings.
- `setup.yml` contains DA settings and setup time span.
- Steps live under `projects/setup_*/steps/step_*`.
- Members live under `.../ensembles/{prior,posterior}/member_*`.



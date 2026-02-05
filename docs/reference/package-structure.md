---
layout: default
title: Package Structure
parent: Reference
nav_order: 1
---

# Package Structure

High-level map of the Python package. Public entry points are linked to the CLI where possible; anything not listed is considered internal/unstable.

## Top-level packages

- `openamundsen_da.core` — configuration merge (`config`), environment helpers (`env`), runner/launcher (`runner`, `launch`), prior ensemble builder (`prior_forcing`), constants/logging (`constants`).
- `openamundsen_da.pipeline` — seasonal orchestration (`season`), skeleton scaffolding (`season_skeleton`), cleanup (`cleanup`).
- `openamundsen_da.observer` — observation prep: snow cover (`snowcover`, `satellite_scf`), wet snow (`wetsnow`, `satellite_wet_snow_s1`), plotting (`plot_fractions`, `plot_scf_summary`).
- `openamundsen_da.methods` — assimilation components: particle filter (`pf/*`), observation operators (`h_of_x/model_scf`), wet-snow helpers (`wet_snow/*`), visualization (`viz/*`).
- `openamundsen_da.io` — paths and file discovery (`paths`).
- `openamundsen_da.util` — utilities: stats, time series, validation, landcover masks, ROIs, parallel helpers.
- `openamundsen_da.batch` — subregion batch runner (prepare/run/merge/plot/pipeline). CLI: `oa-da-batch` (see Guides → CLI).

## CLI entry points (PyPI console scripts)

Declared in `pyproject.toml` with `oa-da-*` names; see Guides → CLI for usage.

## Imports vs. configuration

- Most user-facing workflows go through CLI/`python -m` modules; direct imports are stable only where noted below.
- Stable-ish helpers for scripting: `openamundsen_da.pipeline.season.cli`, `openamundsen_da.methods.pf.assimilate_scf.assimilate_scf_for_date`, `openamundsen_da.methods.h_of_x.model_scf.compute_model_scf`, `openamundsen_da.batch.pipeline.run_pipeline`.

## Internal conventions

- ROIs live under `project/env/roi.gpkg`; landcover grids auto-resolve from `project/grids/lc_<domain>_<res>.asc`.
- Observations expect `obs_scf_<PRODUCT>_YYYYMMDD.csv` and `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv` under `obs/season_*`.
- Ensemble member directories: `propagation/season_*/step_*/ensembles/{prior,posterior}/member_###`.

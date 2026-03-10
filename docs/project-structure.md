---
layout: default
title: Project Structure
nav_order: 3
---

# Project Structure
{: .no_toc }

Directory layout and configuration ownership.
{: .fs-6 .fw-300 }

## Repository Structure

```text
openamundsen_da/
|-- openamundsen_da/
|   |-- core/
|   |-- observer/
|   |-- methods/
|   |-- pipeline/
|   |-- util/
|   `-- io/
|-- templates/project/
|-- docs/
|-- tests/
|-- scripts/
`-- README.md
```

## Setup Data Structure

```text
setup/
|-- <setup-name>.yml
|-- env/
|   |-- roi.gpkg
|   `-- subdomains.gpkg          # optional; used by sub-domain mode
|-- grids/
|   |-- roi_<domain>_<resolution>.asc  # canonical ROI mask for data assimilation runs
|   `-- lc_<domain>_<resolution>.asc
|-- meteo/
|   |-- stations.csv
|   `-- <station>.csv
|-- obs/
|   |-- snowcover/
|   |-- wetsnow/
|   |-- project_YYYY-YYYY/
|   |   |-- scf_summary.csv
|   |   `-- wet_snow_summary.csv
|   `-- summaries/project_YYYY-YYYY/
|-- projects/
|   `-- project_YYYY-YYYY/
|       |-- project_YYYY-YYYY.yml
|       |-- subdomains/          # optional sub-domain mode workspace
|       |   |-- subdomain_manifest.json
|       |   `-- <subdomain_id>/
|       |-- results/             # project-level data assimilation outputs
|       |   |-- grids/da_output_grids.nc  # compact data assimilation grid summary (single + sub-domain)
|       |   |-- subdomain_overview.csv    # sub-domain mode run summary
|       |   |-- subdomain_assimilation_stats.csv
|       |   `-- subdomain_assimilation_aggregate.csv
|       |-- steps/
|       |   |-- step_00_init/
|       |   |   |-- step_00.yml
|       |   |   `-- ensembles/{prior,posterior}/
|       |   |-- step_01_YYYYMMDD-YYYYMMDD/
|       |   |   |-- step_01.yml
|       |   |   |-- assim/
|       |   |   |-- obs/
|       |   |   `-- ensembles/{prior,posterior}/
|       |   `-- ...
|       `-- plots/
`-- obs_selection.config.yml
```

## Configuration Files

### `<setup-name>.yml` (setup YAML)
Setup-wide and stable openAMUNDSEN configuration.
- Domain, CRS, resolution, timestep
- OA output variables and frequencies
- Environment paths and base model settings
- Observation class mappings and product tags under `obs.*`
- Canonical ROI grid naming follows openAMUNDSEN convention: `grids/roi_<domain>_<resolution>.asc` (generated from ROI vectors when missing)

No data assimilation orchestration keys should be placed here.

### `<project-name>.yml` / `project.yml` (project YAML)
Project-level data assimilation configuration and time span.
- `start_date`, `end_date`
- `data_assimilation.prior_forcing`
- `data_assimilation.h_of_x`
- `data_assimilation.likelihood`
- `data_assimilation.resampling`
- `data_assimilation.rejuvenation`
- `data_assimilation.restart`
- `data_assimilation.output` (retention mode; default compact)
- `data_assimilation.landcover_mask`
- `data_assimilation.assimilation_events`

### `step_XX.yml`
Auto-generated step window configuration.
- `start_date`, `end_date`
- `results_dir`

## Naming Glossary
- `setup`: global, stable OA config/data container
- `project`: one data assimilation configuration unit with its own time span
- `step`: one assimilation window inside a project
- `member`: one ensemble member
- `run`: execution of a project (event), not a config object

## Sub-domain merge behavior
- Sub-domain grid merge is a hard mosaic (no interpolation or blending across sub-domain borders).
- Visible breaks at sub-domain boundaries are expected and represent localized data assimilation behavior by design.
- Sub-domain point outputs are not merged at project root; they remain inside each sub-domain project.

## data assimilation compact grid variables
- `da_output_grids.nc` stores, per modeled grid variable `<var>`:
  `open_loop_<var>`, `ens_mean_<var>`, `ens_std_<var>`, `ens_min_<var>`, `ens_max_<var>`, `increment_<var>`.
- `increment_<var>` is defined as `ens_mean_<var> - open_loop_<var>`.
- Time coordinates in `da_output_grids.nc` cover the full project period across all step windows.

## File Naming
- SCF obs CSV: `obs_scf_<PRODUCT>_YYYYMMDD.csv`
- Wet-snow obs CSV: `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv`
- Weights CSV: `weights_<variable>_YYYYMMDD.csv`
- Resampling indices: `indices_YYYYMMDD.csv`
- Model state default: `model_state.pickle.gz` (configured in project YAML under `data_assimilation.restart`)

## Next Steps
- [Workflow Overview]({{ site.baseurl }}{% link workflow.md %})
- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %})
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments/index.md %})



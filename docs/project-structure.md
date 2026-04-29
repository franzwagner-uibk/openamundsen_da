---
layout: default
title: Project Structure
nav_order: 3
---

# Project Structure
{: .no_toc }

Directory layout and configuration ownership.
{: .fs-6 .fw-300 }

## Setup And Project Layout

The schematic below shows the directory layout of one setup and one project
workspace. It focuses on where setup-level inputs live and where a project keeps
its own configuration, per-step files, results, and plots.

![Setup and project layout showing setup-level inputs, observation summaries, project configuration, step folders, results, and plots]({{ site.baseurl }}/assets/images/diagrams/setup-project-structure.svg)

_Shared layout of a setup root and one project workspace. Optional sub-domain
folders are omitted for clarity._

## Configuration Files

### `<setup-name>.yml` (setup YAML)
Setup-wide and stable openAMUNDSEN configuration.
- Domain, CRS, resolution, timestep
- OA output variables and frequencies
- Environment paths and base model settings
- Canonical ROI grid naming follows openAMUNDSEN convention: `grids/roi_<domain>_<resolution>.asc` (generated from ROI vectors when missing)

No project-specific observation mapping or data assimilation orchestration keys
should be placed here.

### `<project-name>.yml` / `project.yml` (project YAML)
Project-level data assimilation configuration and time span.
- `start_date`, `end_date`
- `obs.*`
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

## DA Compact Grid Variables
- `da_output_grids.nc` stores, per modeled grid variable `<var>`: `open_loop_<var>`, `ens_mean_<var>`, `ens_std_<var>`, `ens_min_<var>`, `ens_max_<var>`, `increment_<var>`, `analysis_mean_<var>`, and `analysis_increment_<var>`.
- `increment_<var>` is the open-loop departure: `ens_mean_<var> - open_loop_<var>`.
- `analysis_increment_<var>` is the DA-event increment: `analysis_mean_<var> - ens_mean_<var>`, where `analysis_mean` is the event-weighted posterior mean. Positive values mean the event added snow/water to the ensemble mean.
- Analysis fields are populated on assimilation-event dates with matching weights and remain empty elsewhere on the project timeline.

## File Naming
- SCF obs CSV: `obs_scf_<PRODUCT>_YYYYMMDD.csv`
- Wet-snow obs CSV: `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv`
- Weights CSV: `weights_<variable>_YYYYMMDD.csv`
- Project plots/maps PDF collection: `results/reports/project_report.pdf`
- Resampling indices: `indices_YYYYMMDD.csv`
- Model state default: `model_state.pickle.gz` (configured in project YAML under `data_assimilation.restart`)

## Next Steps
- [Workflow Overview]({{ site.baseurl }}{% link workflow.md %})
- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %})
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments/index.md %})

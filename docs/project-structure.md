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

## Project Data Structure

```text
project/
|-- project.yml
|-- env/
|   `-- roi.gpkg
|-- grids/
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
|       |-- setup.yml
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

### `project.yml`
Project-wide and stable openAMUNDSEN configuration.
- Domain, CRS, resolution, timestep
- OA output variables and frequencies
- Environment paths and base model settings
- Observation class mappings and product tags under `obs.*`

No DA orchestration keys should be placed here.

### `setup.yml`
Setup-level DA configuration and time span.
- `start_date`, `end_date`
- `data_assimilation.prior_forcing`
- `data_assimilation.h_of_x`
- `data_assimilation.likelihood`
- `data_assimilation.resampling`
- `data_assimilation.rejuvenation`
- `data_assimilation.restart`
- `data_assimilation.landcover_mask`
- `data_assimilation.assimilation_events`

### `step_XX.yml`
Auto-generated step window configuration.
- `start_date`, `end_date`
- `results_dir`

## Naming Glossary
- `project`: global, stable OA config/data container
- `setup`: one DA configuration unit with its own time span
- `step`: one assimilation window inside a setup
- `member`: one ensemble member
- `run`: execution of a setup (event), not a config object

## File Naming
- SCF obs CSV: `obs_scf_<PRODUCT>_YYYYMMDD.csv`
- Wet-snow obs CSV: `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv`
- Weights CSV: `weights_<variable>_YYYYMMDD.csv`
- Resampling indices: `indices_YYYYMMDD.csv`
- Model state default: `model_state.pickle.gz` (configured in `setup.yml`)

## Next Steps
- [Workflow Overview]({{ site.baseurl }}{% link workflow.md %})
- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %})
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments/index.md %})



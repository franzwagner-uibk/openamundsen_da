---
layout: default
title: Running Experiments
parent: Guides
nav_order: 4
permalink: /guides/experiments/
---

# Running Experiments
{: .no_toc }

Fast path to run one data assimilation project inside a setup.
{: .fs-6 .fw-300 }

## Overview
1. Prepare the setup folder and ROI.
2. Add meteo forcing and observations.
3. Configure setup YAML and project YAML.
4. Preprocess observations.
5. Build the project skeleton and distribute observations.
6. Run the project pipeline.

## 1) Prepare Setup
Copy the template setup:

```bash
cp -r templates/project /path/to/your/project
```

Core layout:

```text
setup/
|-- setup.yml
|-- env/roi.gpkg
|-- grids/lc_<domain>_<resolution>.asc
|-- meteo/stations.csv
|-- obs/
|-- projects/project_YYYY-YYYY/project_YYYY-YYYY.yml
```

## 2) Data Inputs
- Meteo forcing in `meteo/`.
- Snow-cover rasters under `obs/snowcover/`.
- Optional wet-snow rasters under `obs/wetsnow/`.

## 3) Configure
Use the split explicitly:
- setup YAML (`setup.yml` or `<setup-name>.yml`): shared, pure openAMUNDSEN config.
- project YAML (`project_YYYY-YYYY.yml` or `project.yml`): data assimilation config, observation mapping, project dates and assimilation events.

Essentials in setup YAML:
- Domain/CRS/resolution/timestep.
- Required output variables for data assimilation (`swe`, `hs`, `lwc` daily grids).
- Shared input paths such as grids and meteo forcing.

Essentials in project YAML:
- `start_date` and `end_date`.
- Observation class mappings, product tags and summary paths under `obs.*`.
- `data_assimilation.prior_forcing.ensemble_size` and perturbation sigmas.
- `data_assimilation.h_of_x` configuration.
- `data_assimilation.likelihood`, `resampling`, `rejuvenation`, `restart`, `landcover_mask`.
- `data_assimilation.assimilation_events`.

## 4) Preprocess Observations
Snow-cover summary:

```bash
docker compose run --rm oa oa-da-snowcover \
  --input-dir /data/obs/snowcover \
  --project-label project_2019-2020 \
  --setup-dir /data
```

Wet-snow summary (optional):

```bash
docker compose run --rm oa oa-da-wetsnow \
  --input-dir /data/obs/wetsnow \
  --project-label project_2019-2020 \
  --setup-dir /data \
  --output-root /data/obs
```

## 5) Build Project Skeleton

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.project_skeleton \
  --setup-dir /data \
  --project-dir /data/projects/project_2019-2020
```

## 6) Distribute Observations to Steps

```bash
docker compose run --rm oa oa-da-scf \
  --project-dir /data/projects/project_2019-2020 \
  --summary-csv /data/obs/project_2019-2020/scf_summary.csv \
  --overwrite
```

```bash
docker compose run --rm oa oa-da-wetsnow-project \
  --project-dir /data/projects/project_2019-2020 \
  --summary-csv /data/obs/project_2019-2020/wet_snow_summary.csv \
  --overwrite
```

## 7) Run Project Pipeline

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.project \
  --setup-dir /data \
  --project-dir /data/projects/project_2019-2020 \
  --max-workers 8
```

Key outputs: per-step `assim/weights_*.csv`, optional `indices_*.csv`, compact grids under `results/grids/`, plots under `results/plots/`, maps under `results/maps/` and the optional PDF report under `results/reports/`.



---
layout: default
title: Running Experiments
parent: Guides
nav_order: 4
permalink: /guides/experiments/
---

# Running Experiments
{: .no_toc }

Fast path to run one setup in a project.
{: .fs-6 .fw-300 }

## Overview
1. Prepare the project folder and ROI.
2. Add meteo forcing and observations.
3. Configure `project.yml` and `setup.yml`.
4. Preprocess observations.
5. Build setup skeleton and distribute obs.
6. Run the setup pipeline.

## 1) Prepare Project
Copy the template project:

```bash
cp -r templates/project /path/to/your/project
```

Core layout:

```text
project/
|-- project.yml
|-- env/roi.gpkg
|-- grids/lc_<domain>_<resolution>.asc
|-- meteo/stations.csv
|-- obs/
|-- projects/project_YYYY-YYYY/setup.yml
```

## 2) Data Inputs
- Meteo forcing in `meteo/`.
- Snow-cover rasters under `obs/snowcover/`.
- Optional wet-snow rasters under `obs/wetsnow/`.

## 3) Configure
Use the split explicitly:
- `project.yml`: project-wide, pure openAMUNDSEN config.
- `setup.yml`: DA config under `data_assimilation` plus `start_date`/`end_date`.

Essentials in `project.yml`:
- Domain/CRS/resolution/timestep.
- Required output variables for DA (`swe`, `hs`, `lwc` daily grids).
- Observation class mappings and product tags under `obs.*`.

Essentials in `setup.yml`:
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
  --project-dir /data
```

Wet-snow summary (optional):

```bash
docker compose run --rm oa oa-da-wetsnow \
  --project-dir /data \
  --output /data/obs/project_2019-2020/wet_snow_summary.csv
```

## 5) Build Setup Skeleton

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.project_skeleton \
  --project-dir /data \
  --setup-dir /data/projects/project_2019-2020
```

## 6) Distribute Observations to Steps

```bash
docker compose run --rm oa oa-da-scf \
  --setup-dir /data/projects/project_2019-2020 \
  --summary-csv /data/obs/project_2019-2020/scf_summary.csv \
  --overwrite
```

```bash
docker compose run --rm oa oa-da-wetsnow-project \
  --setup-dir /data/projects/project_2019-2020 \
  --summary-csv /data/obs/project_2019-2020/wet_snow_summary.csv \
  --overwrite
```

## 7) Run Setup Pipeline

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.project \
  --project-dir /data \
  --setup-dir /data/projects/project_2019-2020 \
  --max-workers 8
```

Key outputs: per-step `assim/weights_*.csv`, optional `indices_*.csv`, and plots under `projects/<setup>/plots/`.





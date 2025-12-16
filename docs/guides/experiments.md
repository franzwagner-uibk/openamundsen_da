---
layout: default
title: Running Experiments
parent: Guides
nav_order: 4
---

# Running Experiments
{: .no_toc }

Fast path to run a seasonal data assimilation experiment.
{: .fs-6 .fw-300 }

---

## Overview

Minimal steps to get a season running:

1. Prepare the project folder and ROI.
2. Add meteo forcing and download observations.
3. Configure `project.yml`/`season.yml`.
4. Preprocess observations.
5. Build season skeleton and distribute obs.
6. Run the season pipeline (test, then full).

---

## Prerequisites

- Docker installed and running
- Repository cloned; image built (`docker build -t oa-da .`)
- `.env` configured with your paths

See the [Installation Guide]({{ site.baseurl }}{% link installation.md %}).

---

## 1) Prepare Project

- Copy `templates/project` to your workspace (contains `env/`, `meteo/`, `obs/`, `project.yml`).
- ROI polygon at `env/roi.gpkg` (single polygon; projected CRS; field `region_id` by default).
- Optional glacier mask at `env/glaciers.gpkg` (same CRS as ROI) if glacier masking is enabled.

---

## 2) Data Inputs

- Meteorological forcing: openAMUNDSEN station files (`meteo/stations.csv` + time series). See openAMUNDSEN input docs for details.
- MODIS MOD10A1 HDFs under `obs/MOD10A1_61_HDF/` (all tiles covering ROI).
- Optional Sentinel-2 Snowflake FSC rasters (GeoTIFF) under `obs/FSC_snowflake/`.
- Optional Sentinel-1 wet-snow masks (GeoTIFF) under `obs/WSM_S1_SAR/`.

---

## 3) Configure

Edit `project.yml` (essentials):

- Domain/CRS/resolution/timestep (`crs`, `resolution`, `timestep`).
- `data_assimilation.prior_forcing.ensemble_size` and perturbation sigmas.
- `data_assimilation.h_of_x` method/params for SCF.
- Observation errors: `likelihood.scf.obs_sigma` (and `wet_snow.obs_sigma` if used).
- Resampling threshold: `resampling.ess_threshold_ratio`.
- Glacier mask path if enabled.

Set season bounds in `propagation/<season>/season.yml` (`start_date`, `end_date`); leave `assimilation_dates: []` until observations are summarized.

---

## 4) Preprocess Observations

MODIS SCF summary (required):
```bash
docker compose run --rm oa oa-da-mod10a1 \
  --input-dir /data/obs/MOD10A1_61_HDF \
  --season-label season_2019-2020 \
  --project-dir /data \
  --ndsi-threshold 40
```

Optional Snowflake FSC summary:
```bash
docker compose run --rm oa \
  python -m openamundsen_da.observer.snowflake_fsc \
  --input-dir /data/obs/FSC_snowflake \
  --season-label season_2019-2020 \
  --project-dir /data
```

Optional Sentinel-1 wet-snow summary:
```bash
docker compose run --rm oa oa-da-wet-snow-s1 \
  --project-dir /data \
  --output /data/obs/season_2019-2020/wet_snow_summary.csv
```

---

## 5) Build Season Skeleton

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.season_skeleton \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020
```

Populate `assimilation_dates` in `season.yml` using the dates you want to assimilate (from `scf_summary.csv`, and optionally `wet_snow_summary.csv`).

---

## 6) Distribute Observations to Steps

SCF per-step files from `scf_summary.csv`:
```bash
docker compose run --rm oa oa-da-scf \
  --season-dir /data/propagation/season_2019-2020 \
  --summary-csv /data/obs/season_2019-2020/scf_summary.csv \
  --product MOD10A1 \
  --overwrite
```

Wet snow per-step files (optional):
```bash
docker compose run --rm oa oa-da-wet-snow-s1-season \
  --season-dir /data/propagation/season_2019-2020 \
  --summary-csv /data/obs/season_2019-2020/wet_snow_summary.csv \
  --overwrite
```

---

## 7) Run Season Pipeline

Smoke test (low parallelism):
```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.season \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020 \
  --max-workers 4
```

Full run (adjust `--max-workers` to your CPUs):
```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.season \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020 \
  --max-workers 8 \
  --monitor-perf
```

Key outputs: per-step `assim/weights_scf_YYYYMMDD.csv` (and `indices_*.csv` when resampling), plots under `propagation/<season>/plots/`.

---

## References

- Strasser, U., Warscher, M., Rottler, E., and Hanzer, F. (2024). openAMUNDSEN v1.0: an open-source snow-hydrological model for mountain regions. Geoscientific Model Development, 17, 6775-6797. https://doi.org/10.5194/gmd-17-6775-2024.
- Barella, R., Marin, C., Gianinetto, M., and Notarnicola, C. (2022). A novel approach to high resolution snow cover fraction retrieval in mountainous regions. IGARSS 2022 - IEEE International Geoscience and Remote Sensing Symposium, 3856-3859. https://doi.org/10.1109/IGARSS46834.2022.9884177.
- Nagler, T., Rott, H., Ripper, E., Bippus, G., and Hetzenecker, M. (2016). Advancements for snowmelt monitoring by means of Sentinel-1 SAR. Remote Sensing, 8(4), 348. https://doi.org/10.3390/rs8040348.
- Rottler, E., Warscher, M., Hanzer, F., and Strasser, U. (2024). Spatio-temporal wet snow dynamics from model simulations and remote sensing: a case study from the Rofental, Austria. Hydrological Processes, 38, e15279. https://doi.org/10.1002/hyp.15279.
- Cluzet, B., Magnusson, J., Queno, L., Mazzotti, G., Mott, R., and Jonas, T. (2024). Exploring how Sentinel-1 wet-snow maps can inform fully distributed physically based snowpack models. The Cryosphere, 18, 5753-5767. https://doi.org/10.5194/tc-18-5753-2024.

# openamundsen_da - Data Assimilation for openAMUNDSEN

Lightweight tools to build and run openAMUNDSEN ensembles and assimilate satellite snow cover fraction (SCF) with a particle filter. This README is Docker-only to ensure copy/pasteable, platform-independent usage on Windows/macOS/Linux.

This file uses plain ASCII to avoid encoding issues on Windows.

## Overview

- Goal: seasonal snow cover prediction with an ensemble openAMUNDSEN model and a particle filter. The model advances in time, satellite SCF updates the ensemble weights, and the posterior becomes the next prior.
- Status: prior ensemble building, ensemble launcher, MOD10A1 preprocessing, single-region SCF extraction, H(x) model SCF, assimilation weights, and plotting utilities.

Example project layout:

```
examples/test-project/
  project.yml                # project configuration
  env/                       # AOI etc. (e.g., GMBA_Inventory_*.gpkg)
  grids/                     # model grids
  meteo/                     # station CSVs + stations.csv (prior builder input)
  obs/
    MOD10A1_61_HDF/         # raw MODIS HDF input
    season_2017-2018/       # preprocessed GeoTIFFs + scf_summary.csv
  propagation/
    season_2017-2018/
      step_00_init/
        obs/
        ensembles/
          prior/
            open_loop/{meteo,results}
            member_XXX/{meteo,logs,results}
```

## General Information

- Per-member logs: `<member_dir>/logs/member.log`.
- Quote bind mounts like "${repo}:/workspace" to avoid PowerShell parsing issues.
- Use forward slashes for in-container paths (`/workspace`, `/data`).
- Prefer `--backend SVG` for plots on all platforms.

## Theory Primer (Very Short)

- Prior: An ensemble represents p(x_t) via N members created by perturbing forcings.
- Observation: Satellite SCF y_t with uncertainty sigma (quality-driven where available).
- H(x): Maps model outputs to SCF in the AOI (threshold or logistic).
- Likelihood: For each member i, L_i = N(y_t - H(x_t^i); 0, sigma^2).
- Weights: w_i = L_i / sum_j L_j (computed in log-space for stability).
- ESS: 1/sum(w_i^2) quantifies degeneracy; low ESS suggests resampling.
- Next steps (future): resampling and rejuvenation to form the posterior/next prior.

## Docker Quick Start

- Install Docker Desktop (Windows/macOS) or Docker Engine (Linux).
- On Windows, enable WSL2 and share the drive that contains your repo/data in Docker Desktop Settings.
- From repo root, build the image once:

```
docker build -t oa-da .
```

Editable run (Compose + .env):

```
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_results_ensemble `
  --step-dir /data/propagation/season_2017-2018/step_00_init `
  --ensemble prior `
  --time-col time `
  --var-col swe `
  --start-date 2017-11-01 `
  --end-date 2018-04-30 `
  --resample D `
  --rolling 1 `
  --title "Model Results Ensemble (prior)" `
  --backend SVG `
  --output-dir /data/propagation/season_2017-2018/step_00_init/assim/plots/results
```

Minimal files and keys used by the workflow.

- project.yml (at project root)

  - data_assimilation.prior_forcing (required)
    - ensemble_size: int
    - random_seed: int
    - sigma_t: float # temperature additive stddev (degC)
    - mu_p: float # log-space mean for precip factor
    - sigma_p: float # log-space stddev for precip factor
  - data_assimilation.h_of_x (optional defaults for H(x))
    - method: depth_threshold | logistic
    - variable: hs | swe
    - params.h0: float
    - params.k: float
  - likelihood (optional; assimilation)
    - obs_sigma: float (base sigma if no quality info)
    - sigma_floor: float (lower bound for sigma)
    - sigma_cloud_scale: float (extra sigma per unit cloud_fraction)
    - min_sigma: float (final clamp)
  - environment (optional)
    - Keys like GDAL_DATA/PROJ_LIB can be set; inside this Docker image they are auto-configured and usually not needed.

- season.yml (under the season directory)

  - start_date: YYYY-MM-DD
  - end_date: YYYY-MM-DD

- Step YAML (any \*.yml in the step directory)

  - start_date: 2017-10-01 00:00:00
  - end_date: 2018-09-30 23:59:59
  - scf (optional)
    - ndsi_threshold: 40
    - region_id_field: region_id
  - h_of_x (optional; same keys as in project.yml)

- Meteo input (under project meteo dir)

  - stations.csv (as required by openAMUNDSEN)
  - One CSV per station with at least column `date` (ISO). Columns `temp` and/or `precip` are optional; if present, `precip` must not contain negative values.

- AOI requirements
  - Vector file with exactly one polygon feature and a field `region_id` (or override name). CRS must match the raster/model outputs.

Tip: The launcher discovers YAMLs automatically: project.yml at project root; season.yml in the season folder; the first \*.yml in the step folder.

Templates

- Starter templates are included under `templates/project`. You can copy them
  into a new project folder and adapt:
  - `templates/project/project.yml`
  - `templates/project/propagation/season_YYYY-YYYY/season.yml`
  - `templates/project/propagation/season_YYYY-YYYY/step_00_init/step_00.yml`


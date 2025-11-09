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

## Docker Quick Start

- Install Docker Desktop (Windows/macOS) or Docker Engine (Linux).
- On Windows, enable WSL2 and share the drive that contains your repo/data in Docker Desktop Settings.
- From repo root, build the image once:

```
docker build -t oa-da .
```

Set paths once per session (Windows PowerShell example):

```
$repo = "C:\Daten\PhD\openamundsen_da"
$proj = "$repo\examples\test-project"
```

Editable install inside the container (run after you change the source code):

```
docker run --rm -it -v "${repo}:/workspace" -v "${proj}:/data" oa-da `
  python -m pip install -e /workspace --no-deps
```

Notes on updates:

- For code edits only: no image rebuild is needed. Re‑run the editable install command above.
- If you change dependencies, environment.yml, or the Dockerfile: rebuild the image (`docker build -t oa-da .`).

## Configuration Cheat Sheet

Minimal files and keys used by the workflow.

- project.yml (at project root)

  - data_assimilation.prior_forcing (required)
    - ensemble_size: int
    - random_seed: int
    - sigma_t: float # temperature additive stddev (°C)
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

## Workflow and Commands (Docker)

The commands below follow the project framework order: Build Ensemble -> Run Ensemble -> Observation Processing -> H(x) -> Assimilation -> Plots -> General Info.

### Build Ensemble (Prior Forcing)

Context

- Represents the prior distribution via an ensemble of meteorological forcings. Each member uses a constant additive temperature offset (Gaussian, sigma_t) and a constant multiplicative precipitation factor (lognormal, mu_p/sigma_p) across time and stations.
- The open-loop is an unperturbed copy used as a baseline for comparison.
- The step dates define an inclusive filter so all members share the same time window.

Required keys in `project.yml` (example):

```yaml
data_assimilation:
  prior_forcing:
    ensemble_size: 30
    random_seed: 42
    sigma_t: 0.5 # additive temperature stddev
    mu_p: 0.0 # log-space mean for precip factor
    sigma_p: 0.2 # log-space stddev for precip factor
```

Each step YAML defines the time window:

```yaml
start_date: 2017-10-01 00:00:00
end_date: 2018-09-30 23:59:59
```

Run the builder:

```
docker run --rm -it -v "${repo}:/workspace" -v "${proj}:/data" oa-da `
  python -m openamundsen_da.core.prior_forcing `
    --input-meteo-dir /data/meteo `
    --project-dir /data `
    --step-dir /data/propagation/season_2017-2018/step_00_init `
    --overwrite
```

Output under the step:

```
/data/propagation/season_2017-2018/step_00_init/ensembles/prior/
  open_loop/{meteo,results}
  member_001/{meteo,results}
  member_002/...
```

### Run Ensemble

Context

- Forecast step of the particle filter: propagate each prior member forward with openAMUNDSEN to produce model states and outputs (e.g., daily snow depth rasters).
- These outputs feed the observation operator H(x) for comparison against satellite SCF.

Launch openAMUNDSEN for all members of an ensemble (e.g., prior). Results land in each member's `results` directory.

```
docker run --rm -it -v "${repo}:/workspace" -v "${proj}:/data" oa-da `
  python -m openamundsen_da.core.launch `
    --project-dir /data `
    --season-dir  /data/propagation/season_2017-2018 `
    --step-dir    /data/propagation/season_2017-2018/step_00_init `
    --ensemble    prior `
    --max-workers 2 `
    --log-level   INFO `
    --overwrite
```

### Observation Processing

Context

- MOD10A1 preprocess converts HDF to analysis-ready GeoTIFFs and maintains a season summary CSV. A simple NDSI thresholding yields snow/no-snow classification, and SCF is the fraction of snow-classified pixels inside the AOI.
- Single-image SCF extraction computes an observed SCF for a specific date and AOI; this is the observation y_t for assimilation.
- Note: The Docker image includes the GDAL HDF4 plugin; if you ever rebuild dependencies, ensure hdf4 and libgdal-hdf4 are present.

- MOD10A1 preprocess (HDF -> GeoTIFF + season summary):

```
docker run --rm -it -v "${repo}:/workspace" -v "${proj}:/data" oa-da `
  python -m openamundsen_da.observer.mod10a1_preprocess `
    --input-dir   /data/obs/MOD10A1_61_HDF `
    --project-dir /data `
    --season-label season_2017-2018 `
    --aoi         /data/env/GMBA_Inventory_L8_15422.gpkg `
    --target-epsg 25832 `
    --resolution  500 `
    --max-cloud-fraction 0.1 `
    --overwrite
```

- Single-image SCF extraction (preprocessed GeoTIFF):

```
docker run --rm -it -v "${repo}:/workspace" -v "${proj}:/data" oa-da `
  python -m openamundsen_da.observer.satellite_scf `
    --raster  /data/obs/season_2017-2018/NDSI_Snow_Cover_20180110.tif `
    --region  /data/env/GMBA_Inventory_L8_15422.gpkg `
    --step-dir /data/propagation/season_2017-2018/step_00_init
```

- Plot SCF summary (SVG backend recommended):

```
docker run --rm -it -v "${repo}:/workspace" -v "${proj}:/data" oa-da `
  python -m openamundsen_da.observer.plot_scf_summary `
    /data/obs/season_2017-2018/scf_summary.csv `
    --output /data/obs/season_2017-2018/scf_summary.svg `
    --backend SVG
```

### Model SCF Operator (H(x))

Context

- H(x) maps model state/outputs into observation space. Here, it converts model snow depth (hs) or SWE into a probabilistic or deterministic SCF over the AOI to match the satellite product.
  - depth_threshold: I = 1 if X > h0 else 0; SCF = mean(I).
  - logistic: p = 1/(1 + exp(-k\*(X - h0))); SCF = mean(p). Parameters h0 and k are calibratable per site/region.

Derive model SCF within the AOI so it is comparable to satellite SCF.

Methods:

- depth_threshold (deterministic): indicator on X > h0, SCF = mean(I).
- logistic (probabilistic): p = 1/(1+exp(-k\*(X - h0))), SCF = mean(p).

```
docker run --rm -it -v "${repo}:/workspace" -v "${proj}:/data" oa-da `
  python -m openamundsen_da.methods.h_of_x.model_scf `
    --member-results /data/propagation/season_2017-2018/step_00_init/ensembles/prior/member_001/results `
    --aoi /data/env/GMBA_Inventory_L8_15422.gpkg `
    --date 2018-01-10 `
    --variable hs `
    --method depth_threshold
```

Optional configuration in `project.yml`:

```yaml
data_assimilation:
  h_of_x:
    method: logistic # or depth_threshold
    variable: hs # or swe
    params:
      h0: 0.05
      k: 80
```

### Assimilation (SCF Weights)

Context

- Bayesian update step: compute per-member likelihoods p(y_t | x_t^i) using a Gaussian error model with sigma for observation uncertainty, then normalize to weights w_t^i.
- Effective Sample Size (ESS) indicates weight degeneracy; low ESS suggests resampling. This module computes weights and ESS; resampling/rejuvenation can be applied in subsequent steps.

Compute Gaussian likelihood weights for a date by comparing observed SCF with H(x) across members. Outputs a CSV and reports ESS.

```
docker run --rm -it -v "${repo}:/workspace" -v "${proj}:/data" oa-da `
  python -m openamundsen_da.methods.pf.assimilate_scf `
    --project-dir /data `
    --step-dir    /data/propagation/season_2017-2018/step_00_init `
    --ensemble    prior `
    --date        2018-01-10 `
    --aoi         /data/env/GMBA_Inventory_L8_15422.gpkg
```

Notes:

- Observation CSV is auto-discovered at `/data/.../step_XX/obs/obs_scf_MOD10A1_YYYYMMDD.csv`.
- If available, `n_valid` and `cloud_fraction` inform sigma; otherwise a fixed sigma from `project.yml` is used.
- H(x) parameters (variable, method, h0, k) can be set under `h_of_x` in the step config and are reused.

### Plots

Context

- Weight plots visualize the normalized weights and residual distribution to diagnose degeneracy and sigma calibration.
- ESS timeline summarizes how informative observations are across dates and helps plan resampling frequency.

- Weights (single date, SVG):

```
docker run --rm -it -v "${repo}:/workspace" -v "${proj}:/data" oa-da `
  python -m openamundsen_da.methods.pf.plot_weights `
    /data/propagation/season_2017-2018/step_00_init/assim/weights_scf_20180110.csv `
    --output /data/propagation/season_2017-2018/step_00_init/assim/weights_scf_20180110.svg `
    --backend SVG
```

- ESS timeline (from all `weights_scf_*.csv`):

```
docker run --rm -it -v "${repo}:/workspace" -v "${proj}:/data" oa-da `
  python -m openamundsen_da.methods.pf.plot_ess_timeline `
    --step-dir /data/propagation/season_2017-2018/step_00_init `
    --normalized `
    --threshold 0.5 `
    --output /data/propagation/season_2017-2018/step_00_init/assim/ess_timeline.svg `
    --backend SVG
```

### Ensemble Visualization

- Forcing per-station (temperature + cumulative precip):

```
docker run --rm -it -v "${repo}:/workspace" -v "${proj}:/data" oa-da \
  python -m openamundsen_da.methods.viz.plot_forcing_ensemble \
    --step-dir /data/propagation/season_2017-2018/step_00_init \
    --ensemble prior \
    --time-col date --temp-col temp --precip-col precip \
    --start-date 2017-10-01 --end-date 2018-03-31 \
    --resample D --rolling 1 \
    --title "Forcing Ensemble (prior)" \
    --backend SVG \
    --output-dir /data/propagation/season_2017-2018/step_00_init/assim/plots/forcing
```

- Results per-point (SWE or snow*depth) from `point*\*.csv`:

```
docker run --rm -it -v "${repo}:/workspace" -v "${proj}:/data" oa-da \
  python -m openamundsen_da.methods.viz.plot_results_ensemble \
    --step-dir /data/propagation/season_2017-2018/step_00_init \
    --ensemble prior \
    --time-col time --var-col swe \
    --start-date 2017-10-01 --end-date 2018-03-31 \
    --resample D --rolling 1 \
    --title "Results Ensemble (prior, SWE)" \
    --backend SVG \
    --output-dir /data/propagation/season_2017-2018/step_00_init/assim/plots/results
```

### Help (no install needed)

Use PYTHONPATH with the repo mounted so modules can be imported without a prior editable install.

```
docker run --rm -it -e PYTHONPATH=/workspace -v "${repo}:/workspace" oa-da python -m openamundsen_da.core.prior_forcing --help
docker run --rm -it -e PYTHONPATH=/workspace -v "${repo}:/workspace" oa-da python -m openamundsen_da.core.launch --help
docker run --rm -it -e PYTHONPATH=/workspace -v "${repo}:/workspace" oa-da python -m openamundsen_da.observer.mod10a1_preprocess --help
docker run --rm -it -e PYTHONPATH=/workspace -v "${repo}:/workspace" oa-da python -m openamundsen_da.observer.satellite_scf --help
docker run --rm -it -e PYTHONPATH=/workspace -v "${repo}:/workspace" oa-da python -m openamundsen_da.observer.plot_scf_summary --help
docker run --rm -it -e PYTHONPATH=/workspace -v "${repo}:/workspace" oa-da python -m openamundsen_da.methods.h_of_x.model_scf --help
docker run --rm -it -e PYTHONPATH=/workspace -v "${repo}:/workspace" oa-da python -m openamundsen_da.methods.pf.assimilate_scf --help
docker run --rm -it -e PYTHONPATH=/workspace -v "${repo}:/workspace" oa-da python -m openamundsen_da.methods.pf.plot_weights --help
docker run --rm -it -e PYTHONPATH=/workspace -v "${repo}:/workspace" oa-da python -m openamundsen_da.methods.pf.plot_ess_timeline --help
```

## General Information

- Per-member logs: `<member_dir>/logs/member.log`.
- Quote bind mounts like `"${repo}:/workspace"` to avoid PowerShell parsing issues.
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

## Troubleshooting

- Python package not found in container when running help:
  - Use `-e PYTHONPATH=/workspace -v "${repo}:/workspace"` with `python -m ...`.
  - Or install editable inside the container for the current session.
- HDF not recognized in preprocess:
  - Rebuild the image to include `hdf4` and `libgdal-hdf4` (already in environment.yml). Verify: `gdalinfo --formats | findstr HDF4`.
- Windows bind mounts and file metadata:
  - Some operations (e.g., copy2) may fail to preserve times/permissions. The code falls back to a content-only copy automatically.
- Plots crash or produce no output on Windows:
  - Use `--backend SVG` to avoid backend/DLL issues.

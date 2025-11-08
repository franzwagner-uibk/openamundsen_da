# openamundsen_da — Data Assimilation for openAMUNDSEN

Lightweight tools to build and run openAMUNDSEN ensembles and assimilate satellite snow cover fraction (SCF) with a particle filter. This README is Docker-only to ensure copy/pasteable, platform‑independent usage on Windows/macOS/Linux.

This file uses plain ASCII to avoid encoding issues on Windows.

## Overview

- Goal: seasonal snow cover prediction with an ensemble openAMUNDSEN model and a particle filter. The model advances in time, satellite SCF updates the ensemble weights, and the posterior becomes the next prior.
- Status: prior ensemble building, ensemble launcher, MOD10A1 preprocessing, single‑region SCF extraction, H(x) model SCF, assimilation weights, and plotting utilities.

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

## Workflow and Commands (Docker)

The commands below follow the project framework order: Build Ensemble → Run Ensemble → Observation Processing → H(x) → Assimilation → Plots → General Info.

### Build Ensemble (Prior Forcing)

Required keys in `project.yml` (example):

```yaml
data_assimilation:
  prior_forcing:
    ensemble_size: 30
    random_seed: 42
    sigma_t: 0.5   # additive temperature stddev
    mu_p: 0.0      # log-space mean for precip factor
    sigma_p: 0.2   # log-space stddev for precip factor
```

Each step YAML defines the time window:

```yaml
start_date: 2017-10-01 00:00:00
end_date:   2018-09-30 23:59:59
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

Launch openAMUNDSEN for all members of an ensemble (e.g., prior). Results land in each member’s `results` directory.

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

- MOD10A1 preprocess (HDF → GeoTIFF + season summary):

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

Derive model SCF within the AOI so it is comparable to satellite SCF.

Methods:

- depth_threshold (deterministic): indicator on X > h0, SCF = mean(I).
- logistic (probabilistic): p = 1/(1+exp(-k·(X-h0))), SCF = mean(p).

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
    method: logistic   # or depth_threshold
    variable: hs       # or swe
    params:
      h0: 0.05
      k: 80
```

### Assimilation (SCF Weights)

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

### Help

```
docker run --rm oa-da python -m openamundsen_da.core.prior_forcing --help
docker run --rm oa-da python -m openamundsen_da.core.launch --help
docker run --rm oa-da python -m openamundsen_da.observer.mod10a1_preprocess --help
docker run --rm oa-da python -m openamundsen_da.observer.satellite_scf --help
docker run --rm oa-da python -m openamundsen_da.observer.plot_scf_summary --help
docker run --rm oa-da python -m openamundsen_da.methods.h_of_x.model_scf --help
docker run --rm oa-da python -m openamundsen_da.methods.pf.assimilate_scf --help
docker run --rm oa-da python -m openamundsen_da.methods.pf.plot_weights --help
docker run --rm oa-da python -m openamundsen_da.methods.pf.plot_ess_timeline --help
```

## General Information

- Per-member logs: `<member_dir>/logs/member.log`.
- Quote bind mounts like `"${repo}:/workspace"` to avoid PowerShell parsing issues.
- Use forward slashes for in-container paths (`/workspace`, `/data`).
- Prefer `--backend SVG` for plots on all platforms.


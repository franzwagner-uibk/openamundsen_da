# openamundsen_da — Data Assimilation for openAMUNDSEN

Lightweight tooling to run openAMUNDSEN ensembles and assimilate satellite snow cover fraction (SCF). It focuses on practical usage: launching ensemble members, preprocessing MODIS MOD10A1, extracting SCF for assimilation, and inspecting outputs/logs.

This README gives a concise overview, the basic workflow, and copy‑pasteable commands. It is not the full documentation and can be extended as the project evolves.

## Overview

- Goal: seasonal snow cover prediction using an ensemble openAMUNDSEN model and a particle filter. Throughout a season, the model predicts forward, observations provide SCF updates, and the posterior ensemble becomes the next prior.
- Status: prior ensemble building, ensemble launch orchestration, MOD10A1 preprocessing, single‑region SCF extraction, and plotting utilities are available. Likelihood, resampling, and rejuvenation steps are under active development.

### Workflow at a Glance

1) Initialize prior ensemble (perturb forcing/parameters; create member directories).
2) Predict: run openAMUNDSEN for each member over the next window.
3) Observe: preprocess MOD10A1 and extract SCF time series aligned to assimilation dates.
4) Update (WIP): compute likelihoods, reweight members, resample, rejuvenate.
5) Repeat for the next window.

## Project Layout

The repository includes an example project you can use to explore the workflow on Windows.

```
examples/test-project/
  project.yml                # project configuration (incl. prior_forcing block)
  env/                       # AOI etc. (e.g., GMBA_Inventory_*.gpkg)
  grids/                     # model grids
  meteo/                     # station CSVs + stations.csv (input for prior builder)
  obs/
    MOD10A1_61_HDF/         # raw MODIS HDF input
    season_2017-2018/       # preprocessed GeoTIFFs + scf_summary.csv (after preprocess)
  propagation/
    season_2017-2018/
      step_00_init/
        obs/                # assimilation inputs per step
        ensembles/
          prior/
            open_loop/
              meteo/        # date-filtered, unperturbed CSVs
              results/
            member_XXX/
              meteo/        # perturbed CSVs per member
              logs/member.log
              results/
```

Member results are written to `<member_dir>/results/` by default. You can optionally collect results under a single root via `--results-root` (see below).

## Quick Start (PowerShell on Windows)

Prerequisites:

- A conda or virtualenv with openAMUNDSEN and this package installed (`pip install -e .`).
- GDAL/PROJ available in the environment. `project.yml` can provide `GDAL_DATA` and `PROJ_LIB` (applied automatically by the launcher).

Define paths (adjust to your local clone):

```powershell
# Activate the environment first
conda activate openamundsen

$repo = "C:\\Daten\\PhD\\openamundsen_da"
$proj = "$repo\\examples\\test-project"
$seas = "$proj\\propagation\\season_2017-2018"
$step = "$seas\\step_00_init"
```

Launch a prior ensemble with 16–18 workers at INFO level:

```powershell
python -m openamundsen_da.core.launch `
  --project-dir $proj `
  --season-dir  $seas `
  --step-dir    $step `
  --ensemble    prior `
  --max-workers 18 `
  --log-level   INFO `
  --overwrite
```

- Parent progress appears in the console (start/finish per member + summary).
- Each worker writes its own log file to `<member_dir>\logs\member.log`.

Optional: collect results under a global root (members write to `<root>\member_XXX`).

```powershell
$resultsRoot = "D:\\oa_runs\\2025-11-04"
python -m openamundsen_da.core.launch `
  --project-dir $proj `
  --season-dir  $seas `
  --step-dir    $step `
  --ensemble    prior `
  --max-workers 18 `
  --results-root $resultsRoot `
  --log-level   INFO
```

Run single‑threaded with detailed logs:

```powershell
python -m openamundsen_da.core.launch `
  --project-dir $proj `
  --season-dir  $seas `
  --step-dir    $step `
  --ensemble    prior `
  --max-workers 1 `
  --log-level   DEBUG
```

Show help:

```powershell
python -m openamundsen_da.core.launch --help
```

## Observations: Single‑Image SCF Extraction

Compute SCF from one preprocessed MODIS/Terra MOD10A1 (C6/6.1) `NDSI_Snow_Cover` GeoTIFF and a single AOI polygon. Use the CLI entry point `oa-da-scf` or the module form.

```powershell
# Using the entry point
oa-da-scf --raster "C:\\data\\modis\\NDSI_Snow_Cover_20250315.tif" `
          --region "C:\\data\\modis\\region.gpkg" `
          --step-dir $step

# Or using python -m
python -m openamundsen_da.observer.satellite_scf `
  --raster  "C:\\data\\modis\\NDSI_Snow_Cover_20250315.tif" `
  --region  "C:\\data\\modis\\region.gpkg" `
  --step-dir $step
```

Input expectations:

- Raster: exported `NDSI_Snow_Cover` GeoTIFF (0..100), nodata applied; CRS must match AOI.
- AOI: exactly one polygon feature with field `region_id` (or configured field), same CRS as raster.
- Output: `<step>\obs\obs_scf_MOD10A1_YYYYMMDD.csv` with columns `date,region_id,scf`.

Overrides via step YAML (`step_XX.yml`):

```yaml
scf:
  ndsi_threshold: 35       # default 40
  region_id_field: region_id
```

Alternative output path:

```powershell
oa-da-scf --raster ... --region ... --output C:\\tmp\\myscf.csv
```

The tool fails if the AOI lacks the region field, has multiple features, or contains no valid NDSI pixels after masking.

## MOD10A1 Preprocess (HDF ➜ GeoTIFF + Summary)

Batch‑convert MODIS/Terra MOD10A1 (C6/6.1) HDF files into `NDSI_Snow_Cover_YYYYMMDD.tif` ready for SCF extraction, and maintain a season‑level `scf_summary.csv`.

```powershell
# Using the entry point
oa-da-mod10a1 `
  --input-dir   "$proj\obs\MOD10A1_61_HDF" `
  --project-dir "$proj" `
  --season-label season_2017-2018 `
  --aoi         "$proj\env\GMBA_Inventory_L8_15422.gpkg" `
  --target-epsg 25832 `
  --resolution  500 `
  --max-cloud-fraction 0.1 `
  --overwrite

# Or using python -m
python -m openamundsen_da.observer.mod10a1_preprocess `
  --input-dir   "$proj\obs\MOD10A1_61_HDF" `
  --project-dir "$proj" `
  --season-label season_2017-2018 `
  --aoi         "$proj\env\GMBA_Inventory_L8_15422.gpkg" `
  --target-epsg 25832 `
  --resolution  500 `
  --max-cloud-fraction 0.1 `
  --overwrite
```

Outputs (under `$proj\obs\season_yyyy-yyyy`):

- `NDSI_Snow_Cover_YYYYMMDD.tif` — reprojected/cropped GeoTIFF
- `NDSI_Snow_Cover_YYYYMMDD_class.tif` — 0=invalid, 1=no snow, 2=snow
- `scf_summary.csv` — `date,region_id,scf,cloud_fraction,source`

Notes:

- Use `--resolution` to set output pixel size (meters); omit for native.
- Use `--max-cloud-fraction` (0..1) to reject scenes dominated by cloud pixels (value 200).
- Use `--ndsi-threshold` to adjust the snow classification threshold (default 40).
- Envelope crop is default; add `--no-envelope` for exact polygon cutline.

## Plot SCF Summary

Render SCF over time from a `scf_summary.csv` as a small PNG.

```powershell
oa-da-plot-scf "$proj\obs\season_2017-2018\scf_summary.csv" `
  --output  "$proj\obs\season_2017-2018\scf_summary.png" `
  --title   "SCF 2017-2018" `
  --subtitle "derived from MODIS 10A1 v6 NDSI"
```

## Build Prior Forcing Ensemble (Standalone)

Create an open‑loop set and N perturbed members for a step. Dates are read from the step YAML; prior parameters live under `data_assimilation.prior_forcing` in `project.yml`.

Required keys in `project.yml` (example):

```yaml
data_assimilation:
  prior_forcing:
    ensemble_size: 30
    random_seed: 42
    sigma_t: 0.5   # additive temperature stddev
    mu_p:   0.0   # log-space mean for precip factor
    sigma_p: 0.2  # log-space stddev for precip factor
```

Step YAML must define the time window:

```yaml
start_date: 2017-10-01 00:00:00
end_date:   2018-09-30 23:59:59
```

Run the builder:

```powershell
$meteo = "$proj\meteo"   # original long-span meteo

python -m openamundsen_da.core.prior_forcing `
  --input-meteo-dir $meteo `
  --project-dir     $proj `
  --step-dir        $step `
  --overwrite
```

Output structure under the step:

```
<step>\ensembles\prior\
  open_loop\
    meteo\
    results\
  member_001\
    meteo\
    results\
    INFO.txt
  member_002\
    ...
```

Notes:

- CSV schema is strict: column `date` is required; `temp` and `precip` are optional.
- If a station file has `precip`, it must not contain negative values (aborts otherwise).
- Temperature and precipitation perturbations are stationary per member across stations/timesteps.

## Logs and Troubleshooting

- Per‑member logs: `<member_dir>\logs\member.log`. Tail a log:

```powershell
$log = "$step\ensembles\prior\member_0001\logs\member.log"
Get-Content $log -Tail 50 -Wait
```

- `--log-level` controls the parent launcher and passes through to openAMUNDSEN inside workers.
- Quote paths with spaces: `"C:\\path with spaces\\..."`.
- Environment variables for GDAL/PROJ and numeric threading are applied from `project.yml` when present.

## What’s Here (Modules)

- `openamundsen_da/core/launch.py` — orchestrates ensemble runs (fan‑out, logging).
- `openamundsen_da/core/prior_forcing.py` — builds open‑loop and perturbed meteo members.
- `openamundsen_da/observer/mod10a1_preprocess.py` — HDF ➜ GeoTIFF conversion + season summary.
- `openamundsen_da/observer/satellite_scf.py` — single‑image, single‑region SCF extraction.
- `openamundsen_da/observer/plot_scf_summary.py` — quick SCF time‑series plotter.

Roadmap (planned): observation operator H(x), likelihood, resampling, rejuvenation utilities under `methods/`.

---

Dev install:

```bash
pip install -e .
```


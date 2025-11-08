# openamundsen_da - Data Assimilation for openAMUNDSEN

Lightweight tooling to run openAMUNDSEN ensembles and assimilate satellite snow cover fraction (SCF). This README focuses on context and practical usage: launching ensemble members, preprocessing MOD10A1, extracting SCF for assimilation, computing weights, and inspecting outputs/logs.

This file uses plain ASCII and standard hyphens to avoid encoding issues on Windows.

## General Project Information

- Goal: seasonal snow cover prediction with an ensemble openAMUNDSEN model and a particle filter. Over the season, the model predicts forward, observations provide SCF updates, and the posterior becomes the next prior.
- Status: prior ensemble building, ensemble launch orchestration, MOD10A1 preprocessing, single-region SCF extraction, and plotting utilities are available. Likelihood, resampling, and rejuvenation are in progress.

Project layout (example):

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
        obs/
        ensembles/
          prior/
            open_loop/{meteo,results}
            member_XXX/{meteo,logs,results}
```

Dev install:

```powershell
cd C:\Daten\PhD\openamundsen_da
& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m pip install -e C:\Daten\PhD\openamundsen_da --no-deps
& C:\Users\franz\miniconda3\envs\gistools\python.exe -m pip install -e C:\Daten\PhD\openamundsen_da --no-deps
& C:\Users\franz\miniconda3\envs\oa-viz\python.exe -m pip install -e C:\Daten\PhD\openamundsen_da --no-deps
```

## Build Ensemble (Prior Forcing)

Create an open-loop set and N perturbed members for a step. Dates are read from the step YAML; prior parameters live under data_assimilation.prior_forcing in project.yml.

Required keys in project.yml (example):

```yaml
data_assimilation:
  prior_forcing:
    ensemble_size: 30
    random_seed: 42
    sigma_t: 0.5 # additive temperature stddev
    mu_p: 0.0 # log-space mean for precip factor
    sigma_p: 0.2 # log-space stddev for precip factor
```

Step YAML must define the time window:

```yaml
start_date: 2017-10-01 00:00:00
end_date: 2018-09-30 23:59:59
```

Run the builder (PowerShell):

```powershell
$repo = "C:\Daten\PhD\openamundsen_da"
$proj = "$repo\examples\test-project"
$seas = "$proj\propagation\season_2017-2018"
$step = "$seas\step_00_init"
$meteo = "$proj\meteo"

& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m openamundsen_da.core.prior_forcing `
  --input-meteo-dir $meteo `
  --project-dir     $proj `
  --step-dir        $step `
  --overwrite
```

Output structure under the step:

```
<step>\ensembles\prior\
  open_loop\{meteo,results}
  member_001\{meteo,results}\INFO.txt
  member_002\...
```

Notes:

- CSV schema is strict: column date is required; temp and precip are optional.
- If a station file has precip, it must not contain negative values (aborts otherwise).
- Temperature and precipitation perturbations are stationary per member across stations/timesteps.

## Run Ensemble

Launch openAMUNDSEN for all members of a chosen ensemble (e.g., prior). Member results go to <member_dir>\results by default; use --results-root to collect under a single directory.

```powershell
$repo = "C:\Daten\PhD\openamundsen_da"
$proj = "$repo\examples\test-project"
$seas = "$proj\propagation\season_2017-2018"
$step = "$seas\step_00_init"

& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m openamundsen_da.core.launch `
  --project-dir $proj `
  --season-dir  $seas `
  --step-dir    $step `
  --ensemble    prior `
  --max-workers 18 `
  --log-level   INFO `
  --overwrite
```

Global results root (optional):

```powershell
$resultsRoot = "D:\oa_runs\2025-11-04"
& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m openamundsen_da.core.launch `
  --project-dir $proj `
  --season-dir  $seas `
  --step-dir    $step `
  --ensemble    prior `
  --max-workers 18 `
  --results-root $resultsRoot `
  --log-level   INFO
```

Single-threaded debug run:

```powershell
& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m openamundsen_da.core.launch `
  --project-dir $proj `
  --season-dir  $seas `
  --step-dir    $step `
  --ensemble    prior `
  --max-workers 1 `
  --log-level   DEBUG
```

Help:

```powershell
& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m openamundsen_da.core.launch --help
```

## Observation Processing

### MOD10A1 Preprocess (HDF -> GeoTIFF + Summary)

Batch-convert MODIS/Terra MOD10A1 (C6/6.1) HDF files into NDSI_Snow_Cover_YYYYMMDD.tif and maintain a season-level scf_summary.csv.

```powershell
& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m openamundsen_da.observer.mod10a1_preprocess `
  --input-dir   "$proj\obs\MOD10A1_61_HDF" `
  --project-dir "$proj" `
  --season-label season_2017-2018 `
  --aoi         "$proj\env\GMBA_Inventory_L8_15422.gpkg" `
  --target-epsg 25832 `
  --resolution  500 `
  --max-cloud-fraction 0.1 `
  --overwrite
```

Outputs (under $proj\obs\season_yyyy-yyyy):

- NDSI_Snow_Cover_YYYYMMDD.tif — reprojected/cropped GeoTIFF
- NDSI_Snow_Cover_YYYYMMDD_class.tif — 0=invalid, 1=no snow, 2=snow
- scf_summary.csv — date,region_id,scf,cloud_fraction,source

Notes:

- Use --resolution (meters) and --max-cloud-fraction (0..1)
- Use --ndsi-threshold (default 40)
- Envelope crop is default; add --no-envelope for cutline

### Single-Image SCF Extraction

Compute SCF from one preprocessed NDSI_Snow_Cover_YYYYMMDD.tif and a single AOI polygon.

```powershell
& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m openamundsen_da.observer.satellite_scf `
  --raster  "$proj\obs\season_2017-2018\NDSI_Snow_Cover_20180110.tif" `
  --region  "$proj\env\GMBA_Inventory_L8_15422.gpkg" `
  --step-dir $step
```

Expectations and overrides:

- AOI: exactly one polygon with field region_id (same CRS as raster)
- Output: <step>\obs\obs_scf_MOD10A1_YYYYMMDD.csv
- Step YAML overrides (step_XX.yml):

```yaml
scf:
  ndsi_threshold: 35
  region_id_field: region_id
```

Custom output path:

```powershell
& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m openamundsen_da.observer.satellite_scf --raster ... --region ... --output C:\tmp\myscf.csv
```

### Plot SCF Summary

```powershell
& C:\Users\franz\miniconda3\envs\gistools\python.exe -m openamundsen_da.observer.plot_scf_summary "$proj\obs\season_2017-2018\scf_summary.csv" `
  --output  "$proj\obs\season_2017-2018\scf_summary.png" `
  --title   "SCF 2017-2018" `
  --subtitle "derived from MODIS 10A1 v6 NDSI"
```

## Model SCF Operator (H(x))

Purpose

- Derive model-based Snow Cover Fraction (SCF) from openAMUNDSEN outputs within an AOI to match satellite SCF for data assimilation.

Inputs

- Member results directory: contains daily rasters snowdepth_daily_YYYY-MM-DDT0000.tif and/or swe_daily_YYYY-MM-DDT0000.tif.
- AOI polygon: single feature vector file; reprojected to raster CRS if needed.
- Date: YYYY-MM-DD.

Methods

- Depth threshold (deterministic)
  - Per-cell indicator: I = 1 if X > h0 else 0; SCF = mean(I).
  - h0 is in the same units as X (m for HS, or SWE units if using SWE).
- Logistic (probabilistic)
  - Per-cell probability: p = 1 / (1 + exp(-k \* (X - h0))); SCF = mean(p).
  - h0 is the 50% point; k controls sharpness (1/units of X).

Output

- CSV per member/date: model_scf_YYYYMMDD.csv in the member results directory.
- Columns: date, member_id, region_id, variable, method, h0, k, n_valid, scf_model, raster.

CLI

```powershell
& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m openamundsen_da.methods.h_of_x.model_scf `
  --member-results "$step\ensembles\prior\member_001\results" `
  --aoi "$proj\env\GMBA_Inventory_L8_15422.gpkg" `
  --date 2018-01-10 `
  --variable hs `
  --method logistic
```

Configuration (optional)

```yaml
data_assimilation:
  h_of_x:
    method: logistic # or depth_threshold
    variable: hs # or swe
    params:
      h0: 0.05
      k: 80
```

## Assimilation (SCF Weights)

Compute Gaussian weights for one assimilation date by comparing the observed SCF with model-derived SCF (H(x)) for all members. Outputs a CSV with per-member weights and reports ESS.

```powershell
& C:\Users\franz\miniconda3\envs\openamundsen\python.exe -m openamundsen_da.methods.pf.assimilate_scf `
  --project-dir $proj `
  --step-dir    $step `
  --ensemble    prior `
  --date        2018-01-10 `
  --aoi         "$proj\env\GMBA_Inventory_L8_15422.gpkg"
```

Notes:

- Observation CSV is auto-discovered at <step>\obs\obs_scf_MOD10A1_YYYYMMDD.csv (from the SCF tool).
- If present, n_valid and cloud_fraction are used to set sigma via a binomial + quality model; otherwise a fixed sigma from project.yml is used.
- H(x) parameters (variable, method, h0, k) can be set in the step YAML under h_of_x and are reused here.

### Plot Weights (single date)

```powershell
& C:\\Users\\franz\\miniconda3\\envs\\perturb\\python.exe -m openamundsen_da.methods.pf.plot_weights `
  "$step\\assim\\weights_scf_20180110.csv" `
  --output "$step\\assim\\weights_scf_20180110.png" `
  --title  "SCF Weights 2018-01-10" `
  --subtitle "prior ensemble vs observed SCF"
```

### Plot ESS timeline (multiple dates)

```powershell
& C:\\Users\\franz\\miniconda3\\envs\\oa-viz\\python.exe -m openamundsen_da.methods.pf.plot_ess_timeline `
  --step-dir "$step" `
  --normalized `
  --threshold 0.5 `
  --output "$step\\assim\\ess_timeline.png"
```

## General Information (Logging, Environment, Tips)

- Per-member logs: <member_dir>\logs\member.log. Tail a log:

```powershell
$log = "$step\ensembles\prior\member_0001\logs\member.log"
Get-Content $log -Tail 50 -Wait
```

- --log-level controls the parent launcher and passes through to openAMUNDSEN inside workers.
- Quote paths with spaces: "C:\path with spaces\...".
- Environment variables for GDAL/PROJ and numeric threading are applied from project.yml when present.

## Modules

- openamundsen_da/core/launch.py - orchestrates ensemble runs (fan-out, logging)
- openamundsen_da/core/prior_forcing.py - builds open-loop and perturbed meteo members
- openamundsen_da/observer/mod10a1_preprocess.py - HDF to GeoTIFF + season summary
- openamundsen_da/observer/satellite_scf.py - single-image, single-region SCF extraction
- openamundsen_da/observer/plot_scf_summary.py - SCF time-series plotter
- openamundsen_da/methods/h_of_x/model_scf.py - model-derived SCF operator and CLI
- openamundsen_da/methods/pf/assimilate_scf.py - Gaussian likelihood weights for SCF

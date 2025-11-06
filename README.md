# openamundsen_da — Data Assimilation for openAMUNDSEN

Lightweight tooling to run openAMUNDSEN ensembles and assimilate satellite snow cover fraction (SCF). It focuses on practical usage: launching ensemble members, preprocessing MODIS MOD10A1, extracting SCF for assimilation, and inspecting outputs/logs.

This README is structured along the project workflow and can be extended over time.

## General Project Information

- Goal: seasonal snow cover prediction with an ensemble openAMUNDSEN model and a particle filter. Over the season, the model predicts forward, observations provide SCF updates, and the posterior becomes the next prior.
- Status: prior ensemble building, ensemble launch orchestration, MOD10A1 preprocessing, single‑region SCF extraction, and plotting utilities are available. Likelihood, resampling, and rejuvenation are in progress.

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

```bash
pip install -e .
```

## Build Ensemble (Prior Forcing)

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

Run the builder (PowerShell):

```powershell
conda activate openamundsen

$repo = "C:\\Daten\\PhD\\openamundsen_da"
$proj = "$repo\\examples\\test-project"
$seas = "$proj\\propagation\\season_2017-2018"
$step = "$seas\\step_00_init"
$meteo = "$proj\\meteo"   # original long-span meteo

python -m openamundsen_da.core.prior_forcing `
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

- CSV schema is strict: column `date` is required; `temp` and `precip` are optional.
- If a station file has `precip`, it must not contain negative values (aborts otherwise).
- Temperature and precipitation perturbations are stationary per member across stations/timesteps.

## Run Ensemble

Launch openAMUNDSEN for all members of a chosen ensemble (e.g., prior). Member results go to `<member_dir>\results\` by default; use `--results-root` to collect under a single directory.

```powershell
conda activate openamundsen

$repo = "C:\\Daten\\PhD\\openamundsen_da"
$proj = "$repo\\examples\\test-project"
$seas = "$proj\\propagation\\season_2017-2018"
$step = "$seas\\step_00_init"

python -m openamundsen_da.core.launch `
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

Single‑threaded debug run:

```powershell
python -m openamundsen_da.core.launch `
  --project-dir $proj `
  --season-dir  $seas `
  --step-dir    $step `
  --ensemble    prior `
  --max-workers 1 `
  --log-level   DEBUG
```

Help:

```powershell
python -m openamundsen_da.core.launch --help
```

## Observation Processing

### MOD10A1 Preprocess (HDF ➜ GeoTIFF + Summary)

Batch‑convert MODIS/Terra MOD10A1 (C6/6.1) HDF files into `NDSI_Snow_Cover_YYYYMMDD.tif` and maintain a season‑level `scf_summary.csv`.

```powershell
oa-da-mod10a1 `
  --input-dir   "$proj\obs\MOD10A1_61_HDF" `
  --project-dir "$proj" `
  --season-label season_2017-2018 `
  --aoi         "$proj\env\GMBA_Inventory_L8_15422.gpkg" `
  --target-epsg 25832 `
  --resolution  500 `
  --max-cloud-fraction 0.1 `
  --overwrite

# or
python -m openamundsen_da.observer.mod10a1_preprocess ...
```

Outputs (under `$proj\obs\season_yyyy-yyyy`):

- `NDSI_Snow_Cover_YYYYMMDD.tif` — reprojected/cropped GeoTIFF
- `NDSI_Snow_Cover_YYYYMMDD_class.tif` — 0=invalid, 1=no snow, 2=snow
- `scf_summary.csv` — `date,region_id,scf,cloud_fraction,source`

Notes:

- Use `--resolution` (meters) and `--max-cloud-fraction` (0..1)
- Use `--ndsi-threshold` (default 40)
- Envelope crop is default; add `--no-envelope` for cutline

### Single‑Image SCF Extraction

Compute SCF from one preprocessed `NDSI_Snow_Cover_YYYYMMDD.tif` and a single AOI polygon.

```powershell
oa-da-scf --raster "C:\\data\\modis\\NDSI_Snow_Cover_20250315.tif" `
          --region "C:\\data\\modis\\region.gpkg" `
          --step-dir $step

# or
python -m openamundsen_da.observer.satellite_scf ...
```

Expectations and overrides:

- AOI: exactly one polygon with field `region_id` (same CRS as raster)
- Output: `<step>\obs\obs_scf_MOD10A1_YYYYMMDD.csv`
- Step YAML overrides (`step_XX.yml`):

```yaml
scf:
  ndsi_threshold: 35
  region_id_field: region_id
```

Custom output path:

```powershell
oa-da-scf --raster ... --region ... --output C:\\tmp\\myscf.csv
```

### Plot SCF Summary

```powershell
oa-da-plot-scf "$proj\obs\season_2017-2018\scf_summary.csv" `
  --output  "$proj\obs\season_2017-2018\scf_summary.png" `
  --title   "SCF 2017-2018" `
  --subtitle "derived from MODIS 10A1 v6 NDSI"
```

## General Information (Logging, Environment, Tips)

- Per‑member logs: `<member_dir>\logs\member.log`. Tail a log:

```powershell
$log = "$step\ensembles\prior\member_0001\logs\member.log"
Get-Content $log -Tail 50 -Wait
```

- `--log-level` controls the parent launcher and passes through to openAMUNDSEN inside workers.
- Quote paths with spaces: `"C:\\path with spaces\\..."`.
- Environment variables for GDAL/PROJ and numeric threading are applied from `project.yml` when present.

## Modules

- `openamundsen_da/core/launch.py` — orchestrates ensemble runs (fan‑out, logging)
- `openamundsen_da/core/prior_forcing.py` — builds open‑loop and perturbed meteo members
- `openamundsen_da/observer/mod10a1_preprocess.py` — HDF ➜ GeoTIFF + season summary
- `openamundsen_da/observer/satellite_scf.py` — single‑image, single‑region SCF extraction
- `openamundsen_da/observer/plot_scf_summary.py` — SCF time‑series plotter

Roadmap (planned): observation operator H(x), likelihood, resampling, rejuvenation under `methods/`.

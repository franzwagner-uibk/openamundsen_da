# openamundsen_da — Data Assimilation for openAMUNDSEN

Lightweight tools to build and run openAMUNDSEN ensembles and assimilate satellite snow cover fraction (SCF) with a particle filter. Commands are Docker/Compose friendly and use generic variables to work across projects.

## Overview

- Seasonal snow cover prediction with an ensemble model + particle filter.
- Includes prior forcing builder, ensemble launcher, MOD10A1 preprocessing, SCF extraction, H(x) model SCF, assimilation, resampling, rejuvenation, and plotting utilities.

## Setup

- Install Docker Desktop (Windows/macOS) or Docker Engine (Linux).
- Build the image once from repo root: `docker build -t oa-da .`
- Copy `.env.example` to `.env` and edit:
  - `REPO` = path to this repo on your machine
  - `PROJ` = path to your project data
  - Optional: `CPUS`, `MEMORY`, `MAX_WORKERS`
- Set Compose compatibility if needed: `setx COMPOSE_COMPATIBILITY 1` (Windows) or `export COMPOSE_COMPATIBILITY=1` (Linux/macOS).
- Volumes: `${REPO}` → `/workspace`, `${PROJ}` → `/data`.

## Project Variables

Define once per shell and reuse in all commands:

```powershell
$project = "/data"                                # in-container project root
$season  = "$project/propagation/season_YYYY-YYYY" # season folder
$step    = "$season/step_XX_name"                  # current step
$date    = "YYYY-MM-DD"                            # assimilation date
$dateTag = ($date -replace '-', '')
$aoi     = "$project/env/your_aoi.gpkg"            # single-feature AOI
```

Notes
- Use forward slashes in paths (`/workspace`, `/data`).
- Optional flags are listed under each command; examples show only required flags.

## Workflow/Commands

### Prior Forcing (build ensemble)

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.core.prior_forcing `
  --input-meteo-dir $project/meteo `
  --project-dir $project `
  --step-dir $step
```

Optional: `--overwrite`, `--log-level <LEVEL>`

### Run Ensemble

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.core.launch `
  --project-dir $project `
  --season-dir $season `
  --step-dir $step `
  --ensemble prior
```

Optional: `--max-workers <N>`, `--overwrite`, `--restart-from-state`, `--dump-state`, `--state-pattern <glob>`, `--log-level <LEVEL>`

### Observation Processing

- MOD10A1 preprocess (HDF → GeoTIFF + season summary):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.mod10a1_preprocess `
  --input-dir $project/obs/MOD10A1_61_HDF `
  --season-label season_YYYY-YYYY
```

Optional: `--project-dir $project`, `--aoi $aoi`, `--aoi-field <field>`, `--target-epsg <code>`, `--resolution <m>`, `--ndsi-threshold <val>`, `--no-envelope`, `--no-recursive`, `--overwrite`, `--log-level <LEVEL>`

- Single-image SCF extraction (GeoTIFF → obs CSV):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.satellite_scf `
  --raster $project/obs/season_YYYY-YYYY/NDSI_Snow_Cover_YYYYMMDD.tif `
  --region $aoi `
  --step-dir $step
```

Optional: `--output <csv>`, `--ndsi-threshold <val>`, `--log-level <LEVEL>`

### H(x) Model SCF (optional, per-member debug)

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.h_of_x.model_scf `
  --member-results $step/ensembles/prior/member_001/results `
  --aoi $aoi `
  --date $date
```

Optional: `--variable hs|swe`, `--method depth_threshold|logistic`, `--log-level <LEVEL>`

### Assimilation (SCF weights)

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.assimilate_scf `
  --project-dir $project `
  --step-dir $step `
  --ensemble prior `
  --date $date `
  --aoi $aoi
```

Optional: `--obs-csv <path>`, `--output <csv>`, `--log-level <LEVEL>`

### Resampling (posterior ensemble)

```powershell
$dateTag = ($date -replace '-', '')
$weights = "$step/assim/weights_scf_$dateTag.csv"

docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.resample `
  --project-dir $project `
  --step-dir $step `
  --ensemble prior `
  --weights $weights `
  --target posterior
```

Optional: `--ess-threshold-ratio <0..1>`, `--ess-threshold <n|ratio>`, `--seed <int>`, `--overwrite`, `--log-level <LEVEL>`

### Rejuvenation (posterior → prior)

Rebase is default (perturbations are applied relative to open_loop). If rejuvenation sigmas are not set, they fall back to prior_forcing sigmas.

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.rejuvenate `
  --project-dir $project `
  --prev-step-dir $season/step_XX_prev `
  --next-step-dir $season/step_YY_next
```

Optional: `--source-meteo-dir <path>`, `--log-level <LEVEL>`

Project config (example):

```yaml
data_assimilation:
  rejuvenation:
    sigma_t: 0.2
    sigma_p: 0.2
```

## Plots

- Forcing per-station:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_forcing_ensemble `
  --step-dir $step `
  --ensemble prior
```

Optional: `--time-col`, `--temp-col`, `--precip-col`, `--start-date`, `--end-date`, `--resample`, `--rolling`, `--hydro-month`, `--hydro-day`, `--title`, `--subtitle`, `--output-dir`, `--backend`, `--log-level`

- Results per-station:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_results_ensemble `
  --step-dir $step `
  --ensemble prior
```

Optional: `--time-col`, `--var-col`, `--var-label`, `--var-units`, `--start-date`, `--end-date`, `--resample`, `--resample-agg`, `--rolling`, `--band-low`, `--band-high`, `--title`, `--subtitle`, `--output-dir`, `--backend`, `--log-level`

- ESS timeline:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.plot_ess_timeline `
  --step-dir $step
```

Optional: `--normalized`, `--threshold <ratio>`, `--output <svg>`, `--backend`, `--log-level`

Outputs are written to `$season/plots/forcing` and `$season/plots/results` with the season identifier in filenames.

## Season Pipeline

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.pipeline.season `
  --project-dir $project `
  --season-dir $season
```

Optional: `--max-workers <N>`, `--overwrite`, `--log-level <LEVEL>`

Outputs
- Per-step runs in `<step>/ensembles/{prior,posterior}` (open_loop + members)
- Weights and indices in `<step>/assim/`
- Rejuvenated next-step prior (members + open_loop with state_pointer.json)
- Season plots under `<season_dir>/plots/{forcing,results}`

## Troubleshooting

- Plots on Windows: use `--backend SVG`.
- HDF not recognized: ensure HDF4 support is present; check `gdalinfo --formats | findstr HDF4`.
- Windows bind mounts may drop metadata; code falls back to content-only copies.
- Package import in container: Compose sets `PYTHONPATH=/workspace`.

## Logging

- All commands accept `--log-level`.
- Internally uses loguru with the standard format in `openamundsen_da/core/constants.py`.

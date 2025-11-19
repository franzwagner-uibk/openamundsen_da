# openamundsen_da â€” Data Assimilation for openAMUNDSEN

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
- Volumes: `${REPO}` â†’ `/workspace`, `${PROJ}` â†’ `/data`.

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

## Required Project Structure

This repo expects your project to follow the fixed layout shown below. Commands derive everything from these paths, so no CLI flag is needed for files/directories that live under the structure.

```
project/
├── env/
│   └── <single AOI vector, e.g., my_region.gpkg>
├── meteo/
│   ├── stations.csv
│   └── <station>.csv  (long-span forcing inputs)
├── propagation/
│   └── season_YYYY-YYYY/
│       ├── season.yml  (season metadata, dates)
│       ├── step_XX_name/
│       │   ├── step_XX_name.yml  (dates, h_of_x overrides discouraged)
│       │   └── ensembles/
│       │       ├── prior/  (created by season.py; contains member_<NNN>)
│       │       └── posterior/  (produced by resampling)
│       └── ... additional steps ...
├── obs/
│   └── season_YYYY-YYYY/
│       └── obs_scf_MOD10A1_YYYYMMDD.csv
└── project.yml  (contains data_assimilation.h_of_x, resampling, etc.)
```

- `project.yml` must define `data_assimilation.h_of_x` (used by `model_scf` + `assimilate_scf`) and the DA blocks referenced by the pipeline.
- `propagation/season_X/step_Y/ensembles/prior` is created automatically by `season.py` (using `${project}/meteo` for forcing); you only need to ensure the step YAMLs and meteorological inputs exist.
- Observations (MODIS preprocessed GeoTIFFs → CSV) live under `obs/season_X`; the pipeline assumes the CSVs follow `obs_scf_MOD10A1_YYYYMMDD.csv`.
- AOI vector: `env/*.gpkg` (single feature) is mandatory for every command that masks spatial data.

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

- MOD10A1 preprocess (HDF -> GeoTIFF + season summary):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.mod10a1_preprocess `
  --input-dir $project/obs/MOD10A1_61_HDF `
  --season-label season_YYYY-YYYY
```

Optional: --project-dir $project, --aoi $aoi, --aoi-field <field>, --target-epsg <code>, --resolution <m>, --ndsi-threshold <val>, --no-envelope, --no-recursive, --overwrite, --log-level <LEVEL>

- Single-image SCF extraction (GeoTIFF â†’ obs CSV):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.satellite_scf `
  --raster $project/obs/season_YYYY-YYYY/NDSI_Snow_Cover_YYYYMMDD.tif `
  --region $aoi `
  --step-dir $step
```

Optional: `--output <csv>`, `--ndsi-threshold <val>`, `--log-level <LEVEL>`

- Season batch mode (turns every raster in `obs/season_YYYY-YYYY` into the per-step CSVs):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.satellite_scf `
  --season-dir $season `
  --summary-csv $project/obs/season_YYYY-YYYY/scf_summary.csv `
  --overwrite
```

Optional: --overwrite, --log-level <LEVEL> (the summary path defaults to <project>/obs/<season>/scf_summary.csv). No AOI argument is required because the CSV already stores the AOI-derived SCF stats for each date.

Batch mode walks `propagation/season_YYYY-YYYY/step_*`, matches each raster by date to its step (or the step whose `end_date` matches the raster date), and writes `obs_scf_MOD10A1_YYYYMMDD.csv` into `<step>/obs`. Per-step `scf` overrides still apply.

Alternatively, skip reprocessing entirely by driving the season mode from the `scf_summary.csv` produced by `mod10a1_preprocess`. It copies each summary row for an assimilation date into the matching `<step>/obs/` file, so you only need:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.satellite_scf `
  --season-dir $season `
  --summary-csv $project/obs/season_YYYY-YYYY/scf_summary.csv `
  --overwrite
```

Optional: `--overwrite`, `--log-level <LEVEL>` (the summary path defaults to `<project>/obs/<season>/scf_summary.csv`). No AOI argument is required because the CSV already stores the AOI-derived SCF stats for each date.

Note: the summary-based workflow is the recommended way to prepare SCF observations for assimilation; the single-image and raster batch modes are kept for backward compatibility only.

### H(x) Model SCF (optional, per-member debug)

```powershell
  docker compose run --rm oa `
    python -m openamundsen_da.methods.h_of_x.model_scf `
    --project-dir $project `
    --member-results $step/ensembles/prior/member_001/results `
    --aoi $aoi `
    --date $date
```

Model parameters (`variable`, `method`, `h0`, `k`) are now read strictly from `project.yml` under `data_assimilation.h_of_x`, so the CLI no longer accepts overrides; providing `--project-dir` ensures the command uses the same configuration as the rest of the pipeline.

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

### Rejuvenation (posterior â†’ prior)

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

- Season-wide forcing/results (stitch all steps together):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_season_ensemble `
  forcing `
  --season-dir $season
```

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_season_ensemble `
  results `
  --season-dir $season `
  --var-col swe
```

Optional: `--station`, `--max-stations`, `--start-date`, `--end-date`, `--resample`, `--rolling`, `--hydro-month`, `--hydro-day`, `--backend`, `--log-level`, `--var-label`, `--var-units`, `--band-low`, `--band-high`.

Note: running the season pipeline (see below) also generates these season plots automatically under `<season_dir>/plots/{forcing,results}`.

## Season Pipeline

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.pipeline.season `
  --project-dir $project `
  --season-dir $season
```

The launcher automatically pulls the initial forcing from `$project/meteo` and builds the first prior ensemble (errors if the directory is missing), so you no longer need a separate `prior_forcing` run before `season.py` as long as the long-span station files live under `project/meteo`.

Optional: `--max-workers <N>`, `--overwrite`, `--log-level <LEVEL>`

The pipeline drives each step in order, assimilates SCF on the _next_ step's start date, resamples the resulting weights to the posterior, and rejuvenates that posterior into the next prior before proceeding. Assimilation looks for the single-row CSV `obs_scf_MOD10A1_YYYYMMDD.csv` inside `<step>/obs/` for the date being processed; generate those files with `openamundsen_da.observer.satellite_scf` after you preprocess the MOD10A1 NDSI raster for your AOI (projection, QA/masking, and mosaicking). `season.py` never reads raw imagery, so the CSV must already reflect any filtering or thresholding you want applied.

Outputs

- Per-step runs in `<step>/ensembles/{prior,posterior}` (open_loop + members)
- Weights and indices in `<step>/assim/`
- Rejuvenated next-step prior (members + open_loop with state_pointer.json)
- Season plots under `<season_dir>/plots/{forcing,results}`

### Season Skeleton (optional helper)

To create an empty season layout with `step_*` folders and minimal step YAMLs from a list of assimilation dates, add them to `season.yml`:

```yaml
start_date: 2017-10-01
end_date: 2018-09-30
assimilation_dates:
  - 2017-11-23
  - 2017-12-24
  - 2018-01-30
  # ...
```

Then run:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.pipeline.season_skeleton `
  --project-dir $project `
  --season-dir $season
```

This creates `step_00_init`, `step_01_*`, … with `start_date`, `end_date`, and `results_dir: results` aligned to the model timestep and the specified assimilation dates.

The skeleton uses the `timestep` from `project.yml` (e.g. `3H`, `6H`, `1D`) to define step boundaries. For each assimilation date `D_i`, step i runs long enough that openAMUNDSEN produces a daily grid for `D_i` in the preceding step, and step boundaries satisfy `start_{i+1} = end_i + timestep` (no duplicated timesteps). The season pipeline then assimilates SCF on the calendar date of `start_{i+1}`, which matches `D_i`.

## Troubleshooting

- Plots on Windows: use `--backend SVG`.
- HDF not recognized: ensure HDF4 support is present; check `gdalinfo --formats | findstr HDF4`.
- Windows bind mounts may drop metadata; code falls back to content-only copies.
- Package import in container: Compose sets `PYTHONPATH=/workspace`.

## Logging

- All commands accept `--log-level`.
- Internally uses loguru with the standard format in `openamundsen_da/core/constants.py`.

## Warm Start and Step Chaining

- Warm start uses the model state saved at the end of a step (when data_assimilation.restart.dump_state: true) to initialize the following step (when data_assimilation.restart.use_state: true). The runner loads the state pointed to by state_pattern or state_pointer.json under each member's results directory.
- Step boundaries must align with the model time step. If a step ends at end_date = T, the next step must start exactly one model time step later: start_date = T + one model timestep.
  - Example: With a 3-hour model time step and Step i ending at `2018-10-10 00:00:00`, Step i+1 must start at `2018-10-10 03:00:00`.
- Why: Misalignment can cause duplicated/skipped timesteps, inconsistent warm starts, or assimilation at a wrong time.
- Assimilation date: The pipeline uses the next step's start_date as the SCF assimilation date.
- Tips
  - Keep a constant model time step across steps.
  - Verify the effective time step via the merged OA config persisted next to members (e.g., `<step>/ensembles/prior/member_001/config.yml`).
  - Ensure `dump_state: true` for steps feeding warm starts, and `use_state: true` for steps starting warm.

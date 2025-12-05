# openamundsen_da - Data Assimilation for openAMUNDSEN

Lightweight tools to build and run openAMUNDSEN ensembles and assimilate satellite snow cover fraction (SCF) with a particle filter. Commands are Docker/Compose friendly and use generic variables to work across projects.

## Overview

- Seasonal snow cover prediction with an ensemble model + particle filter.
- Includes prior forcing builder, ensemble launcher, MOD10A1 preprocessing, SCF extraction, H(x) model SCF, assimilation, resampling, rejuvenation, and plotting utilities.

## Installation

- Install Docker Desktop (Windows/macOS) or Docker Engine (Linux).
- Build the image once from repo root: `docker build -t oa-da .`
- Copy `.env.example` to `.env` (local only, ignored by Git) and edit:
  - `REPO` = path to this repo on your machine
  - `PROJ` = path to your project data
  - Optional: `CPUS`, `MEMORY`, `MAX_WORKERS`
- Set Compose compatibility if needed: `setx COMPOSE_COMPATIBILITY 1` (Windows) or `export COMPOSE_COMPATIBILITY=1` (Linux/macOS).
- Volumes: `${REPO}` â†’ `/workspace`, `${PROJ}` â†’ `/data`.

### Environment notes

- GDAL/PROJ are required; prefer installing via Conda. Ensure `GDAL_DATA` and `PROJ_LIB` point to your environment (see example `project.yml`).
- Python 3.10+ is required; dependencies are declared in `pyproject.toml`.

The `.env` file is read automatically by `docker compose` from the repo root.
Keep `.env` machine-specific and untracked; `.env.example` is the template you
should commit to the repo.

## Project Variables

Define once per shell and reuse in all commands:

```powershell
$project = "/data"                                # in-container project root
$season  = "$project/propagation/season_YYYY-YYYY" # season folder
$step    = "$season/step_XX_name"                  # current step
$date    = "YYYY-MM-DD"                            # assimilation date
$dateTag = ($date -replace '-', '')
$roi     = "$project/env/roi.gpkg"                 # single-feature ROI
```

Notes

- Use forward slashes in paths (`/workspace`, `/data`).
- Optional flags are listed under each command; examples show only required flags.

## Required Project Structure

This repo expects your project to follow the fixed layout shown below. Commands derive everything from these paths, so no CLI flag is needed for files/directories that live under the structure.

```
project/
  env/
    roi.gpkg                # single ROI (preferred name)
    glaciers.gpkg           # optional glacier outlines
  meteo/
    stations.csv
    <station>.csv           # long-span forcing inputs
  propagation/
    season_YYYY-YYYY/
      season.yml            # season metadata, dates, assimilation events
      step_00_init/
        step_00.yml         # initial spin-up step
        ensembles/
          prior/            # created by season pipeline; contains member_<NNN>
          posterior/        # created by resampling (when enabled)
      step_01_YYYYMMDD-YYYYMMDD/
        step_01.yml
        ensembles/
          prior/
          posterior/
      ... additional steps ...
  obs/
    season_YYYY-YYYY/
      scf_summary.csv                       # season-wide SCF summary
      obs_scf_MOD10A1_YYYYMMDD.csv         # per-date SCF CSVs
      obs_wet_snow_S1_YYYYMMDD.csv         # optional wet-snow CSVs
  project.yml            # contains data_assimilation.h_of_x, resampling, etc.

```

- `project.yml` must define `data_assimilation.h_of_x` (used by `model_scf` + `assimilate_scf`) and the DA blocks referenced by the pipeline.
- `propagation/season_X/step_Y/ensembles/prior` is created automatically by `season.py` (using `${project}/meteo` for forcing); you only need to ensure the step YAMLs and meteorological inputs exist.
- Observations (MODIS preprocessed GeoTIFFs → CSV) live under `obs/season_X`; the pipeline assumes the CSVs follow `obs_scf_MOD10A1_YYYYMMDD.csv`.
- ROI vector: `env/roi.gpkg` (single feature) is the default for all masking; other vectors under `env/` are ignored unless you explicitly pass a different ROI.
- Glacier masking (optional but recommended for SCF/wet-snow DA): place a glacier outline at `env/glaciers.gpkg` (may contain many polygons; only those intersecting the ROI are used). When enabled, glaciers are subtracted from the ROI for model H(x), wet-snow fractions, and FSC/S1 summaries so firn/ice pixels are excluded before comparing model vs obs. Toggle via `project.yml`:

```yaml
data_assimilation:
  glacier_mask:
    enabled: true # default: auto-on when env/glaciers.gpkg exists
    path: env/glaciers.gpkg # optional override
```

Why glacier masking matters

- openAMUNDSEN (multilayer, seasonal snow) does not represent firn/ice, but SCF/FSC and wet-snow products “see” all bright or radar-wet surfaces (snow + firn + ice), especially on glaciers.
- Comparing unmasked obs to a seasonal-snow-only model biases both diagnostics and assimilation.
- Masking out glaciers keeps model vs obs consistent by removing firn/ice pixels from both sides.

## Workflow/Commands

### Prior Forcing (build ensemble)

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.core.prior_forcing `
  --input-meteo-dir $project/meteo `
  --project-dir $project `
  --step-dir $step
```

Optional flags: `--overwrite`, `--log-level <LEVEL>`

### Run Ensemble

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.core.launch `
  --project-dir $project `
  --season-dir $season `
  --step-dir $step `
  --ensemble prior
```

Optional flags: `--max-workers <N>`, `--overwrite`, `--state-pattern <glob>`, `--log-level <LEVEL>`

Parallelism and CPU limits

- The `--max-workers` value is an upper bound. The launcher clamps the actual worker count to `os.cpu_count()` inside the container and to the number of available members, so the effective workers are `min(max_workers, CPUs visible, #members)`.
- Under Docker/WSL2 the CPUs visible to the container are controlled by your WSL `.wslconfig` and the `CPUS` variable used in `compose.yml` (`deploy.resources.limits.cpus: "${CPUS:-8}"`).
- Each prior run launches `open_loop` plus `ensemble_size` members from `project.yml`. If you want to run “one process per core” in a single batch, a common pattern is: set `CPUS = N`, `ensemble_size = N-1`, and use `--max-workers N`.

### Observation Processing

- MOD10A1 preprocess (HDF -> GeoTIFF + season summary):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.mod10a1_preprocess `
  --input-dir $project/obs/MOD10A1_61_HDF `
  --season-label season_YYYY-YYYY
```

Optional flags: `--project-dir $project`, `--roi $roi`, `--roi-field <field>`, `--target-epsg <code>`, `--resolution <m>`, `--ndsi-threshold <val>`, `--no-envelope`, `--no-recursive`, `--overwrite`, `--log-level <LEVEL>`

- Single-image SCF extraction (GeoTIFF â†’ obs CSV):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.satellite_scf `
  --raster $project/obs/season_YYYY-YYYY/NDSI_Snow_Cover_YYYYMMDD.tif `
  --region $roi `
  --step-dir $step
```

Optional flags: `--output <csv>`, `--ndsi-threshold <val>`, `--log-level <LEVEL>`

- Season batch mode (turns every raster in `obs/season_YYYY-YYYY` into the per-step CSVs):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.satellite_scf `
  --season-dir $season `
  --summary-csv $project/obs/season_YYYY-YYYY/scf_summary.csv `
  --overwrite
```

Optional flags: `--log-level <LEVEL>` (the summary path defaults to `<project>/obs/<season>/scf_summary.csv`). No ROI argument is required because the CSV already stores the ROI-derived SCF stats for each date.

Batch mode walks `propagation/season_YYYY-YYYY/step_*`, matches each raster by date to its step (or the step whose `end_date` matches the raster date), and writes `obs_scf_MOD10A1_YYYYMMDD.csv` into `<step>/obs`. Per-step `scf` overrides still apply.

Alternatively, skip reprocessing entirely by driving the season mode from the `scf_summary.csv` produced by `mod10a1_preprocess`. It copies each summary row for an assimilation date into the matching `<step>/obs/` file, so you only need:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.satellite_scf `
  --season-dir $season `
  --summary-csv $project/obs/season_YYYY-YYYY/scf_summary.csv `
  --overwrite
```

Optional: `--overwrite`, `--log-level <LEVEL>` (the summary path defaults to `<project>/obs/<season>/scf_summary.csv`). No ROI argument is required because the CSV already stores the ROI-derived SCF stats for each date.

Note: the summary-based workflow is the recommended way to prepare SCF observations for assimilation; the single-image and raster batch modes are kept for backward compatibility only.

## Snowflake FSC (Sentinel-2) summarization (GeoTIFF -> season summary)

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.snowflake_fsc `
  --input-dir $project/obs/FSC_snowflake `
  --season-label season_YYYY-YYYY `
  --project-dir $project
```

Optional flags: `--roi <path>` (auto-detect single ROI under env/ if omitted), `--roi-field <field>`, `--recursive`, `--log-level`. Writes `obs/<season>/scf_summary.csv` with `date, region_id, n_valid, n_snow, scf, source` from FSC values 0..100.

## Product-aware SCF CSVs

Season mode for SCF supports a product tag so filenames match assimilation events (e.g., `--product SNOWFLAKE` -> `obs_scf_SNOWFLAKE_YYYYMMDD.csv`).

### Wet Snow Classification

Classify wet-versus-dry snow grids directly from the OA raster outputs following the volumetric liquid water content definition (Rottler et al., 2024): sum the layer-wise liquid water (kg m-2), divide by water density (1000 kg m-3) and snow depth (m), then multiply by 100 for percent. The CLI below walks every step and ensemble member (or a single step) and writes the binary mask (1 = wet, 0 = dry, 255 = nodata) plus an optional percent raster per timestamp.

```powershell
$project = "/data"
$season  = "$project/propagation/season_2019-2020"

docker compose run --rm oa `
  python -m openamundsen_da.methods.wet_snow.classify `
  --season-dir $season
```

Optional flags: `--step-dir <path>` (mutually exclusive with `--season-dir`), `--members member_001 ...`, `--threshold <percent>`, `--write-fraction`, `--min-depth-mm <mm>`. Outputs land under each member's `results/<output-subdir>` (default `wet_snow`): `wet_snow_mask_<timestamp>.tif` and `lwc_fraction_<timestamp>.tif` when `--write-fraction` is set.

Sentinel-1 wet-snow observations use pre-classified WSM rasters with four classes:

- `110` = wet snow
- `125` = dry/no snow
- `200` = radar shadow (excluded from the statistics)
- `210` = water (excluded from the statistics)

The S1 summary CLI (`oa-da-wet-snow-s1`) clips each WSM raster to the single-feature ROI, drops shadow and water pixels, and computes a two-class wet-snow fraction as:

```text
wet_snow_fraction = (# pixels == 110) / (# pixels in {110, 125})
```

This fraction is written to `wet_snow_summary.csv` along with `n_valid`, `n_wet`, and the source filename, and is later converted into per-step `obs_wet_snow_S1_YYYYMMDD.csv` files by the season helper.

## Wet-snow assimilation workflow

- Summarize observations into `wet_snow_summary.csv` (e.g., `oa-da-wet-snow-s1`), then drive the season helper to write per-step `obs_wet_snow_*.csv` aligned to assimilation dates.
- The season pipeline reads `data_assimilation.assimilation_events` from `season.yml`; it now errors if fewer events than DA steps are configured.
- Wet-snow masks/fractions are computed for all members before DA using the project wet-snow threshold; assimilation/resampling/rejuvenation then proceed like SCF.

## Per-step forcing plots

Forcing (temperature in K, cumulative precipitation) is plotted per step with all members and the open loop. The season pipeline calls this automatically for each step. Manual trigger:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_forcing_ensemble `
  --step-dir $step `
  --ensemble prior
```

## Season-level model envelopes for plotting

Season runs now aggregate member ROI series into:

- `point_scf_roi_envelope.csv`
- `point_wet_snow_roi_envelope.csv`

Each contains `date, value_mean, value_min, value_max, n` computed from all available prior member `point_*_roi.csv` files. Generate manually if needed:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.aggregate_fractions `
  --season-dir $season `
  --filename point_scf_roi.csv `
  --value-col scf `
  --output-name point_scf_roi_envelope.csv
```

## Plotting SCF + wet-snow obs/model overlay

Use the combined plot helper to overlay observations, optional single-model series, and envelopes:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.plot_fractions `
  --season-dir $season `
  --project-dir $project
```

Defaults read obs from `obs/<season>/scf_summary.csv` and `obs/<season>/wet_snow_summary.csv`, envelopes from the season root, and write `plots/results/fraction_timeseries.png`. Add `--scf-model-csv` / `--wet-model-csv` to overlay specific member series or `--scf-env-csv` / `--wet-env-csv` to use custom envelopes. Plot mode can be switched with `--mode band|members` (pipeline default: `band`).

## Season point results (SWE / snow depth, member mode)

Generate season-wide point plots (members only, legend shows just open loop + assimilation markers):

```powershell
docker compose run --rm oa python -m openamundsen_da.methods.viz.plot_season_ensemble results --season-dir $season --var-col swe --mode members --log-level INFO
docker compose run --rm oa python -m openamundsen_da.methods.viz.plot_season_ensemble results --season-dir $season --var-col snow_depth --mode members --log-level INFO
```

Outputs are written to `<season>/plots/results/season_results_point_<station>_{swe|snow_depth}_<season>.png`. The season pipeline calls the same functions with `mode=members` after each step and at the end.

### H(x) Model SCF (optional, per-member debug)

```powershell
  docker compose run --rm oa `
    python -m openamundsen_da.methods.h_of_x.model_scf `
    --project-dir $project `
    --member-results $step/ensembles/prior/member_001/results `
    --roi $roi `
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
  --roi $roi
```

Optional flags: `--obs-csv <path>`, `--output <csv>`, `--log-level <LEVEL>`

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

Optional flags: `--ess-threshold-ratio <0..1>`, `--ess-threshold <n|ratio>`, `--seed <int>`, `--overwrite`, `--log-level <LEVEL>`

Resampling configuration (season + CLI)

- The pipeline and CLI both read `data_assimilation.resampling` from `project.yml`.
- Keys: `algorithm` (systematic), `ess_threshold_ratio` (recommended 0.5–0.66), optional `ess_threshold` (absolute), and `seed`.
- Behavior: if ESS >= threshold, resampling is skipped and the prior is mirrored to the posterior; a log line like `Skipping resampling | ESS=38.2 >= thr_abs=30.0 (ensemble healthy; mirroring source->target; ess_ratio=0.637)` is emitted.
- If no threshold is set, resampling always runs.

### Rejuvenation (posterior -> prior)

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

Optional flags: `--time-col`, `--temp-col`, `--precip-col`, `--start-date`, `--end-date`, `--resample`, `--rolling`, `--hydro-month`, `--hydro-day`, `--title`, `--subtitle`, `--output-dir`, `--backend`, `--log-level`

- Results per-station:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_results_ensemble `
  --step-dir $step `
  --ensemble prior
```

Optional flags: `--time-col`, `--var-col`, `--var-label`, `--var-units`, `--start-date`, `--end-date`, `--resample`, `--resample-agg`, `--rolling`, `--band-low`, `--band-high`, `--title`, `--subtitle`, `--output-dir`, `--backend`, `--log-level`

- ESS timeline:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.plot_ess_timeline `
  --step-dir $step
```

Optional: `--normalized`, `--threshold <ratio>`, `--output <svg>`, `--backend`, `--log-level`

Outputs are written to `$season/plots/forcing` and `$season/plots/results` with the season identifier in filenames.

- Season-level assimilation plots:

  Each weights plot shows the posterior probability assigned to every member after assimilating SCF on that date. The y-axis is normalized to `[0,1]` so you can directly compare different steps, and the subtitle now records `Step <n> - <YYYY-MM-DD>` when the CSV lives under the expected `step_XX_*/assim/` layout.

  Keep these interpretation tips in mind:

  - A steep fall-off after the top members implies the observation strongly favors a few particles; this also drives the ESS timeline downward for that step.
  - A flatter trend with many weights ≈ `0.05` means the observation is not differentiating members, which can reflect broad uncertainties or overly similar ensemble members.
  - Use the residual histogram and sigma markers on the right panel: tight residuals centered near zero mean the model already matched the observation, while heavy tails or offsets may flag issue with the obs CSV or indicate the model spread is too small.

  - Per-step weights (season view):

  ```powershell
  $weights = "$season/step_01_20171122-20171224/assim/weights_scf_20171122.csv"

  docker compose run --rm oa `
    python -m openamundsen_da.methods.pf.plot_weights `
    $weights
  ```

  When the CSV lives under `$season/step_XX_*/assim/`, the plot is written to `$season/plots/assim/weights/step_XX_weights.png`.

  - Season ESS timeline (all steps):

  ```powershell
  docker compose run --rm oa `
    python -c "from pathlib import Path; from openamundsen_da.methods.pf.plot_ess_timeline import plot_season_ess_timeline; plot_season_ess_timeline(Path('$season'))"
  ```

  This scans `step_*/assim/weights_scf_*.csv` under `$season` and writes the timeline to `$season/plots/assim/ess/season_ess_timeline_<season_id>.png`.

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

- Season-wide SCF (model + obs SCF):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_season_ensemble `
  results `
  --season-dir $season `
  --var-col scf `
  --station point_scf_roi.csv
```

This uses per-member `point_scf_roi.csv` files (model SCF derived from HS/SWE grids) written under each member's `results` directory and overlays observed SCF from `obs/<season>/scf_summary.csv` when available.

Defaults: ensemble members are hidden; plots show the ensemble mean, the 90% envelope (5–95% quantiles), and the open loop. Use `--show-members` to draw all members.  
Optional: `--station`, `--max-stations`, `--start-date`, `--end-date`, `--resample`, `--rolling`, `--hydro-month`, `--hydro-day`, `--backend`, `--log-level`, `--var-label`, `--var-units`, `--band-low`, `--band-high`, `--show-members`.

Note: running the season pipeline (see below) also generates these season plots automatically under `<season_dir>/plots/{forcing,results}` and a SCF season plot when SCF data and obs summaries are present.

- Manual station result plotting (single CSV, single variable):

  For quick manual inspection of one variable from a single station results CSV (e.g., SWE, snow_depth, temperature), use the lightweight CLI `plot_station_variable`. It works on exactly one CSV and one column at a time and writes a PNG next to the CSV.

  ```powershell
  $project = "/data"
  $season  = "$project/propagation/season_2019-2020"
  $step    = "$season/step_00_init"

  docker compose run --rm oa `
    python -m openamundsen_da.methods.viz.plot_station_variable `
    "$step/ensembles/prior/member_001/results/point_latschbloder.csv" `
    --var swe
  ```

  Key options:

  - `--time-col` timestamp column in the CSV (default: `time`)
  - `--var` column to plot (e.g., `swe`, `snow_depth`, `temp`) – required
  - `--var-label` pretty y-axis/title label (defaults to column name)
  - `--var-units` units appended to the label (e.g., `mm`, `m`, `K`)
  - `--start-date`, `--end-date` optional ISO dates (`YYYY-MM-DD`) to restrict the time window
  - `--backend` Matplotlib backend (default: `Agg`, headless)

  The output file is written next to the input CSV as `<basename>.<var>.png`, e.g., `point_latschbloder.swe.png`.

## Season Pipeline

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.pipeline.season `
  --project-dir $project `
  --season-dir $season
```

The launcher automatically pulls the initial forcing from `$project/meteo` and builds the first prior ensemble (errors if the directory is missing), so you no longer need a separate `prior_forcing` run before `season.py` as long as the long-span station files live under `project/meteo`.

Optional: `--max-workers <N>`, `--overwrite`, `--no-live-plots`, `--log-level <LEVEL>` (`--no-live-plots` skips plotting during the run; all plots are created once at the end).

The pipeline drives each step in order, assimilates SCF on the _next_ step's start date, resamples the resulting weights to the posterior, and rejuvenates that posterior into the next prior before proceeding. Assimilation looks for the single-row CSV `obs_scf_MOD10A1_YYYYMMDD.csv` inside `<step>/obs/` for the date being processed; generate those files with `openamundsen_da.observer.satellite_scf` after you preprocess the MOD10A1 NDSI raster for your ROI (projection, QA/masking, and mosaicking). `season.py` never reads raw imagery, so the CSV must already reflect any filtering or thresholding you want applied.

Outputs

- Per-step runs in `<step>/ensembles/{prior,posterior}` (open_loop + members)
- Weights and indices in `<step>/assim/`
- Rejuvenated next-step prior (members + open_loop with state_pointer.json)
- Season plots under `<season_dir>/plots/{forcing,results}`
- When model SCF is enabled, daily ROI-mean SCF per member is written to `<step>/ensembles/prior/<member>/results/point_scf_roi.csv`; the combined SCF + wet-snow fraction plot (`plots/results/fraction_timeseries.png`) provides the season-level view.
  Season results plots now show the ensemble mean, the 90% envelope, and the open loop by default; individual members are hidden unless `--show-members` is passed to the plot CLI. Wet-snow season plots overlay available observations from `obs/<season>/wet_snow_summary.csv` automatically.
  At the end of the season run, per-step weights plots (`step_XX_weights.png`) and the season ESS timeline (`season_ess_timeline_<season_id>.png`) are also generated under `<season_dir>/plots/assim/{weights,ess}`.

### Backfilling model SCF for an existing season (optional)

If you have already run a season and want to compute daily ROI-mean model SCF for all members (to enable SCF season plots), you can run:

```powershell
$project = "/data"
$season  = "$project/propagation/season_YYYY-YYYY"
$roi     = "$project/env/roi.gpkg"

docker compose run --rm oa `
  python -c "from openamundsen_da.methods.h_of_x.model_scf import cli_season_daily; import sys; sys.exit(cli_season_daily(['--project-dir','$project','--season-dir','$season','--roi','$roi','--max-workers','20']))"
```

This writes per-member SCF time series to `<step>/ensembles/prior/<member>/results/point_scf_roi.csv` for all steps, so `plot_season_ensemble` with `var_col="scf"` can consume them.

### Season Skeleton (optional helper)

To create an empty season layout with `step_*` folders and minimal step YAMLs from a list of assimilation dates, you can either use the legacy flat list or the structured DA block.

Legacy (SCF-only) schema:

```yaml
start_date: 2017-10-01
end_date: 2018-09-30
assimilation_dates:
  - 2017-11-23
  - 2017-12-24
  - 2018-01-30
  # ...
```

Structured schema with per-date observable/product (preferred):

```yaml
start_date: 2017-10-01
end_date: 2018-09-30
data_assimilation:
  assimilation_events:
    - date: 2017-11-23
      variable: scf
      product: MOD10A1
    - date: 2018-03-19
      variable: wet_snow
      product: S1
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

### Performance monitoring (CPU / RAM / disk)

The framework includes a lightweight performance monitor that can be used to inspect
system bottlenecks during a season run.

What it records

- CPU and RSS memory of processes related to the season run (Python workers, etc.).
- System memory usage (used vs. total).
- Total size of the season directory on disk.
- Static metadata from project/season config:
  - ROI area (km²)
  - spatial resolution (m)
  - model timestep (e.g. `3H`)
  - number of days in the season
  - number of DA dates configured in `season.yml`
  - number of workers used (`--max-workers`)
- Wall-clock time since the season started.

Outputs

- CSV time series:
  - `<season_dir>/plots/perf/season_perf_metrics.csv`
- PNG plot (updated every few seconds):
  - `<season_dir>/plots/perf/season_perf.png`

The CSV contains, per sample:

- `timestamp` (UTC)
- `cpu_tracked_pct` – sum of CPU% over tracked processes
- `mem_tracked_mb` – sum of RSS memory over tracked processes (MB)
- `mem_used_gb`, `mem_total_gb` – system memory used/total (GB)
- `season_size_gb` – total size of `<season_dir>` on disk (GB)
- `elapsed_run_sec` – seconds since the season run started
- `roi_km2`, `resolution_m`, `timestep`, `season_days`, `num_da_dates`, `num_workers`
- `progress_steps` – fraction of completed steps (0..1)
- `done_steps` / `total_steps` – step progress counters
- `eta_utc`, `eta_local` – estimated finish time in UTC and local (if configured)

Enabling monitoring for a season run

```powershell
docker compose run --rm oa `
  oa-da-season `
  --project-dir $project `
  --season-dir $season `
  --max-workers 20 `
  --monitor-perf `
  --perf-sample-interval 5 `
  --perf-plot-interval 30
```

- `--monitor-perf` turns on the background monitor thread.
- `--perf-sample-interval` and `--perf-plot-interval` are optional; defaults are
  5 seconds and 30 seconds respectively.

Running the monitor manually

You can also attach the monitor manually to an existing season directory (for example,
while a season run is already in progress from another shell):

```powershell
$season = "$project/propagation/season_YYYY-YYYY"

docker compose run --rm oa `
  oa-da-perf-monitor `
  --season-dir $season `
  --sample-interval 5 `
  --plot-interval 30
```

This foreground command will keep updating the CSV and plot until interrupted
with `Ctrl+C`. The ETA shown in the plot is based on a simple linear model
using completed steps vs total steps; it is intended as a rough indication
only and can change as the run progresses.

## Troubleshooting

- Plots on Windows: use `--backend SVG`.
- HDF not recognized: ensure HDF4 support is present; check `gdalinfo --formats | findstr HDF4`.
- Windows bind mounts may drop metadata; code falls back to content-only copies.
- Package import in container: Compose sets `PYTHONPATH=/workspace`.

## Logging

- All commands accept `--log-level`.
- Internally uses loguru with the standard format in `openamundsen_da/core/constants.py`.

## Warm Start and Step Chaining

- Warm start uses the model state saved at the end of each step. The runner loads the state pointed to by `state_pointer.json` under each member's directory and writes a new state file in the results directory (optionally named via `--state-pattern`).
- Step boundaries must align with the model time step. If a step ends at end_date = T, the next step must start exactly one model time step later: start_date = T + one model timestep.
  - Example: With a 3-hour model time step and Step i ending at `2018-10-10 00:00:00`, Step i+1 must start at `2018-10-10 03:00:00`.
- Why: Misalignment can cause duplicated/skipped timesteps, inconsistent warm starts, or assimilation at a wrong time.
- Assimilation date: The pipeline uses the next step's start_date as the SCF assimilation date.
- Tips
  - Keep a constant model time step across steps.
  - Verify the effective time step via the merged OA config persisted next to members (e.g., `<step>/ensembles/prior/member_001/config.yml`).

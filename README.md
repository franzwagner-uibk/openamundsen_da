# openAMUNDSEN-DA - Data Assimilation for openAMUNDSEN

openAMUNDSEN-DA is an open-source data assimilation framework designed to be executable on standard workstation hardware. The framework provides documented, reproducible command-line workflows and supports parallel execution on local CPU cores; for computationally demanding applications (e.g., larger domains, finer resolutions, or larger ensembles), the same workflow can be scaled to HPC environments. openAMUNDSEN-DA is coupled to the open-source openAMUNDSEN model, and both codebases are publicly available on GitHub. For end users, openAMUNDSEN-DA is distributed as a Docker image that includes the openAMUNDSEN coupling and example data, while developers who want to contribute to this open-source project can work directly with the corresponding GitHub repositories.

## Documentation

Live docs: https://openamundsen-da.pages.dev/ (Cloudflare Pages).
This replaces the old GitHub Pages site.

## Overview

- Setup-based snow cover prediction with an ensemble model + particle filter.
- Includes prior forcing builder, ensemble launcher, generic snow-cover and wet-snow summarization, H(x) model SCF, assimilation, resampling, rejuvenation, and plotting utilities.

## Tutorial

See the docs tutorial for the full Rofental walkthrough:

`https://openamundsen-da.pages.dev/tutorial/`

Developer workflow (clone + compose) remains documented in the installation page.

## Installation (for contributors)

- Install Docker Desktop (Windows/macOS) or Docker Engine (Linux).
- Build locally if needed: `docker build -t ghcr.io/franzwagner-uibk/openamundsen_da:local .`  
  (Otherwise pull `:latest`.)
- Compose defaults now work without an `.env` file: volumes default to your current repo (`REPO=.`) and the bundled example (`PROJ=./examples/rofental`). Override per command if you need different paths, e.g.  
  `REPO=/my/repo PROJ=/my/project docker compose run --rm oa ...`
- Docker permissions: ensure your user can access the Docker daemon (Linux: docker group or sudo).

### Container image (GHCR) and CI

- Images are built/published to GHCR at `ghcr.io/franzwagner-uibk/openamundsen_da` (tags: `main-YYYYMMDD`, short SHA, `latest`).
- GitHub Actions workflow `.github/workflows/ci.yml` runs unit + integration tests and publishes to GHCR on pushes to `main` only after tests pass; requires repo secret `GHCR_PAT` with `write:packages`.
- To pull from GHCR locally/servers: `echo "$GHCR_PAT_RO" | docker login ghcr.io -u <github-user> --password-stdin` then `docker pull ghcr.io/franzwagner-uibk/openamundsen_da:<tag>`.

### Environment notes

- GDAL/PROJ are bundled in the image; if running natively, install via Conda and ensure `GDAL_DATA` / `PROJ_LIB` are set.
- Python 3.10+ is required; dependencies are declared in `pyproject.toml`.

## Project Variables

Define once per shell and reuse in all commands:

```powershell
$setup   = "/data"                                        # setup root (top level)
$project = "$setup/projects/project_YYYY-YYYY"            # one DA project
$step    = "$project/steps/step_XX_name"                  # current step
$date    = "YYYY-MM-DD"                            # assimilation date
$dateTag = ($date -replace '-', '')
$roi     = "$setup/env/roi.gpkg"                   # optional ROI vector; DA always uses grids/roi_<domain>_<resolution>.asc
```

Notes

- Use forward slashes in paths (`/workspace`, `/data`).
- Optional flags are listed under each command; examples show only required flags.

## Required Project Structure

This repo expects the following setup/project hierarchy:

```
setup/
  <setup-name>.yml         # setup-level openAMUNDSEN config (template fallback: setup.yml)
  env/
    roi.gpkg                # optional ROI vector (preferred name)
    subdomains.gpkg         # optional multi-feature regions file for sub-domain mode
  grids/
    roi_<domain>_<resolution>.asc  # canonical ROI mask used by DA runs
    lc_<domain>_<resolution>.asc  # land-cover classes used for masking
  meteo/
    stations.csv
    <station>.csv           # long-span forcing inputs
  projects/
    project_YYYY-YYYY/
      project_YYYY-YYYY.yml # project-level DA config + start/end + assimilation_events
      steps/
        step_00_init/
          step_00.yml         # initial spin-up step
          ensembles/
            prior/            # created by project pipeline; contains member_<NNN>
            posterior/        # created by resampling (when enabled)
        step_01_YYYYMMDD-YYYYMMDD/
          step_01.yml
          ensembles/
            prior/
            posterior/
        ... additional steps ...
  obs/
    project_YYYY-YYYY/
      scf_summary.csv                       # project-wide SCF summary
      obs_scf_SNOWCOVER_YYYYMMDD.csv       # per-date SCF CSVs
      obs_wet_snow_WETSNOW_YYYYMMDD.csv    # optional wet-snow CSVs
    summaries/
      project_YYYY-YYYY/
        scf_summary.csv                     # default location for snow-cover summaries
        wet_snow_summary.csv                # default location for wet-snow summaries
```

You can use the scaffold under `templates/project` as a starting point.
Each directory in that template contains a small `readme.txt` describing the expected files
and naming conventions.

- Setup YAML (`<setup-name>.yml`/`setup.yml`) must stay pure openAMUNDSEN config (no DA block).
- Project YAML (`<project-name>.yml`/`project.yml`) must define `data_assimilation` (`h_of_x`, `likelihood`, `resampling`, `rejuvenation`, `restart`, `landcover_mask`, `assimilation_events`) plus `start_date` and `end_date`.
- `projects/project_X/steps/step_Y/ensembles/prior` is created automatically by the project pipeline (using `${setup}/meteo` forcing).
- Observations live under `obs/project_X`; the pipeline assumes the CSVs follow `obs_scf_<PRODUCT>_YYYYMMDD.csv` and `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv`. Configure product tags explicitly in project YAML under `obs.*`.
- DA uses `grids/roi_<domain>_<resolution>.asc` as canonical ROI mask; if missing, it is generated silently from ROI vectors under `env/` (`roi.gpkg` preferred, `subdomains.gpkg` supported).
- Land-cover masking (applied to obs + model SCF/wet-snow): land-cover ASCII is resolved as `grids/lc_<domain>_<resolution>.asc` from setup config; excluded classes come from project YAML `data_assimilation.landcover_mask.classes_to_exclude`.

```yaml
data_assimilation:
  landcover_mask:
    enabled: true
    classes_to_exclude: [2, 8, 9, 10, 11, 12, 13]
```

Why land-cover masking matters

- Dense forest and built-up surfaces often hide snow in satellite products while the model may still simulate snow below canopy or within cities. Ice/glacier classes are also excluded by default via the land-cover grid.
- Masking keeps model vs obs consistent by removing pixels where observation/model support diverges.
- Best-practice split: use `landcover_mask.classes_to_exclude` for truly unusable classes (for example ice/water/urban), and handle usable-but-uncertain classes (for example forest/shadow) with uncertainty penalties rather than hard exclusion.

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
  --setup-dir $setup `
  --step-dir $step `
  --ensemble prior
```

Optional flags: `--max-workers <N>`, `--overwrite`, `--state-pattern <glob>`, `--log-level <LEVEL>`

Parallelism and CPU limits

- The `--max-workers` value is an upper bound. The launcher clamps the actual worker count to `os.cpu_count()` inside the container and to the number of available members, so the effective workers are `min(max_workers, CPUs visible, #members)`.
- Under Docker/WSL2 the CPUs visible to the container are controlled by your WSL `.wslconfig` and the `CPUS` variable used in `compose.yml` (`deploy.resources.limits.cpus: "${CPUS:-8}"`).
- Each prior run launches `open_loop` plus `ensemble_size` members from `setup.yml` (`data_assimilation.prior_forcing`). `open_loop` is the unperturbed, unassimilated baseline and is carried through all steps for reference/plots. If you want to run "one process per core" in a single batch, a common pattern is: set `CPUS = N`, `ensemble_size = N-1`, and use `--max-workers N`.

### Observation Processing

- Snow-cover summary (GeoTIFF/NetCDF -> `scf_summary.csv`):

```powershell
docker compose run --rm oa `
  oa-da-snowcover `
  --input-dir $setup/obs/snowcover `
  --project-label project_YYYY-YYYY `
  --setup-dir $setup `
  --overwrite
```

Classes and product tags are read from project YAML `obs.snowcover`.

- Wet-snow summary (categorical rasters -> `wet_snow_summary.csv`):

```powershell
docker compose run --rm oa `
  oa-da-wetsnow `
  --input-dir $setup/obs/wetsnow `
  --project-label project_YYYY-YYYY `
  --setup-dir $setup `
  --overwrite
```

Classes come from project YAML `obs.wetsnow.classes`; the project-level DA land-cover exclusions are applied automatically.

- Per-step obs CSVs (align summaries to assimilation events):

```powershell
docker compose run --rm oa oa-da-scf --project-dir $project --overwrite
docker compose run --rm oa oa-da-wetsnow-project --project-dir $project --overwrite
```

Both commands default to `<setup>/obs/<project-name>/scf_summary.csv` and `wet_snow_summary.csv`, and write per-step obs CSVs under `<project>/steps/*/obs/`.

### Wet Snow Classification

Classify wet-versus-dry snow grids directly from the OA raster outputs following the volumetric liquid water content definition (Rottler et al., 2024): sum the layer-wise liquid water (kg m-2), divide by water density (1000 kg m-3) and snow depth (m), then multiply by 100 for percent. The CLI below walks every step and ensemble member (or a single step) and writes the binary mask (1 = wet, 0 = dry, 255 = nodata) plus an optional percent raster per timestamp.

```powershell
$setup = "/data"
$project = "$setup/projects/project_2019-2020"

docker compose run --rm oa `
  python -m openamundsen_da.methods.wet_snow.classify `
  --setup-dir $project
```

Optional flags: `--step-dir <path>` (mutually exclusive with `--setup-dir`), `--members member_001 ...`, `--threshold <percent>`, `--write-fraction`, `--min-depth-mm <mm>`. Outputs land under each member's `results/<output-subdir>` (default `wet_snow`): `wet_snow_mask_<timestamp>.tif` and `lwc_fraction_<timestamp>.tif` when `--write-fraction` is set.

Wet-snow observations use categorical rasters (e.g., Sentinel-1 WSM). `oa-da-wetsnow` clips rasters to the ROI, applies land-cover exclusions, and writes `wet_snow_summary.csv`; the project helper converts summary rows into per-step `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv`.

## Wet-snow assimilation workflow

- Summarize observations into `wet_snow_summary.csv` (e.g., `oa-da-wetsnow`), then drive the setup helper to write per-step `obs_wet_snow_*.csv` aligned to assimilation dates.
- The project pipeline reads `data_assimilation.assimilation_events` from project YAML; it requires exactly one event per non-final step.
- Per-step observation preparation (`oa-da-scf`, `oa-da-wetsnow-project`) is fail-fast:
  - event date must lie inside the associated step window,
  - the summary CSV must contain a row for each configured event date of that variable.
- Wet-snow masks/fractions are computed for all members before DA using the project wet-snow threshold; assimilation/resampling/rejuvenation then proceed like SCF.

## Per-step forcing plots

Forcing (temperature in K, cumulative precipitation) is plotted per step with all members and the open loop. The setup pipeline calls this automatically for each step. Manual trigger:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_forcing_ensemble `
  --step-dir $step `
  --ensemble prior
```

## Setup-level model envelopes for plotting

Setup runs now aggregate member ROI series into:

- `point_scf_roi_envelope.csv`
- `point_wet_snow_roi_envelope.csv`

Each contains `date, value_mean, value_min, value_max, n` computed from all available prior member `point_*_roi.csv` files. Generate manually if needed:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.aggregate_fractions `
  --setup-dir $setup `
  --filename point_scf_roi.csv `
  --value-col scf `
  --output-name point_scf_roi_envelope.csv
```

## Plotting SCF + wet-snow obs/model overlay

Use the combined plot helper to overlay observations, optional single-model series, and envelopes:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.observer.plot_fractions `
  --setup-dir $setup `
  --project-dir $project
```

Defaults read obs from `obs/summaries/<setup>/scf_summary.csv` and `obs/summaries/<setup>/wet_snow_summary.csv`, envelopes from the setup root, and write `plots/results/fraction_timeseries.png`. Add `--scf-model-csv` / `--wet-model-csv` to overlay specific member series or `--scf-env-csv` / `--wet-env-csv` to use custom envelopes. Plot mode can be switched with `--mode band|members` (pipeline default: `band`).

## Setup point results (SWE / snow depth, member mode)

Generate setup-wide point plots (members only, legend shows just open loop + assimilation markers):

```powershell
docker compose run --rm oa python -m openamundsen_da.methods.viz.plot_project_ensemble results --setup-dir $setup --var-col swe --mode members --log-level INFO
docker compose run --rm oa python -m openamundsen_da.methods.viz.plot_project_ensemble results --setup-dir $setup --var-col snow_depth --mode members --log-level INFO
```

Outputs are written to `<setup>/plots/results/setup_results_point_<station>_{swe|snow_depth}_<setup>.png`. The setup pipeline calls the same functions with `mode=members` after each step and at the end.

### H(x) Model SCF (optional, per-member debug)

```powershell
  docker compose run --rm oa `
    python -m openamundsen_da.methods.h_of_x.model_scf `
    --project-dir $project `
    --member-results $step/ensembles/prior/member_001/results `
    --roi $roi `
    --date $date
```

Model parameters (`variable`, `method`, `h0`, `k`) are read from `setup.yml` under `data_assimilation.h_of_x`; the CLI no longer accepts overrides.

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

Resampling configuration (setup + CLI)

- The pipeline and CLI both read `data_assimilation.resampling` from `setup.yml`.
- Keys: `algorithm` (systematic), `ess_threshold_ratio` (recommended `0.50-0.66`), optional `ess_threshold` (absolute), and `seed`.
- Behavior: if ESS >= threshold, resampling is skipped and the prior is mirrored to the posterior; a log line like `Skipping resampling | ESS=38.2 >= thr_abs=30.0 (ensemble healthy; mirroring source->target; ess_ratio=0.637)` is emitted.
- If no threshold is set, resampling always runs.

### Rejuvenation (posterior -> prior)

Rebase is default (perturbations are applied relative to open_loop). If rejuvenation sigmas are not set, they fall back to prior_forcing sigmas.

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.rejuvenate `
  --project-dir $project `
  --prev-step-dir $setup/steps/step_XX_prev `
  --next-step-dir $setup/steps/step_YY_next
```

Optional: `--source-meteo-dir <path>`, `--log-level <LEVEL>`

Setup config (example):

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

Outputs are written to `$setup/plots/forcing` and `$setup/plots/results` with the setup identifier in filenames.

- Setup-level assimilation plots:

  Each weights plot shows the posterior probability assigned to every member after assimilating SCF on that date. The y-axis is normalized to `[0,1]` so you can directly compare different steps, and the subtitle now records `Step <n> - <YYYY-MM-DD>` when the CSV lives under the expected `step_XX_*/assim/` layout.

  Keep these interpretation tips in mind:

  - A steep fall-off after the top members implies the observation strongly favors a few particles; this also drives the ESS timeline downward for that step.
  - A flatter trend with many weights  `0.05` means the observation is not differentiating members, which can reflect broad uncertainties or overly similar ensemble members.
  - Use the residual histogram and sigma markers on the right panel: tight residuals centered near zero mean the model already matched the observation, while heavy tails or offsets may flag issue with the obs CSV or indicate the model spread is too small.

  - Per-step weights (setup view):

  ```powershell
  $weights = "$setup/steps/step_01_20171122-20171224/assim/weights_scf_20171122.csv"

  docker compose run --rm oa `
    python -m openamundsen_da.methods.pf.plot_weights `
    $weights
  ```

  When the CSV lives under `$setup/steps/step_XX_*/assim/`, the plot is written to `$setup/plots/assim/weights/step_XX_weights.png`.

  - Setup ESS timeline (all steps):

  ```powershell
  docker compose run --rm oa `
    python -c "from pathlib import Path; from openamundsen_da.methods.pf.plot_ess_timeline import plot_setup_ess_timeline; plot_setup_ess_timeline(Path('$setup'))"
  ```

  This scans `steps/step_*/assim/weights_scf_*.csv` under `$setup` and writes the timeline to `$setup/plots/assim/ess/setup_ess_timeline_<setup_id>.png`.

- Setup-wide forcing/results (stitch all steps together):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_project_ensemble `
  forcing `
  --setup-dir $setup
```

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_project_ensemble `
  results `
  --setup-dir $setup `
  --var-col swe
```

- Setup-wide SCF (model + obs SCF):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_project_ensemble `
  results `
  --setup-dir $setup `
  --var-col scf `
  --station point_scf_roi.csv
```

This uses per-member `point_scf_roi.csv` files (model SCF derived from HS/SWE grids) written under each member's `results` directory and overlays observed SCF from `obs/summaries/<setup>/scf_summary.csv` when available.

Defaults: ensemble members are hidden; plots show the ensemble mean, the 90% envelope (595% quantiles), and the open loop. Use `--show-members` to draw all members.  
Optional: `--station`, `--max-stations`, `--start-date`, `--end-date`, `--resample`, `--rolling`, `--hydro-month`, `--hydro-day`, `--backend`, `--log-level`, `--var-label`, `--var-units`, `--band-low`, `--band-high`, `--show-members`.

Note: running the setup pipeline (see below) also generates these setup plots automatically under `<setup_dir>/plots/{forcing,results}` and a SCF setup plot when SCF data and obs summaries are present.

- Manual station result plotting (single CSV, single variable):

  For quick manual inspection of one variable from a single station results CSV (e.g., SWE, snow_depth, temperature), use the lightweight CLI `plot_station_variable`. It works on exactly one CSV and one column at a time and writes a PNG next to the CSV.

  ```powershell
  $project = "/data"
  $setup  = "$project/projects/project_2019-2020"
  $step    = "$setup/steps/step_00_init"

  docker compose run --rm oa `
    python -m openamundsen_da.methods.viz.plot_station_variable `
    "$step/ensembles/prior/member_001/results/point_latschbloder.csv" `
    --var swe
  ```

  Key options:

  - `--time-col` timestamp column in the CSV (default: `time`)
  - `--var` column to plot (e.g., `swe`, `snow_depth`, `temp`)  required
  - `--var-label` pretty y-axis/title label (defaults to column name)
  - `--var-units` units appended to the label (e.g., `mm`, `m`, `K`)
  - `--start-date`, `--end-date` optional ISO dates (`YYYY-MM-DD`) to restrict the time window
  - `--backend` Matplotlib backend (default: `Agg`, headless)

  The output file is written next to the input CSV as `<basename>.<var>.png`, e.g., `point_latschbloder.swe.png`.

## Setup Pipeline

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.pipeline.project `
  --project-dir $project `
  --setup-dir $setup
```

The launcher automatically pulls the initial forcing from `$project/meteo` and builds the first prior ensemble (errors if the directory is missing), so you no longer need a separate `prior_forcing` run before `setup.py` as long as the long-span station files live under `project/meteo`.

Optional: `--max-workers <N>`, `--overwrite`, `--live-plots`, `--log-level <LEVEL>` (`--live-plots` enables in-run plotting; default is off and plots are created once at the end).

At startup the launcher validates assimilation prerequisites: required grid outputs configured in `project.yml` (snow depth for SCF, liquid water content for wet-snow), matching model outputs in prior/open_loop results, and the expected obs CSV in each step directory. Missing items are listed and the run aborts early.

The pipeline drives each step in order, assimilates SCF on the _next_ step's start date, resamples the resulting weights to the posterior, and rejuvenates that posterior into the next prior before proceeding. Assimilation looks for the single-row CSV `obs_scf_SNOWCOVER_YYYYMMDD.csv` inside `<step>/obs/` for the date being processed; generate those files with `openamundsen_da.observer.satellite_scf` after you summarize your snow-cover rasters into `scf_summary.csv`. `setup.py` never reads raw imagery, so the CSV must already reflect any filtering or thresholding you want applied.

Outputs

- Per-step runs in `<step>/ensembles/{prior,posterior}` (open_loop + members)
- Weights and indices in `<step>/assim/`
- Rejuvenated next-step prior (members + open_loop with state_pointer.json)
- Compact DA summary grids in `<project>/results/grids/da_output_grids.nc`
  - Per variable `<var>`: `open_loop_<var>`, `ens_mean_<var>`, `ens_std_<var>`, `ens_min_<var>`, `ens_max_<var>`, `increment_<var>`
  - `increment_<var>` is defined as `ens_mean_<var> - open_loop_<var>`
  - Time axis spans the full project timeline across all steps (not only the last step)
- Setup plots under `<setup_dir>/plots/{forcing,results}`
- When model SCF is enabled, daily ROI-mean SCF per member is written to `<step>/ensembles/prior/<member>/results/point_scf_roi.csv`; the combined SCF + wet-snow fraction plot (`plots/results/fraction_timeseries.png`) provides the setup-level view.
  Setup results plots now show the ensemble mean, the 90% envelope, and the open loop by default; individual members are hidden unless `--show-members` is passed to the plot CLI. Wet-snow setup plots overlay available observations from `obs/<setup>/wet_snow_summary.csv` automatically.
  At the end of the setup run, per-step weights plots (`step_XX_weights.png`) and the setup ESS timeline (`setup_ess_timeline_<setup_id>.png`) are also generated under `<setup_dir>/plots/assim/{weights,ess}`.
  Default retention is compact (`data_assimilation.output.retention: compact`), which prunes heavy member grid artifacts after writing `da_output_grids.nc`. Set `retention: full` to keep all member grid files.

### Backfilling model SCF for an existing setup (optional)

If you have already run a setup and want to compute daily ROI-mean model SCF for all members (to enable SCF setup plots), you can run:

```powershell
$project = "/data"
$setup  = "$project/projects/project_YYYY-YYYY"
$roi     = "$project/env/roi.gpkg"

docker compose run --rm oa `
  python -c "from openamundsen_da.methods.h_of_x.model_scf import cli_setup_daily; import sys; sys.exit(cli_setup_daily(['--project-dir','$project','--setup-dir','$setup','--roi','$roi','--max-workers','20']))"
```

This writes per-member SCF time series to `<step>/ensembles/prior/<member>/results/point_scf_roi.csv` for all steps, so `plot_setup_ensemble` with `var_col="scf"` can consume them.

### Setup Skeleton (optional helper)

To create an empty setup layout with `steps/step_*` folders and minimal step YAMLs, use the structured DA block:

```yaml
start_date: 2017-10-01
end_date: 2018-09-30
data_assimilation:
  assimilation_events:
    - date: 2017-11-23
      variable: scf
      product: SNOWCOVER
    - date: 2018-03-19
      variable: wet_snow
      product: S1
    # ...
```

Then run:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.pipeline.project_skeleton `
  --project-dir $project `
  --setup-dir $setup
```

This creates `step_00_init`, `step_01_*`,  with `start_date`, `end_date`, and `results_dir: results` aligned to the model timestep and the specified assimilation dates.

The skeleton uses the `timestep` from `project.yml` (e.g. `3H`, `6H`, `1D`) to define step boundaries. For each assimilation date `D_i`, step i runs long enough that openAMUNDSEN produces a daily grid for `D_i` in the preceding step, and step boundaries satisfy `start_{i+1} = end_i + timestep` (no duplicated timesteps). The setup pipeline then assimilates SCF on the calendar date of `start_{i+1}`, which matches `D_i`.

### Performance monitoring (CPU / RAM)

A minimal monitor samples system CPU% and RAM% (enabled by default for `oa-da-project`).
Outputs under `<setup_dir>/plots/perf/`:
- `setup_perf_metrics.csv` (timestamp, cpu_total_pct, mem_used_pct, mem_used_gb, mem_total_gb)
- `setup_perf.png` (CPU/RAM%)

Suggested intervals: sample every 5-10 seconds; refresh the plot every 30-60 seconds.

Project run with default monitoring

```powershell
docker compose run --rm oa `
  oa-da-project `
  --setup-dir $setup
```

- `--monitor-perf` explicitly enables monitoring (default behavior).
- `--no-monitor-perf` disables monitoring for the project run.
- `--project-dir` is optional; it is auto-detected by walking up from `--setup-dir` to the nearest `project.yml`.
- `--perf-sample-interval` and `--perf-plot-interval` default to 5 seconds and 30 seconds respectively.

Running the monitor manually

You can also attach the monitor manually to an existing setup directory (for example,
while a setup run is already in progress from another shell):

```powershell
$setup = "$project/projects/project_YYYY-YYYY"

docker compose run --rm oa `
  oa-da-perf-monitor `
  --setup-dir $setup `
  --sample-interval 5 `
  --plot-interval 30
```

This foreground command will keep updating the CSV and plot until interrupted
with `Ctrl+C`.


## State cleanup (free disk space)

- Automatic: set `data_assimilation.restart.cleanup_after_setup: true` (default) in `setup.yml` to delete member state pickle files after a successful setup run.
- Manual (ignores the toggle): clean one or all setups via Docker Compose.

All setups under a project:

```powershell
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.cleanup \
  --project-dir /data/your_project \
  --all-setups \
  --log-level INFO
```

Single setup:

```powershell
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.cleanup \
  --project-dir /data/your_project \
  --setup-dir /data/your_project/projects/project_YYYY-YYYY \
  --log-level INFO
```

If you rebuilt the image with the latest code, you can replace the `python -m ...cleanup` line with the shorter `oa-da-clean-project`.


## Sub-domain Mode

Use `oa-da-subdomain` to split a large setup into non-overlapping sub-domains, run one independent DA project per sub-domain, write project-level reports, and merge compact DA grids.

Minimal flow:
- Prepare sub-domain setups from ROI polygons:
  `oa-da-subdomain prepare --setup-dir <setup> --project-dir <setup>/projects/<project> --roi <setup>/env/subdomains.gpkg --id-field id`
- Run all sub-domains in parallel:
  `oa-da-subdomain run --project-dir <setup>/projects/<project>`
- Write project-level reports and merge DA grids (hard mosaic, no interpolation/blending):
  `oa-da-subdomain merge --project-dir <setup>/projects/<project>`

Defaults:
- Sub-domain root is `<project>/subdomains` (override with `--subdomain-root`).
- Manifest path is `<subdomain_root>/subdomain_manifest.json` (or pass `--manifest` explicitly).
- Each sub-domain run lives under `<subdomain_root>/<subdomain_id>/`.
- Project-level outputs are written under `<project>/results/`.
- Compact DA grid output is `<project>/results/grids/da_output_grids.nc`.
  - Variables in `da_output_grids.nc`: `open_loop_<var>`, `ens_mean_<var>`, `ens_std_<var>`, `ens_min_<var>`, `ens_max_<var>`, `increment_<var>`.
- Sub-domain reports are written under `<project>/results/subdomain_*.csv`.
- Point outputs and plots remain inside each sub-domain project.
- Station selection uses a 50 km default buffer (`--station-buffer-km`).
- Tiny polygon overlaps are tolerated up to 100 m^2 (`--overlap-area-tol-m2`), with optional sliver correction (`--sliver-fix-m`).
- Sub-domain mode requires at least two polygons in the ROI file.
- `--id-field` must exist in the regions file; there is no automatic fallback to another field name.
- If `--roi` is omitted in sub-domain prepare/pipeline, `<setup>/env/subdomains.gpkg` is preferred and `<setup>/env/roi.gpkg` is the fallback.
- Sub-domain runs fail fast if configured assimilation events are not available in the local sub-domain observation summaries.
- Default retention is compact (`data_assimilation.output.retention: compact`) and removes heavy member grid artifacts after merge. Set `retention: full` to keep them.

Ready-made example:
- setup: `examples/subdomains`
- regions file: `examples/subdomains/env/subdomains.gpkg` (pass with `--roi`)
- note: this lightweight example reuses raw grids/meteo/obs from `examples/rofental` via relative paths.

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





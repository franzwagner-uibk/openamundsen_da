# openAMUNDSEN-DA - Data Assimilation for openAMUNDSEN

openAMUNDSEN-DA is an open-source data assimilation framework designed to be executable on standard workstation hardware. The framework provides documented, reproducible command-line workflows and supports parallel execution on local CPU cores; for computationally demanding applications (e.g., larger domains, finer resolutions, or larger ensembles), the same workflow can be scaled to HPC environments. openAMUNDSEN-DA is coupled to the open-source openAMUNDSEN model, and both codebases are publicly available on GitHub. For end users, openAMUNDSEN-DA is distributed as a Docker image that includes the openAMUNDSEN coupling and example data, while developers who want to contribute to this open-source project can work directly with the corresponding GitHub repositories.

## Documentation

Live docs: https://openamundsen-da.pages.dev/ (Cloudflare Pages).
This replaces the old GitHub Pages site.

Useful entry points:

- Configuration: `https://openamundsen-da.pages.dev/guides/configuration/`
- Station assimilation: `https://openamundsen-da.pages.dev/guides/station-assimilation/`
- Tutorial: `https://openamundsen-da.pages.dev/tutorial/`

## Overview

- Setup-based snow cover prediction with an ensemble model + particle filter.
- Includes prior forcing builder, ensemble launcher, generic snow-cover and wet-snow summarization, H(x) model SCF, assimilation, resampling, rejuvenation, and plotting utilities.

## How to Use

See the docs How to Use section for the full Rofental walkthrough:

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
$project = "$setup/projects/project_YYYY-YYYY"            # one data assimilation project
$step    = "$project/steps/step_XX_name"                  # current step
$date    = "YYYY-MM-DD"                            # assimilation date
$dateTag = ($date -replace '-', '')
$roi     = "$setup/env/roi.gpkg"                   # optional ROI vector; data assimilation always uses grids/roi_<domain>_<resolution>.asc
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
    roi_<domain>_<resolution>.asc  # canonical ROI mask used by data assimilation runs
    lc_<domain>_<resolution>.asc  # land-cover classes used for masking
  meteo/
    stations.csv
    <station>.csv           # long-span forcing inputs
  projects/
      project_YYYY-YYYY/
        project_YYYY-YYYY.yml # project-level data assimilation config + start/end + assimilation_events
        results/
          benchmark/          # scientific benchmark tables + manifest + summary
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
    stations/
      <station>.csv                      # setup-level station observations (time,snow_depth,swe)
      stations_da_metadata.csv          # optional station DA metadata (station_id,station_uncertainty_pct)
    project_YYYY-YYYY/
      scf_summary.csv                       # project-wide SCF summary
      obs_scf_SNOWCOVER_YYYYMMDD.csv       # per-date SCF CSVs
      obs_wet_snow_WETSNOW_YYYYMMDD.csv    # optional WSF CSVs
      obs_wet_snow_line_WETSNOW_YYYYMMDD.csv # optional WSLA CSVs
    summaries/
      project_YYYY-YYYY/
        scf_summary.csv                     # default location for snow-cover summaries
        wet_snow_summary.csv                # default location for WSF summaries
        wet_snow_line_diagnostics.csv       # optional seasonal WSLA diagnostics
```

You can use the scaffold under `templates/project` as a starting point.
Each directory in that template contains a small `readme.txt` describing the expected files
and naming conventions.

- Setup YAML (`<setup-name>.yml`/`setup.yml`) must stay pure openAMUNDSEN config (no data assimilation block).
- Project YAML (`<project-name>.yml`/`project.yml`) must define `data_assimilation` (`h_of_x`, `likelihood`, `resampling`, `rejuvenation`, `restart`, `landcover_mask`, `assimilation_events`; add `station` when using station HS/SWE assimilation) plus `start_date` and `end_date`.
- `projects/project_X/steps/step_Y/ensembles/prior` is created automatically by the project pipeline (using `${setup}/meteo` forcing).
- Observations live under `obs/project_X`; the pipeline assumes the per-step CSVs follow `obs_scf_<PRODUCT>_YYYYMMDD.csv`, `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv`, and `obs_wet_snow_line_<PRODUCT>_YYYYMMDD.csv` when those observables are active. Configure product tags and summary sources explicitly in project YAML under `obs.*` (`summary_csv` for SCF/wet-snow summaries).
- Station observations live under `obs/stations`; ROI-based station assimilation uses `assimilation_events` variables `station_hs` and `station_swe` and reads optional per-station uncertainty metadata from `obs/stations/stations_da_metadata.csv`. Station metadata may also include `use_for_da` and `use_for_benchmark` flags to keep stations out of assimilation or benchmark scoring without deleting their observation files.
- The station assimilation method itself is documented in the docs guide: `guides/station-assimilation`.
- Scientific benchmarking always runs at the end of `oa-da-project` and writes observation-based score tables under `results/benchmark/` plus the headline DA-skill plot `results/plots/assim/scores/performance_scores.png`. Station benchmark rows now also carry sigma-aware `zSkill` based on the configured station uncertainty metadata.
- data assimilation uses `grids/roi_<domain>_<resolution>.asc` as canonical ROI mask; if missing, it is generated silently from ROI vectors under `env/` (`roi.gpkg` preferred, `subdomains.gpkg` supported).
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

Current forcing perturbations are:

- additive `temp` offset (`sigma_t`)
- multiplicative precipitation factor (`mu_p`, `sigma_p`)
- additive relative humidity offset (`sigma_rh`)
- multiplicative shortwave factor (`sigma_sw`)

`sigma_rh` and `sigma_sw` default to `0.0` when omitted, which preserves the previous two-variable behavior.

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

Classes come from project YAML `obs.wetsnow.classes`; the project-level data assimilation land-cover exclusions are applied automatically.

- Per-step obs CSVs (align summaries to assimilation events):

```powershell
docker compose run --rm oa oa-da-scf --project-dir $project --overwrite
docker compose run --rm oa oa-da-wetsnow-project --project-dir $project --overwrite
```

Both commands resolve summaries from `obs.snowcover.summary_csv` / `obs.wetsnow.summary_csv` when configured, otherwise from the legacy defaults under `<setup>/obs/<project-name>/` or `<setup>/obs/summaries/<project-name>/`. When run with `--summary-csv`, they record that path in project YAML so maps and benchmarking use the same source, then write per-step obs CSVs under `<project>/steps/*/obs/`.

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

Wet-snow observations use categorical rasters (e.g., Sentinel-1 WSM). `oa-da-wetsnow` clips rasters to the ROI, applies land-cover exclusions, and writes `wet_snow_summary.csv` for wet snow fraction (WSF). That summary now also carries basin-wide wet snow line altitude (WSLA) diagnostics based on the **50% wet-fraction crossing** plus companion `wet_snow_line_diagnostics.csv` / `wet_snow_line_profile_YYYYMMDD.csv` outputs when the project defines `data_assimilation.wet_snow_line`. The same diagnostics surface also stores separate aspect-sector relative-threshold WSLA diagnostics for analysis only. The project helper converts summary rows into per-step `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv` and `obs_wet_snow_line_<PRODUCT>_YYYYMMDD.csv`.

## Wet-snow assimilation workflow

- Summarize observations into `wet_snow_summary.csv` (e.g., `oa-da-wetsnow`), then drive the setup helper to write per-step `obs_wet_snow_*.csv` aligned to assimilation dates.
- When `wet_snow_line` is used, the same summary pass also derives the WSLA diagnostics and the project helper writes `obs_wet_snow_line_*.csv` for the configured WSLA dates.
- The project pipeline reads `data_assimilation.assimilation_events` from project YAML; it requires exactly one event per non-final step.
- Per-step observation preparation (`oa-da-scf`, `oa-da-wetsnow-project`) is fail-fast:
  - event date must lie inside the associated step window,
  - the summary CSV must contain a row for each configured event date of that variable.
- Wet-snow masks/fractions are computed for all members before data assimilation using the project wet-snow classification method. The default `liquid_water_fraction` method keeps the existing ratio threshold, while `liquid_water_amount` classifies cells by summed snowpack liquid water in mm. `wet_snow` keeps the scalar WSF update, while `wet_snow_line` derives a scalar WSLA from the **50% wet-fraction crossing elevation** and assimilates it with a Gaussian likelihood in meters.

## Per-step forcing plots

Forcing (temperature in K, cumulative precipitation) is plotted per step with all members and the open loop. The setup pipeline calls this automatically for each step. Manual trigger:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plots.forcing_ensemble `
  --step-dir $step `
  --ensemble prior
```

## Setup-level model envelopes for plotting

Project runs now aggregate member ROI series into:

- `results/misc/point_scf_roi_envelope.csv`
- `results/misc/point_wet_snow_roi_envelope.csv`
- `results/misc/point_wet_snow_line_roi_envelope.csv`

Each contains `date, value_mean, value_min, value_max, n` computed from all available prior member `point_*_roi.csv` files. Generate manually if needed:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.aggregate_fractions `
  --setup-dir $setup `
  --filename point_scf_roi.csv `
  --value-col scf `
  --output-name point_scf_roi_envelope.csv
```

## Plotting the result overview

Use the combined plot helper to overlay observations, optional single-model series, and envelopes:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plots.result_overview `
  --setup-dir $setup `
  --project-dir $project
```

Defaults read obs from `obs/summaries/<setup>/scf_summary.csv`, `wet_snow_summary.csv`, and `wet_snow_line_diagnostics.csv`, use the canonical project envelopes under `results/misc/` for SCF / WSF / WSLA, and write `results/plots/results/result_overview.png`. The default setup overview expands to a 5-panel layout when WSLA data are available: SCF, wet snow fraction (WSF), wet snow line altitude (WSLA), ROI mean SWE, and ROI mean snow depth. The WSLA panel uses the daily `open_loop` scalar from `point_wet_snow_line_roi.csv`, while the ensemble WSLA line / band is built from stitched prior-member `point_wet_snow_line_roi.csv` series: the center line is the daily prior **median** and the band is the daily prior **min-max** span. On `wet_snow_line` assimilation dates the panel also draws compact event-date coverage bars from `weights_wet_snow_line_YYYYMMDD.csv`, using the exact support-aware PF `value_model` scalars for the min / median / max spread at that event. In v1 those `wet_snow_line` values represent the basin-wide **50% wet-fraction crossing**, while the aspect-aware relative-threshold WSLA remains a diagnostics-only companion written to the same seasonal/profile CSV family. When that `50%` crossing is undefined, the overview keeps a true gap in the WSLA model and ensemble lines and omits observation dots / assimilation `x` markers for that date instead of falling back to `p95`. Observation `x` markers are only drawn on the panel for the actually assimilated variable, so `wet_snow_line` events do not mark the WSF panel unless `wet_snow` itself is assimilated. Add `--scf-model-csv` / `--wet-model-csv` / `--wsl-model-csv` to overlay specific member series or `--scf-env-csv` / `--wet-env-csv` / `--wsl-env-csv` to use custom envelopes. Plot mode can be switched with `--mode band|members` (pipeline default: `band`). When `<project-dir>/plots.yml` exists, the project pipeline also writes `results/plots/results/result_overview_custom.png` using the requested panel order and station panels from that YAML; custom panels now support `WSF`, `WSLA`, `scores-crpss`, `scores-ner`, and station-only `scores-zskill`.

## Setup point results (SWE / snow depth, member mode)

Generate setup-wide point plots (members only, legend shows just open loop + assimilation markers):

```powershell
docker compose run --rm oa python -m openamundsen_da.methods.viz.plots.project_ensemble results --setup-dir $setup --var-col swe --mode members --log-level INFO
docker compose run --rm oa python -m openamundsen_da.methods.viz.plots.project_ensemble results --setup-dir $setup --var-col snow_depth --mode members --log-level INFO
```

Outputs are written to `<setup>/results/plots/points/setup_results_point_<station>_{swe|snow_depth}_<setup>.png`. The setup pipeline calls the same functions with `mode=members` after each step and at the end.

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
  python -m openamundsen_da.methods.pf.assimilate_fraction `
  --project-dir $project `
  --step-dir $step `
  --ensemble prior `
  --date $date `
  --roi $roi
```

Optional flags: `--obs-csv <path>`, `--output <csv>`, `--log-level <LEVEL>`

Fraction DA remains a scalar particle-filter update. For `scf` and `wet_snow`, the model-side scalar is derived on the event-date observation-valid support; `wet_snow` is the WSF scalar. The written weights CSV keeps `value_model` as the assimilated support-aware value and also records `value_model_full_roi`, `value_model_obs_support`, `obs_support_n_valid`, and `obs_support_coverage_ratio`. `wet_snow_line` is the WSLA scalar: it reuses the same scalar PF path, but its residuals and likelihood sigmas live in meters of elevation. In v1 the assimilated WSLA is the support-aware **50% wet-fraction crossing**, while the weights CSV also stores the full-ROI companion diagnostic for envelope-style summaries. If the crossing is undefined the event becomes a no-op update with equal weights; there is no fallback to the retired highest-band WSLA or to `p95`. In project maps, each WSLA panel draws only the WSLA contour diagnosed from that panel's own field; observation WSLA is shown only in observation/reference panels.

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
    sigma_rh: 0.0
    sigma_sw: 0.0
```

## Plots

- Forcing per-station:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plots.forcing_ensemble `
  --step-dir $step `
  --ensemble prior
```

Optional flags: `--time-col`, `--temp-col`, `--precip-col`, `--start-date`, `--end-date`, `--resample`, `--rolling`, `--hydro-month`, `--hydro-day`, `--title`, `--subtitle`, `--output-dir`, `--backend`, `--log-level`

- Results per-station:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plots.results_ensemble `
  --step-dir $step `
  --ensemble prior
```

Optional flags: `--time-col`, `--var-col`, `--var-label`, `--var-units`, `--start-date`, `--end-date`, `--resample`, `--resample-agg`, `--rolling`, `--band-low`, `--band-high`, `--title`, `--subtitle`, `--output-dir`, `--backend`, `--log-level`

- ESS timeline:

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plots.assimilation.ess_timeline `
  --step-dir $step
```

Optional: `--normalized`, `--threshold <ratio>`, `--output <svg>`, `--backend`, `--log-level`

Project-level stitched forcing and point-result panels are written to `$setup/results/plots/points` with the setup identifier in the filenames.

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
    python -m openamundsen_da.methods.viz.plots.assimilation.weights `
    $weights
  ```

  When the CSV lives under `$setup/steps/step_XX_*/assim/`, the plot is written to `$setup/plots/assim/weights/step_XX_weights.png`.

  - Setup ESS timeline (all steps):

  ```powershell
  docker compose run --rm oa `
    python -c "from pathlib import Path; from openamundsen_da.methods.viz.plots.assimilation.ess_timeline import plot_setup_ess_timeline; plot_setup_ess_timeline(Path('$setup'))"
  ```

  This scans `steps/step_*/assim/weights_scf_*.csv` under `$setup` and writes the timeline to `$setup/results/plots/assim/ess/setup_ess_timeline_<setup_id>.png`.

- Setup-wide forcing/results (stitch all steps together):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plots.project_ensemble `
  forcing `
  --setup-dir $setup
```

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plots.project_ensemble `
  results `
  --setup-dir $setup `
  --var-col swe
```

- Setup-wide SCF (model + obs SCF):

```powershell
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plots.project_ensemble `
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
    python -m openamundsen_da.methods.viz.plots.station_variable `
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

- Project maps (compact project grid -> publication-style PNGs):

  Use `oa-da-plot-project-maps` to render generated DA-event maps plus optional custom YAML map recipes from a completed project. The renderer reads `results/grids/da_output_grids.nc`, setup grids/ROI/stations, and project observation summaries automatically. By default it generates one `da_*` map per assimilation event from `<project-dir>/project_*.yml`; `<project-dir>/maps.yml` is now for custom maps such as `setup_overview`.
  The custom `maps.yml` sidecar still uses the same panel catalog:

  ```yaml
  # Available panel kinds:
  # - overview                 # scale: 1000000 ; optional roi_label
  # - roi
  # - hillshade
  # - dem
  # - svf
  # - srf
  # - landcover
  # - snow_depth               # source: open_loop | ensemble_mean | analysis_mean | increment | analysis_increment
  # - swe                      # source: open_loop | ensemble_mean | analysis_mean | increment | analysis_increment
  # - liquid_water_content     # source: open_loop | ensemble_mean | analysis_mean | increment | analysis_increment
  # - fsc                      # source: open_loop | ensemble_mean | open_loop_binary | prior_probability | posterior_probability
  # - wet_snow                 # source: open_loop | ensemble_mean | prior_probability | posterior_probability
  # - uncertainty              # observation: scf | wet_snow
  # - wet_snow_line            # source: open_loop | prior_probability | posterior_probability | posterior
  # - wet_snow_elevation_fraction # source: open_loop | prior_probability | posterior_probability
  # - legend
  # - colorbar
  # Optional panel keys:
  # - title, name, date, legend, show_colorbar, show_scalebar, show_grid, show_hillshade, hillshade_extent
  # - observation (uncertainty only), show_roi, show_station_marker, show_stations_name, show_stations_elev
  # Optional recipe-level row zoom views:
  # row_views:
  #   - row: 1
  #     center: [643767, 5191680] # setup/project CRS by default
  #     zoom: 13                 # Google/Slippy-map zoom
  #     # center_crs: EPSG:4326
  #     # viewport_px: [1024, 1024]
  ```

  ```powershell
  docker compose run --rm oa `
    oa-da-plot-project-maps `
    --project-dir /data/projects/project_2022_2023 `
    --max-workers 4
  ```

  Outputs are split by type:
  - generated DA-event maps under `results/maps/da_events/`
  - custom YAML maps at the root of `results/maps/`

  Generated DA-event rows use four consistent columns: `open loop`, `prior`, `posterior`, and `reference`. Snow-state reference columns show `analysis_increment` (`posterior - prior`, so positive values mean DA added snow/water). FSC and wet-snow reference columns show the satellite observation. Generated FSC, WSF, WSLA, and elevation-band WSF rows use spatial prior/posterior probability maps where applicable; WSLA contours are panel-local, so model columns do not overlay observation WSLA. Top-level sub-domain SCF events use a taller same-file layout with a 2x2 snow-cover block above the 2x2 snow-depth response block; exact rerendering requires retained per-sub-domain grids. If the event resampling manifest reports skipped resampling, the map title is suffixed with `resampling skipped`.

  By default the renderer parallelizes across independent recipe PNGs inside the Docker container and clamps the effective worker count to `min(visible CPUs, selected recipes)`; use `--max-workers 1` to force sequential rendering. `oa-da-project` and merged sub-domain runs also render project maps automatically as a best-effort post-run stage. If a map fails because supporting data are missing, the pipeline logs a rerun command and continues.
  Project maps now use a simplified public panel catalog: context panels (`overview`, `roi`, `hillshade`, `dem`, `svf`, `srf`, `landcover`), result panels (`snow_depth`, `swe`, `liquid_water_content`, `fsc`, `wet_snow`, `uncertainty`, `wet_snow_line`, `wet_snow_elevation_fraction`), and optional support panels (`legend`, `colorbar`). `uncertainty` renders `*_uncertainty.tif` companion rasters for `observation: scf` or `observation: wet_snow` on the valid observation support. In prepared sub-domain projects, top-level project maps automatically overlay the configured sub-domain polygons from `subdomain_manifest.json` on ROI-bearing map panels and overview panels. Generated DA-event maps additionally mark sub-domain polygons as `no DA` only when that event was dropped for the sub-domain and recorded in `results/subdomain_dropped_events.csv`; healthy events with computed weights are not marked just because resampling was skipped. Recipe-level `row_views` can assign a shared Google/Slippy-map zoom extent to every panel in a row. `wet_snow` renders WSF, while `wet_snow_line` renders the wet-snow raster context together with a DEM contour at the diagnosed WSLA. `prior_probability` and `posterior_probability` sources render spatial ensemble probability fields; observation overlays are kept in observation/reference panels.

- Project plots (all post-run plots without rerunning DA):

  Use `oa-da-plot-project-plots` to recreate the full `results/plots/` tree from existing project outputs. The command reuses the same post-run plot orchestration as the project pipeline: forcing plots, setup point-result plots, weights, ESS timeline, and the result overview. It also rebuilds the ROI fraction envelopes in `results/misc/` first, because the overview plots depend on them. `wet_snow_line` weights plots use meter residuals, carry dedicated WSLA title/legend labels, and annotate skipped updates with the gate reason plus available model/observation WSLA context.

  ```powershell
  docker compose run --rm oa `
    oa-da-plot-project-plots `
    --project-dir /data/projects/project_2022_2023 `
    --plot-workers 4 `
    --max-workers 4
  ```

  The command expects an already finished project with step outputs under `<project-dir>/steps/`. It does not rerun openAMUNDSEN or data assimilation; it only regenerates plot artifacts under `results/plots/` and the fraction envelopes under `results/misc/`.

- Project PDF collection (curated project overview and DA maps):

  `oa-da-project` attempts this automatically at the end of the project run, after final plots, maps, and benchmark-dependent overview panels are current. It writes `results/reports/project_report.pdf`. Report generation is best-effort inside the project pipeline: if a required plot or map is missing, the run stays successful and the log prints the manual rerun command.

  Use `oa-da-project-pdf` to manually reassemble the compact report summary page, curated project overview outputs, diagnostics, and DA-event maps into a DIN A4 portrait PDF without rerunning the model. It does not regenerate any source plot or map; run `oa-da-plot-project-plots` and `oa-da-plot-project-maps` first when outputs are stale or missing. When running against a mounted source checkout with an older Docker image, use the equivalent `python -m openamundsen_da.methods.viz.reports` entry point so the command is loaded from the checkout.

  ```powershell
  docker compose run --rm oa `
    python -m openamundsen_da.methods.viz.reports `
    --project-dir /data/projects/project_2022_2023
  ```

  The PDF starts with a generated one-page project report containing basic setup YAML settings, wet-snow classification and liquid-water-content settings, DA-event counts, computing-cost stats from project logs and `results/plots/perf/project_perf_metrics.csv` when available, plus a bottom `Content` table with page numbers first and section names second. It then includes `result_overview.png`, optional `result_overview_custom.png`, `setup_overview.png`, all `setup_weights_overview*.png` pages, station snow-depth point plots on one page, `performance_scores.png`, `project_perf.png`, and generated DA-event maps under `results/maps/da_events/da_<n>.png` in temporal order. Source PNGs are placed at their shared export-DPI size rather than scaled down to fit a page; consecutive DA maps are packed onto a page only while the reserved bottom gap is preserved. Standalone per-event weights plots and other remaining plot/map PNGs are not included. Missing required overview, setup map, setup weights overview, or DA map outputs cause a fail-fast error listing all paths to regenerate.

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

The pipeline drives each step in order, assimilates SCF on the _next_ step's start date, resamples the resulting weights to the posterior, and rejuvenates that posterior into the next prior before proceeding. Assimilation looks for the single-row CSV `obs_scf_SNOWCOVER_YYYYMMDD.csv` inside `<step>/obs/` for the date being processed; generate those files with `openamundsen_da.observer.satellite_scf` after you summarize your snow-cover rasters into `scf_summary.csv`. `setup.py` never reads source observation rasters directly, so the CSV must already reflect any filtering or thresholding you want applied.

Outputs

- Per-step runs in `<step>/ensembles/{prior,posterior}` (open_loop + members)
- Weights and indices in `<step>/assim/`
- Rejuvenated next-step prior (members + open_loop with state_pointer.json)
- Compact data assimilation summary grids in `<project>/results/grids/da_output_grids.nc`
  - Per variable `<var>`: `open_loop_<var>`, `ens_mean_<var>`, `ens_std_<var>`, `ens_min_<var>`, `ens_max_<var>`, `increment_<var>`, and event analysis fields `analysis_mean_<var>` / `analysis_increment_<var>` where assimilation weights are available
  - The exported variables and metrics follow `data_assimilation.output.grids.variables[*]`; omitted output-grid config preserves the legacy all-variable/all-metric behavior
  - `increment_<var>` is the open-loop departure: `ens_mean_<var> - open_loop_<var>`
  - `analysis_increment_<var>` is the DA-event increment: `analysis_mean_<var> - ens_mean_<var>`; positive values mean the event added snow/water to the ensemble mean
  - Time axis spans the full project timeline across all steps (not only the last step)
- Setup plots under `<setup_dir>/plots/{forcing,results}`
- Project-level plots under `<project>/results/plots/{results,perf,points,assim/{weights,ess,scores}}`
- Project-level misc artifacts under `<project>/results/misc`
- Project maps under `<project>/results/maps`, with generated DA maps under `da_events/` and optional custom YAML maps at the root
- Project report PDFs under `<project>/results/reports`
- When model SCF is enabled, daily ROI-mean SCF per member is written to `<step>/ensembles/prior/<member>/results/point_scf_roi.csv`.
- Full-ROI daily mean SWE and snow depth are written to `<step>/ensembles/prior/<member>/results/point_swe_roi.csv` and `<step>/ensembles/prior/<member>/results/point_snow_depth_roi.csv`.
- The combined project result overview plot (`results/plots/results/result_overview.png`) now shows SCF, wet-snow, ROI mean SWE, and ROI mean snow depth together.
  Setup results plots now show the ensemble mean, the 90% envelope, and the open loop by default; individual members are hidden unless `--show-members` is passed to the plot CLI. Wet-snow setup plots overlay available observations from `obs/<setup>/wet_snow_summary.csv` automatically.
  At the end of the setup run, per-step weights plots (`step_XX_weights.png`) and the setup ESS timeline (`setup_ess_timeline_<setup_id>.png`) are also generated under `<project_dir>/results/plots/assim/{weights,ess}`.
  Single-domain projects default to compact retention (`data_assimilation.output.retention: compact`), which prunes heavy member grid artifacts after writing `da_output_grids.nc`. Sub-domain projects default to `retention: full` so generated DA-event maps can be regenerated exactly after the run.

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

To create an empty setup layout with `steps/step_*` folders and minimal step YAMLs, use the structured data assimilation block:

```yaml
start_date: 2017-10-01
end_date: 2018-09-30
data_assimilation:
  assimilation_events:
    - date: 2017-11-23
      variable: scf
      product: SNOWCOVER
    - date: 2018-03-19
      variable: wet_snow_line
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

### Performance monitoring (CPU / RAM / disk)

A minimal monitor samples system CPU%, RAM%, filesystem disk pressure, and throttled project directory size (enabled by default for `oa-da-project`).
Outputs under `<project>/results/plots/perf/`:
- `project_perf_metrics.csv` (timestamp, CPU/RAM columns, filesystem used/free/total GB, and project size GB)
- `project_perf.png` (CPU/RAM/filesystem-used % plus project size and free disk GB)

Suggested intervals: sample every 5-10 seconds; refresh the plot every 30-60 seconds; scan recursive project size every 300 seconds or longer for large runs.

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

### Scientific Benchmarking

`oa-da-project` now always ends with an observation-based benchmarking stage. It writes:

- `results/benchmark/manifest.json`
- `results/benchmark/cases/*.csv`
- `results/benchmark/scores/*.csv`
- `results/benchmark/tables/project_summary.csv`
- `results/benchmark/tables/update_summary.csv`
- `results/benchmark/summary.md`
- `results/plots/assim/scores/performance_scores.png`

After benchmarking and any benchmark-dependent overview rerender, `oa-da-project` also attempts to assemble `results/reports/project_report.pdf`. Missing report prerequisites are logged as warnings with a rerun command and do not fail the completed project run.

The raw benchmark backend still scores whole-project propagated `da_informed_ensemble` skill against `open_loop` and, on assimilation dates, explicit analysis-time `prior` and weighted `posterior` skill. The headline plot is intentionally narrower: it shows only assimilation-date `prior` and `posterior` skill (`CRPSS`, `NER`) for assimilated and transfer-observed variables on the DA dates themselves, and adds a third station-only `zSkill` panel when sigma-aware station scores are available, while `project_summary.csv` keeps the whole-project propagated view. `wet_snow` is scored as WSF, while `wet_snow_line` is scored as its own WSLA ROI scalar in meters against `wet_snow_line_diagnostics.csv` and `point_wet_snow_line_roi.csv`, not as a WSF proxy. Results are split into `assimilation_fit`, `semi_independent`, and `independent`: `semi_independent` means the exact variable/date pair was not assimilated, but a same-variable or sister-station assimilation has already happened by that date, while `independent` means no same-variable assimilation has happened yet by that date and no active sister-station linkage applies.

You can add extra independent benchmark families from the current DA-supported set (`scf`, `wet_snow`, `wet_snow_line`, `station_hs`, `station_swe`) in project YAML:

```yaml
data_assimilation:
  benchmark:
    independent_variables:
      - station_swe
    plots: true
    output_dir: results/benchmark
```

You can also rerun the same benchmark logic manually on an existing finished project:

```powershell
docker compose run --rm oa `
  oa-da-benchmark `
  --project-dir $project
```

Running the monitor manually

You can also attach the monitor manually to an existing setup directory (for example,
while a setup run is already in progress from another shell):

```powershell
$setup = "$project/projects/project_YYYY-YYYY"

docker compose run --rm oa `
  oa-da-perf-monitor `
  --project-dir $setup `
  --sample-interval 5 `
  --plot-interval 30 `
  --disk-scan-interval 300
```

This foreground command will keep updating the CSV and plot until interrupted
with `Ctrl+C`.


## State cleanup (free disk space)

- Automatic: set `data_assimilation.restart.cleanup_after_setup: true` (default) in project YAML to delete member state pickle files after a successful project run.
- Manual (ignores the toggle): clean one or all projects via Docker Compose.

All projects under a setup:

```powershell
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.cleanup \
  --setup-dir /data/your_setup \
  --all-projects \
  --log-level INFO
```

Single project:

```powershell
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.cleanup \
  --setup-dir /data/your_setup \
  --project-dir /data/your_setup/projects/project_YYYY-YYYY \
  --log-level INFO
```

If you rebuilt the image with the latest code, you can replace the `python -m ...cleanup` line with the shorter `oa-da-clean-project`.
Cleanup only removes matching state pickle files. It leaves `state_pointer.json`, grids, maps, reports, manifests, logs, and sub-domain workspaces in place; grid artifact pruning is controlled separately by `data_assimilation.output.retention`.


## Sub-domain Mode

Use `oa-da-subdomain` to split a large setup into non-overlapping sub-domains. There are two workflows:

For a start-to-finish guide, see `docs/guides/subdomain-runbook.md`.

### Data Assimilation Sub-domain Workflow

This is the existing openAMUNDSEN-DA workflow: one independent data assimilation project per sub-domain, project-level reports, then compact data assimilation grid merge.

Minimal DA flow:
- Prepare DA sub-domain setups from ROI polygons:
  `oa-da-subdomain prepare --setup-dir <setup> --project-dir <setup>/projects/<project> --roi <setup>/env/subdomains.gpkg --id-field id`
- Run all sub-domains in parallel:
  `oa-da-subdomain run --project-dir <setup>/projects/<project>`
- Merge data assimilation grids (hard mosaic, no interpolation/blending):
  `oa-da-subdomain merge --project-dir <setup>/projects/<project>`

DA defaults:
- Sub-domain root is `<project>/subdomains` (override with `--subdomain-root`).
- Manifest path is `<subdomain_root>/subdomain_manifest.json` (or pass `--manifest` explicitly).
- Each sub-domain run lives under `<subdomain_root>/<subdomain_id>/`.
- Project-level outputs are written under `<project>/results/`.
- Compact data assimilation grid output is `<project>/results/grids/da_output_grids.nc`.
  - Variables in `da_output_grids.nc`: `open_loop_<var>`, `ens_mean_<var>`, `ens_std_<var>`, `ens_min_<var>`, `ens_max_<var>`, `increment_<var>`, and event analysis fields `analysis_mean_<var>` / `analysis_increment_<var>` when weights are available.
- Sub-domain reports are written under `<project>/results/subdomain_*.csv`.
- Point outputs and plots remain inside each sub-domain project.
- Station selection uses a 50 km default buffer (`--station-buffer-km`).
- Tiny polygon overlaps are tolerated up to 100 m^2 (`--overlap-area-tol-m2`), with optional sliver correction (`--sliver-fix-m`).
- Sub-domain mode requires at least two polygons in the ROI file.
- `--id-field` must exist in the regions file; there is no automatic fallback to another field name.
- If `--roi` is omitted in sub-domain prepare/pipeline, `<setup>/env/subdomains.gpkg` is preferred and `<setup>/env/roi.gpkg` is the fallback.
- Sub-domain runs fail fast if configured assimilation events are not available in the local sub-domain observation summaries.
- Projects may enable `data_assimilation.subdomain_event_filter` to drop unavailable SCF, wet-snow, or station events per sub-domain after local observation summaries are generated. Dropped events are recorded in each sub-domain run manifest and in `<project>/results/subdomain_dropped_events.csv`; the final kept/dropped event table is written to `<project>/results/event_plan_by_subdomain.csv`.
- Station benchmark variables are pruned from copied sub-domain project YAMLs when the local station subset contains no benchmark-enabled station observations, allowing mixed FSC-only and FSC+station sub-domain projects in one run.
- Configured `output_data.timeseries.points` are filtered to the active sub-domain ROI when sub-domain setup YAML files are generated.
- Sub-domain projects default to full retention (`data_assimilation.output.retention: full`) and keep the sub-domain NC grids needed for exact DA-event map regeneration. Set `retention: compact` only if you knowingly trade away exact spatial DA-event map rerendering for disk savings.

Example sub-domain event filter:

```yaml
data_assimilation:
  subdomain_event_filter:
    enabled: true
    drop_unavailable: true
    variables:
      scf:
        max_cloud_fraction: 0.20
      station_hs:
        min_active_stations: 1
        max_time_delta_hours: 36
    subdomains:
      AT-07-20:
        variables:
          scf:
            max_cloud_fraction: 0.25
```

### Plain openAMUNDSEN Model Workflow

This workflow uses the same geometry validation, ROI rasterization, grid cropping, meteo station subsetting, manifest, and hard-mosaic merge helpers, but skips data assimilation project setup and raw observation preprocessing. It runs the installed `openamundsen` executable once per prepared sub-domain.

```bash
oa-da-subdomain model-pipeline \
  --setup-dir /data/subdomains \
  --regions /data/subdomains/env/subdomains.gpkg \
  --max-workers 24 \
  --overwrite
```

Docker usage:
- Run from this repository root, where `compose.yml` lives.
- Mount a plain openAMUNDSEN setup with `PROJ=/absolute/path/to/setup`; inside the container it is available as `/data`.
- The Docker image already contains the `openamundsen` executable used by `model-run`.

```bash
PROJ=/absolute/path/to/setup docker compose run --rm oa \
  oa-da-subdomain model-pipeline \
  --setup-dir /data \
  --regions /data/env/subdomains.gpkg \
  --max-workers 24 \
  --overwrite
```

Minimal plain-model setup layout:

```
setup/
  <setup-name>.yml        # or setup.yml; plain openAMUNDSEN config
  env/
    subdomains.gpkg       # non-overlapping polygons with an id column, or pass --id-field
  grids/
    dem_<domain>_<resolution>.asc
    lc_<domain>_<resolution>.asc
    ...                   # any additional grids required by the setup
  meteo/
    stations.csv
    <station>.csv
    ...
```

The setup YAML must define the normal openAMUNDSEN domain settings (`domain`, `resolution`, `crs`, `timestep`, `timezone`), `start_date`, `end_date`, `input_data.grids.dir`, `input_data.meteo.dir`, and the desired `output_data` grid variables. `projects/` and `obs/` directories are not required for plain model mode. The `subdomains/model/` tree is generated by `model-prepare` or `model-pipeline`.

Model defaults and outputs:
- Source setup YAML must define `start_date` and `end_date`.
- Sub-domain root is `<setup>/subdomains/model` (override with `--subdomain-root`).
- Each generated sub-domain setup remains a plain openAMUNDSEN config with sub-domain `domain`, grid/meteo dirs, `results_dir`, and ROI grid.
- Per-subdomain model outputs are written to `<setup>/subdomains/model/<id>/results/`.
- Merged grid outputs are written to `<setup>/subdomains/model/results/grids/`.
- Only grid outputs under each `<id>/results/grids/` are merged in v1; point/timeseries outputs are left per sub-domain.
- Merge is hard mosaic only (no interpolation, blending, or boundary smoothing).
- Per-subdomain run diagnostics are written to `<setup>/subdomains/model/<id>/run.log` and `<setup>/subdomains/model/<id>/run_manifest.json`.

Ready-made example:
- setup: `examples/subdomains`
- regions file: `examples/subdomains/env/subdomains.gpkg` (pass with `--roi`)
- note: this example can be used with either DA commands or the new `model-*` commands.

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

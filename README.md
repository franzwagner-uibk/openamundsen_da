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

## General Information

- Per-member logs: `<member_dir>/logs/member.log`.
- Quote bind mounts like "${repo}:/workspace" to avoid PowerShell parsing issues.
- Use forward slashes for in-container paths (`/workspace`, `/data`).
- Prefer `--backend SVG` for plots on all platforms.

### Warm Start Policy (Strict)

- Step 00 (first step): always cold start; no state is read. A fresh model state is saved at the end of the run for each member.
- Steps â‰¥ 01: always warm start; restart is mandatory. The runner reads a single pointer file at member root:
  - `<step>/ensembles/prior/<member>/state_pointer.json` with a JSON object: `{ "path": "/data/.../model_state.pickle.gz" }`.
  - Only this pointer is used. Local files under `results/` are ignored to avoid ambiguity.
  - If the pointer is missing/corrupt or the target file fails integrity, the run aborts (no silent cold starts).
- Rejuvenation writes the pointer into the next stepâ€™s prior member root. Pointers should use absolute Linux paths inside the container (start with `/data/`).

## Theory Primer (Very Short)

- Prior: An ensemble represents p(x_t) via N members created by perturbing forcings.
- Observation: Satellite SCF y_t with uncertainty sigma (quality-driven where available).
- H(x): Maps model outputs to SCF in the AOI (threshold or logistic).
- Likelihood: For each member i, L_i = N(y_t - H(x_t^i); 0, sigma^2).
- Weights: w_i = L_i / sum_j L_j (computed in log-space for stability).
- ESS: 1/sum(w_i^2) quantifies degeneracy; low ESS suggests resampling.
- Next steps (future): resampling and rejuvenation to form the posterior/next prior.

## Docker Quick Start

- Install Docker Desktop (Windows/macOS) or Docker Engine (Linux).
- On Windows, enable WSL2 and share the drive that contains your repo/data in Docker Desktop Settings.
- From repo root, build the image once:

```
docker build -t oa-da .
```

Editable run (Compose + .env):

```
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_results_ensemble `
  --step-dir /data/propagation/season_2017-2018/step_00_init `
  --ensemble prior `
  --time-col time `
  --var-col swe `
  --start-date 2017-11-01 `
  --end-date 2018-04-30 `
  --resample D `
  --rolling 1 `
  --title "Model Results Ensemble (prior)" `
  --backend SVG `
  --output-dir /data/propagation/season_2017-2018/step_00_init/plots/results
```

Minimal files and keys used by the workflow.

- project.yml (at project root)

  - data_assimilation.prior_forcing (required)
    - ensemble_size: int
    - random_seed: int
    - sigma_t: float # temperature additive stddev (degC)
    - mu_p: float # log-space mean for precip factor
    - sigma_p: float # log-space stddev for precip factor
  - data_assimilation.h_of_x (H(x) configuration used by assimilation)
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
  - h_of_x (optional fallback; same keys as in project.yml)

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

## Docker Compose (Recommended)

Compose simplifies resource limits and paths for all users via a small `.env`.

- One-time build (from repo root):

```
docker build -t oa-da .
```

- Create your environment file:

```
copy .env.example .env    # Windows
# or
cp .env.example .env      # Linux/macOS
```

- Edit `.env` and set:

  - `REPO` â†’ absolute path to this repo on your machine
  - `PROJ` â†’ absolute path to your project data (the example uses `examples/test-project`)
  - `CPUS`, `MEMORY` â†’ per-container limits (keep within your VM/global limits)
  - `MAX_WORKERS` â†’ parallel openAMUNDSEN runs inside the container

- Enable Compose compatibility so limits apply without Swarm:

Windows PowerShell (persist):

```
setx COMPOSE_COMPATIBILITY 1
```

Linux/macOS (current shell):

```
export COMPOSE_COMPATIBILITY=1
```

- Optional (Windows/WSL2): set a global VM ceiling via `%UserProfile%\\.wslconfig`:

```
[wsl2]
memory=25GB
processors=19
swap=16GB
localhostForwarding=true
```

Apply by quitting Docker Desktop â†’ `wsl --shutdown` â†’ start Docker Desktop. Verify:

```
wsl -d docker-desktop grep -i memtotal /proc/meminfo   # ~25 GB
wsl -d docker-desktop sh -c "grep -c ^processor /proc/cpuinfo"  # 19
wsl -d docker-desktop grep -i swaptotal /proc/meminfo  # ~16 GB
```

- Run the launcher explicitly (resources via Compose, command provided here):

```
docker compose run --rm oa `
  python -m openamundsen_da.core.launch `
  --project-dir /data `
  --season-dir /data/propagation/season_2017-2018 `
  --step-dir /data/propagation/season_2017-2018/step_00_init `
  --ensemble prior `
  --max-workers 4 `
  --log-level INFO
```

You can temporarily override knobs without editing `.env`:

```
# Windows PowerShell (this shell only)
$env:MAX_WORKERS=2; $env:MEMORY="20g"; docker compose run --rm oa `
  python -m openamundsen_da.core.launch `
    --project-dir /data `
    --season-dir /data/propagation/season_2017-2018 `
    --step-dir /data/propagation/season_2017-2018/step_00_init `
    --ensemble prior `
    --max-workers $env:MAX_WORKERS `
    --log-level INFO

# (Compose uses .env; override in this PowerShell session only)
# Example above shows adding MAX_WORKERS/MEMORY for a single run.
```

Notes

- Volumes: `${REPO}` â†’ `/workspace` (source), `${PROJ}` â†’ `/data` (project root in commands)
- Resource limits are set via `deploy.resources.limits` in `compose.yml` and applied when `COMPOSE_COMPATIBILITY=1` is set.
- `compose.yml` pins numerical library threads to 1 per worker to avoid oversubscription.
- This service has no default command. Always append the command you want to run, as shown above.

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
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_results_ensemble `
  --step-dir /data/propagation/season_2017-2018/step_00_init `
  --ensemble prior `
  --time-col time `
  --var-col swe `
  --start-date 2017-11-01 `
  --end-date 2018-04-30 `
  --resample D `
  --rolling 1 `
  --title "Model Results Ensemble (prior)" `
  --backend SVG `
  --output-dir /data/propagation/season_2017-2018/step_00_init/plots/results
```

Notes

- For all steps, forcing is perturbed from the step start_date to the season end_date (season-forward). This ensures the next propagation has sufficient forcing without having to rebuild mid-season.

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

docker compose run --rm oa `  python -m openamundsen_da.core.launch`
--project-dir /data `  --season-dir /data/propagation/season_2017-2018`
--step-dir /data/propagation/season_2017-2018/step_00_init `  --ensemble prior`
--max-workers 2`
--log-level INFO `
--overwrite

```

Notes

- The launcher also runs the `open_loop` alongside `member_*` so a continuous, unperturbed reference is produced for each step under `<step>/ensembles/prior/open_loop/results`.
- For steps after the first, the open loop is warm-started via a state pointer created during rejuvenation.

### Observation Processing

Context

- MOD10A1 preprocess converts HDF to analysis-ready GeoTIFFs and maintains a season summary CSV. A simple NDSI thresholding yields snow/no-snow classification, and SCF is the fraction of snow-classified pixels inside the AOI.
- Single-image SCF extraction computes an observed SCF for a specific date and AOI; this is the observation y_t for assimilation.
- Note: The Docker image includes the GDAL HDF4 plugin; if you ever rebuild dependencies, ensure hdf4 and libgdal-hdf4 are present.

- MOD10A1 preprocess (HDF -> GeoTIFF + season summary):

```

docker compose run --rm oa `  python -m openamundsen_da.observer.mod10a1_preprocess`
--input-dir /data/obs/MOD10A1_61_HDF `  --project-dir /data`
--season-label season_2017-2018 `  --aoi /data/env/GMBA_Inventory_L8_15422.gpkg`
--target-epsg 25832 `  --resolution 500`
--max-cloud-fraction 0.1 `
--overwrite

```

- Single-image SCF extraction (preprocessed GeoTIFF):

```

docker compose run --rm oa `  python -m openamundsen_da.observer.satellite_scf`
--raster /data/obs/season_2017-2018/NDSI_Snow_Cover_20180110.tif `  --region /data/env/GMBA_Inventory_L8_15422.gpkg`
--step-dir /data/propagation/season_2017-2018/step_00_init

```

- Plot SCF summary (SVG backend recommended):

```

docker compose run --rm oa `  python -m openamundsen_da.observer.plot_scf_summary`
/data/obs/season_2017-2018/scf_summary.csv `  --output /data/obs/season_2017-2018/scf_summary.svg`
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

Recommended usage (all members)

- Most users should run the allâ€‘members assimilation command, which computes H(x) for every member and produces the weights CSV. Internally, it calls the singleâ€‘member H(x) for each member.

```
docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.assimilate_scf `
  --project-dir /data `
  --step-dir /data/propagation/season_2017-2018/step_00_init `
  --ensemble prior `
  --date 2018-01-10 `
  --aoi /data/env/GMBA_Inventory_L8_15422.gpkg
```

Singleâ€‘member (debugging and tuning)

- The singleâ€‘member tool is intended for quick checks (paths/CRS/AOI), troubleshooting, or tuning `h0`/`k`. It requires a specific memberâ€™s `results` directory.

```
docker compose run --rm oa `
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
docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.assimilate_scf `
  --project-dir /data `
  --step-dir /data/propagation/season_2017-2018/step_00_init `
  --ensemble prior `
  --date 2018-01-10 `
  --aoi /data/env/GMBA_Inventory_L8_15422.gpkg
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
docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.plot_weights `
  /data/propagation/season_2017-2018/step_00_init/assim/weights_scf_20180110.csv `
  --output /data/propagation/season_2017-2018/step_00_init/assim/weights_scf_20180110.svg `
  --backend SVG
```

- ESS timeline (from all `weights_scf_*.csv`):

```
docker compose run --rm oa `
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
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_forcing_ensemble `
  --step-dir /data/propagation/season_2017-2018/step_00_init `
  --ensemble prior `
  --time-col date `
  --temp-col temp `
  --precip-col precip `
  --start-date 2017-10-01 `
  --end-date 2018-03-31 `
  --resample D `
  --rolling 1 `
  --title "Forcing Ensemble (prior)" `
  --backend SVG `
  --output-dir /data/propagation/season_2017-2018/step_00_init/plots/forcing
```

- Results per-station (e.g., SWE from point CSVs):

```
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_results_ensemble `
  --step-dir /data/propagation/season_2017-2018/step_00_init `
  --ensemble prior `
  --time-col time `
  --var-col swe `
  --start-date 2017-11-01 `
  --end-date 2018-01-10 `
  --resample D `
  --rolling 1 `
  --title "Model Results Ensemble (prior)" `
  --backend SVG `
  --output-dir /data/propagation/season_2017-2018/step_00_init/plots/results
```

- Results module details (`openamundsen_da/methods/viz/plot_results_ensemble.py`):
  - Scans `<step>/ensembles/<ensemble>/{open_loop,member_XXX}/results/point*/*.csv`.
  - Expects a datetime column (default `time`) and one variable column (e.g., `swe` or `hs`).
  - Options: `--resample` (pandas rule), `--rolling` (samples), `--start-date/--end-date`.
  - Outputs one PNG per station into `--output-dir` (default: `<step>/plots/results`).
  - Overlays open_loop if available (members only otherwise).

### Help (no install needed)

Compose sets PYTHONPATH=/workspace so modules import directly from your repo without an editable install.

```
docker compose run --rm oa `
  python -m openamundsen_da.core.prior_forcing `
  --help

docker compose run --rm oa `
  python -m openamundsen_da.core.launch `
  --help

docker compose run --rm oa `
  python -m openamundsen_da.observer.mod10a1_preprocess `
  --help

docker compose run --rm oa `
  python -m openamundsen_da.observer.satellite_scf `
  --help

docker compose run --rm oa `
  python -m openamundsen_da.observer.plot_scf_summary `
  --help

docker compose run --rm oa `
  python -m openamundsen_da.methods.h_of_x.model_scf `
  --help

docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.assimilate_scf `
  --help

docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.plot_weights `
  --help

docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.plot_ess_timeline `
  --help
```

## Resampling & Posterior Ensemble

Create a posterior ensemble from single-date weights using systematic resampling. If the effective sample size (ESS) is above a threshold, mirroring (no resampling) can be used.

```
$date = "2018-01-10"
$step = "/data/propagation/season_2017-2018/step_00_init"
$dateTag = ($date -replace '-', '')
$weights = "$step/assim/weights_scf_$dateTag.csv"

docker compose run --rm oa `
  python -m openamundsen_da.methods.pf.resample `
  --project-dir /data `
  --step-dir $step `
  --ensemble prior `
  --weights $weights `
  --target posterior `
  --ess-threshold-ratio 0.5 `
  --seed 123 `
  --overwrite
```

Outputs

- Posterior members under `<step>/ensembles/posterior/member_XXX`.
- Mapping CSV and manifest under `<step>/assim/resample_*`.
  - The mapping CSV includes `posterior_member_id,source_member_id,weight`.

No duplication policy

- Resampling does not duplicate large state files. Each posterior member contains:
  - `state_pointer.json` at member root pointing to the source member's saved state file.
  - `source_pointer.json` at member root with the absolute path of the source member and its weight.
  - Empty `meteo/` and `results/` folders for compatibility.
  - This keeps disk usage low and is portable across Windows/macOS/Linux.

Notes

- The CLI reads defaults from `project.yml` (block `resampling`), including `algorithm`, `ess_threshold`, and falls back to `data_assimilation.prior_forcing.random_seed` for `seed` when present.
- Default materialization copies files (robust and portable). Use `--prefer-symlink` to opt in to symlinks (with copy fallback).
- Manifest also reports ESS, N, threshold, and the mapping CSV; uniqueness stats are logged by the CLI.
- Thresholds: you can specify `--ess-threshold-ratio` in (0,1] to resample when ESS < ratio\*N (e.g., 0.5 for 50%). If you pass `--ess-threshold` in (0,1], it is treated as a ratio; otherwise as an absolute count. In `project.yml`, set either `data_assimilation.resampling.ess_threshold_ratio` (preferred) or `data_assimilation.resampling.ess_threshold`. For backward compatibility, a top-level `resampling` block is also recognized.

When to skip resampling

- High ESS (near N) means weights are close to uniform; resampling would duplicate/drop members needlessly and can reduce diversity (particle impoverishment).
- Skipping avoids injecting Monte Carlo noise when the observation is weak or only mildly informative; the posterior effectively equals the prior.
- Efficiency and reproducibility: mirroring is fast and deterministic; the tool still writes indices and a manifest for traceability.
- Practical thresholds: choose `--ess-threshold-ratio` in 0.5â€“0.67. With N members and ratio r, the absolute threshold is rÂ·N; the tool skips when `ESS >= rÂ·N`.
- To force resampling: raise the ratio above ESS/N, or pass `--ess-threshold 0` to always resample.

## Warm Start (Restart)

Use saved model state from the end of a run as the initial state for the next step.

Config (project.yml):

```
data_assimilation:
  restart:
    use_state: false            # set true for steps after the first
    dump_state: true            # save state at end of runs
    state_pattern: model_state.pickle.gz
```

CLI (PowerShell-safe line breaks with `):

```
docker compose run --rm oa `
  python -m openamundsen_da.core.launch `
  --project-dir /data `
  --season-dir /data/propagation/season_2017-2018 `
  --step-dir /data/propagation/season_2017-2018/step_01_20180110-20180316 `
  --ensemble prior `
  --max-workers 4 `
  --log-level INFO

Note: The launcher supports pointer-based states. If `model_state.pickle.gz` is
not present in a member's `results/`, it will read `state_pointer.json` at the
member root (or `results/state_pointer.json` for backward compatibility) to
locate the external state file.

## Rejuvenation (Posterior â†’ Prior for Next Step)

Create a rejuvenated prior ensemble for the next step by adding light
perturbations to meteo and carrying forward the saved state via a pointer.

Modes

- Compound (default): perturb the resampled source memberâ€™s meteo again for the
  next window (adds small diversity on top of the prior perturbation).
- Rebase: ignore the source memberâ€™s prior perturbation and perturb the
  open_loop meteo for the next window (no compounding).

Configure in `project.yml` under `data_assimilation.rejuvenation`:

```

rejuvenation:
sigma_t: 0.2
sigma_p: 0.2
rebase_open_loop: false # set true to perturb open_loop instead of compounding

```

```

docker compose run --rm oa `  python -m openamundsen_da.methods.pf.rejuvenate`
--project-dir /data `  --prev-step-dir /data/propagation/season_2017-2018/step_00_init`
--next-step-dir /data/propagation/season_2017-2018/step_01_20180110-20180316

```

Rebase (single run override):

```

docker compose run --rm oa `  python -m openamundsen_da.methods.pf.rejuvenate`
--project-dir /data `  --prev-step-dir /data/propagation/season_2017-2018/step_00_init`
--next-step-dir /data/propagation/season_2017-2018/step_01_20180110-20180316 `
--rebase-open-loop

```

Behavior

- Reads `data_assimilation.rejuvenation.{sigma_t,sigma_p}` from `project.yml`.
- For each posterior member in the previous step:
  - Finds its source member (from `source_pointer.json`).
  - Filters meteo to the next step window, applies dT ~ N(0, sigma_t), f_p ~ LogNormal(0, sigma_p).
  - Writes meteo to next step `ensembles/prior/member_XXX/meteo`.
  - Copies `state_pointer.json` to the next step prior member root so warm start can load the saved state without duplication.
```

Notes

- Runner looks for `state_pattern` inside each memberâ€™s `results` dir. If it contains wildcards, it picks the newest match.
- Current implementation supports gzip+pickle files (e.g., `*.pickle.gz`). Support for zip-based states can be added once the file format is confirmed.

## Troubleshooting

- Python package not found in container when running help:
  - The image should already include the package. Compose exports PYTHONPATH=/workspace; if you prefer, you can rebuild the image to bake the package.
    `python -m openamundsen_da.core.launch \`
    `--help`
- HDF not recognized in preprocess:
  - Rebuild the image to include `hdf4` and `libgdal-hdf4` (already in environment.yml). Verify: `gdalinfo --formats | findstr HDF4`.
- Windows bind mounts and file metadata:
  - Some operations (e.g., copy2) may fail to preserve times/permissions. The code falls back to a content-only copy automatically.
- Plots crash or produce no output on Windows:
  - Use `--backend SVG` to avoid backend/DLL issues.

## Plot: Season Ensemble

Create season-wide plots that stitch all step segments together for a given season. Two modes are available:

- Forcing: temperature (timeseries) and cumulative precipitation (hydrological year)
- Results: model outputs per station (e.g., SWE or snow_depth)

The script autodiscovers steps under a season directory (e.g., `.../propagation/season_2017-2018`), draws dashed vertical lines at each assimilation start (the start of each step for i >= 1), and saves figures into `<season_dir>/plots/{forcing,results}` with the season identifier in the filename.

Examples

```
# Forcing (two panels). Hydrological year starts on Oct 1 by default.
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_season_ensemble `
  forcing `
  --season-dir /data/propagation/season_2017-2018 `
  --hydro-month 10 `
  --hydro-day 1 `
  --resample D `
  --rolling 1 `
  --backend Agg

# Results (SWE). Autostops one month after all members reach zero unless an earlier --end-date is provided.
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_season_ensemble `
  results `
  --season-dir /data/propagation/season_2017-2018 `
  --var-col swe `
  --resample D `
  --rolling 1 `
  --backend Agg

# Filter stations and set a season window
docker compose run --rm oa `
  python -m openamundsen_da.methods.viz.plot_season_ensemble `
  results `
  --season-dir /data/propagation/season_2017-2018 `
  --station point_station_001.csv `
  --station point_station_002.csv `
  --start-date 2017-11-01 `
  --end-date 2018-06_01
```

Flags

- `--station <filename>` repeatable; `--max-stations <N>` to limit processing
- `--start-date`, `--end-date` accept `YYYY-MM-DD`; the end date also accepts `YYYY-06_01`
- Forcing: `--date-col`, `--temp-col`, `--precip-col`, `--hydro-month`, `--hydro-day`, `--resample`, `--rolling`
- Results: `--time-col`, `--var-col`, `--var-label`, `--var-units`, `--resample`, `--resample-agg`, `--rolling`, `--band-low`, `--band-high`

Outputs are written to `<season_dir>/plots/forcing` and `<season_dir>/plots/results` and include the season identifier (e.g., `season_results_point_001_swe_2017-2018.png`).

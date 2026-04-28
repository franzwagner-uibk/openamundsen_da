---
layout: default
title: Command-Line Interface
parent: Guides
nav_order: 1
---

# Command-Line Interface
{: .no_toc }

Complete reference for all CLI commands.
{: .fs-6 .fw-300 }

<details markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Overview

The package provides **18 CLI entry points** for workflow automation, organized into 5 categories:

1. **Core Workflow** - Main pipeline commands
2. **Data Assimilation** - data assimilation-specific operations
3. **Wet Snow Analysis** - Wet snow processing
4. **Visualization** - Plotting commands
5. **Utilities** - Helper tools

All commands are available as:
- Python modules: `python -m openamundsen_da.MODULE.COMMAND`
- CLI scripts: `oa-da-COMMAND` (after installation)

---

## Core Workflow

### oa-da-project

**Main setup pipeline orchestrator**

Runs the complete setup data assimilation cycle: prior forcing → ensemble run → assimilation → resampling → rejuvenation.
The pipeline now also runs the scientific benchmark stage automatically at the end of every project.

```bash
oa-da-project \
  --setup-dir /data \
  [OPTIONS]
```

**Required Arguments:**
- `--setup-dir PATH` - Setup directory (project auto-detected if `project.yml` is in a parent)

**Optional Arguments:**
- `--project-dir PATH` - Override project root (auto-detects by walking up from `--setup-dir`)
- `--max-workers N` - Maximum parallel workers (default: 4)
- `--overwrite` - Overwrite existing outputs
- `--live-plots` - Enable plotting during run (default is off; plots are generated once after completion)
- `--monitor-perf` - Enable performance monitoring (default is enabled)
- `--no-monitor-perf` - Disable performance monitoring
- `--perf-sample-interval SEC` - Perf sampling interval (default: 5)
- `--perf-plot-interval SEC` - Perf plotting interval (default: 30)
- `--log-level LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

**Example:**
```bash
docker compose run --rm oa oa-da-project \
  --setup-dir /data \
  --project-dir /data/projects/project_2019-2020 \
  --max-workers 8 \
  --monitor-perf
```

**Benchmark outputs written by default:**
- `results/benchmark/manifest.json`
- `results/benchmark/cases/*.csv`
- `results/benchmark/scores/*.csv`
- `results/benchmark/tables/project_summary.csv`
- `results/benchmark/tables/update_summary.csv`
- `results/benchmark/summary.md`
- `results/plots/assim/scores/performance_scores.png`

**Benchmark config block (optional benchmark controls):**
```yaml
data_assimilation:
  benchmark:
    independent_variables:
      - station_swe
    score_station_sigma_threshold: 200
    plots: true
    output_dir: results/benchmark
```

Configured extra benchmark families can still appear as `semi_independent` in outputs, but only from the first same-variable or sister-station assimilation date onward.
`score_station_sigma_threshold` optionally excludes station rows with high resolved `station_uncertainty_pct` from non-sigma-aware benchmark metrics (`CRPSS`, `NER`) while leaving sigma-aware `zSkill` unchanged.
The headline plot shows only DA-date `prior` and `posterior` skill for assimilated and transfer-observed variables; whole-project propagated skill remains in `project_summary.csv`. Station-point rows also carry sigma-aware `zSkill`, and the headline plot adds a third `zSkill` panel whenever those station scores are available.

---

### oa-da-benchmark

**Re-run scientific benchmarking on an existing finished project**

Runs the same benchmark code path used by `oa-da-project` without re-running the model pipeline.

```bash
oa-da-benchmark \
  --project-dir /data/projects/project_2019-2020 \
  [OPTIONS]
```

**Required Arguments:**
- `--project-dir PATH` - Project directory to benchmark

**Optional Arguments:**
- `--setup-dir PATH` - Override setup root (otherwise inferred from `--project-dir`)
- `--variables NAME [NAME ...]` - Restrict benchmark variables to `scf`, `wet_snow`, `wet_snow_line`, `station_hs`, `station_swe`
- `--output-dir PATH` - Override benchmark results directory
- `--no-plots` - Skip benchmark plots
- `--max-workers N` - Override benchmark preprocessing worker count
- `--overwrite` - Recompute benchmark prerequisites if needed
- `--log-level LEVEL` - Logging level

**Example:**
```bash
docker compose run --rm oa oa-da-benchmark \
  --project-dir /data/projects/project_2019-2020 \
  --variables scf wet_snow_line station_swe
```

---

### oa-da-scf

**Prepare per-step SCF observation CSVs**

Copies SCF rows from `scf_summary.csv` into per-step `obs/obs_scf_<PRODUCT>_YYYYMMDD.csv` files under each step directory.

```bash
oa-da-scf \
  --project-dir PATH \
  [--summary-csv PATH] \
  [--product SNOWCOVER] \
  [--overwrite] \
  [--log-level LEVEL]
```

**Arguments:**
- `--project-dir PATH` - Project directory (e.g., `/data/projects/project_2019-2020`)
- `--summary-csv PATH` - Optional path to `scf_summary.csv`; when provided it is recorded in `obs.snowcover.summary_csv` so later maps and benchmarks use the same source. Without this option the command resolves `obs.snowcover.summary_csv`, then legacy defaults under `<setup>/obs/<project>/` and `<setup>/obs/summaries/<project>/`.
- `--product CODE` - Optional product tag override used in filenames (otherwise read from `project.yml` -> `obs.snowcover.product_tag`)
- `--overwrite` - Overwrite existing `obs_scf_*.csv` files

**Validation behavior (fail-fast):**
- Exactly one assimilation event per non-final step is required.
- The event date must lie within the associated step window.
- A matching summary row must exist for each configured event date.

**Example:**
```bash
oa-da-scf \
  --project-dir /data/projects/project_2019-2020 \
  --summary-csv /data/obs/project_2019-2020/scf_summary.csv \
  --overwrite
```

---

### oa-da-snowcover

**snow-cover Sentinel-2 FSC summarization**

Summarizes snow-cover FSC rasters (GeoTIFF or NetCDF) to a setup-level `scf_summary.csv` with `scf`, `n_valid`, `n_snow`, `n_invalid`, `cloud_fraction`, and `invalid_fraction`. `invalid_fraction` is the ROI-based unusable-scene fraction and captures missing or otherwise unusable ROI pixels even when they are not encoded as explicit cloud classes. If SCF uncertainty is enabled in project YAML, summary also includes `unc_mean`, `unc_min`, `unc_max`, and `unc_n_valid`.

```bash
oa-da-snowcover \
  --input-dir PATH \
  --project-label LABEL \
  [OPTIONS]
```

**Required Arguments:**
- `--input-dir PATH` - Directory with snow-cover FSC rasters (`*.tif/*.tiff/*.nc`)
- `--project-label LABEL` - Project label used under `obs/summaries/<project-label>/`

**Optional Arguments:**
- `--setup-dir PATH` - Setup directory (default: current directory)
- `--roi PATH` - ROI vector (auto-resolved under `<setup>/env`, generated from `grids/roi_<domain>_<resolution>.asc` if needed)
- `--roi-field FIELD` - ROI identifier field (optional)
- `--recursive` - Recurse into subdirectories
- `--start-date/--end-date` - Optional date bounds (defaults to project YAML date window if available)
- `--output-root PATH` - Override summary root (default: `<setup>/obs/summaries`)
- `--overwrite` - Overwrite existing `scf_summary.csv`
- `--log-level LEVEL` - Logging level

**Class handling:** 0..100 = valid FSC (percent), 205 = clouds (excluded; counted in `cloud_fraction`), 210 = water (excluded), 255/_FillValue = nodata.

**Uncertainty ingest (project YAML):**
- NetCDF must contain both value and uncertainty variables plus configured `time_variable`.
- Each SCF GeoTIFF must have `<stem>_uncertainty.tif` in the same directory.
- When uncertainty is enabled, preprocessing is strict fail-fast on missing/invalid uncertainty inputs.

**Example:**
```bash
oa-da-snowcover \
  --input-dir /data/obs/snowcover \
  --project-label project_2019-2020 \
  --setup-dir /data \
  --recursive
```

---

## Data Assimilation

### oa-da-model-scf

**H(x) forward operator**

Computes model-equivalent SCF from snow depth/SWE grids.

```bash
oa-da-model-scf \
  --project-dir PATH \
  --member-results PATH \
  --roi PATH \
  --date YYYY-MM-DD \
  [OPTIONS]
```

**Required Arguments:**
- `--project-dir PATH` - Project root (for config)
- `--member-results PATH` - Member results directory
- `--roi PATH` - ROI vector
- `--date DATE` - Date to process (YYYY-MM-DD)

**Optional Arguments:**
- `--variable {hs,swe}` - State variable (from config if omitted)
- `--method {depth_threshold,logistic}` - H(x) method (from config if omitted)

**Output:**
- Per-member: `results/point_scf_roi.csv`

**Example:**
```bash
oa-da-model-scf \
  --project-dir /data \
  --member-results /data/projects/project_2019-2020/steps/step_01_*/ensembles/prior/member_001/results \
  --roi /data/env/roi.gpkg \
  --date 2019-11-22
```

---

### oa-da-assimilate-scf

**SCF assimilation (weight calculation)**

Computes particle weights from observation-model mismatch.

```bash
oa-da-assimilate-scf \
  --project-dir PATH \
  --step-dir PATH \
  --ensemble {prior,posterior} \
  --date YYYY-MM-DD \
  --roi PATH \
  [OPTIONS]
```

**Required Arguments:**
- `--project-dir PATH` - Project root
- `--step-dir PATH` - Step directory
- `--ensemble {prior,posterior}` - Ensemble to assimilate
- `--date DATE` - Assimilation date
- `--roi PATH` - ROI vector

**Optional Arguments:**
- `--obs-csv PATH` - Observation CSV (auto-detected if omitted)
- `--output PATH` - Output weights CSV (default: assim/weights_scf_{date}.csv)

**Output:**
- `assim/weights_scf_YYYYMMDD.csv`

**Example:**
```bash
oa-da-assimilate-scf \
  --project-dir /data \
  --step-dir /data/projects/project_2019-2020/steps/step_01_* \
  --ensemble prior \
  --date 2019-11-22 \
  --roi /data/env/roi.gpkg
```

---

### oa-da-assimilate-wet-snow

**Wet snow assimilation**

Same as `oa-da-assimilate-scf` but for wet snow observations.

```bash
oa-da-assimilate-wet-snow \
  --project-dir PATH \
  --step-dir PATH \
  --ensemble prior \
  --date YYYY-MM-DD \
  --roi PATH
```

---

### oa-da-resample

**Particle resampling**

Performs systematic resampling based on particle weights.

```bash
oa-da-resample \
  --project-dir PATH \
  --step-dir PATH \
  --ensemble prior \
  --weights PATH \
  --target posterior \
  [OPTIONS]
```

**Required Arguments:**
- `--project-dir PATH` - Project root
- `--step-dir PATH` - Step directory
- `--ensemble {prior,posterior}` - Source ensemble
- `--weights PATH` - Weights CSV file
- `--target {posterior}` - Target ensemble name

**Optional Arguments:**
- `--ess-threshold-ratio RATIO` - ESS threshold as fraction of N (from config if omitted)
- `--ess-threshold N` - Absolute ESS threshold (overrides ratio)
- `--seed INT` - Random seed (from config if omitted)
- `--overwrite` - Overwrite existing posterior

**Behavior:**
- If `ESS ≥ threshold`: Skip resampling, mirror prior → posterior
- If `ESS < threshold`: Perform resampling

**Output:**
- `ensembles/posterior/member_001/...`
- `assim/indices_YYYYMMDD.csv`

**Example:**
```bash
oa-da-resample \
  --project-dir /data \
  --step-dir /data/projects/project_2019-2020/steps/step_01_* \
  --ensemble prior \
  --weights assim/weights_scf_20191122.csv \
  --target posterior
```

---

## Wet Snow Analysis

### oa-da-model-wet-snow

**Model wet snow classification**

Classifies model snow as wet/dry based on liquid water content.

```bash
oa-da-model-wet-snow \
  --setup-dir PATH \
  [OPTIONS]
```

**Required Arguments:**
- `--setup-dir PATH` - Setup directory (processes all steps)
- OR `--step-dir PATH` - Single step directory

**Optional Arguments:**
- `--members LIST` - Specific members (default: all)
- `--classification-method METHOD` - `liquid_water_fraction` or `liquid_water_amount`
- `--threshold PERCENT` - LWC fraction threshold for `liquid_water_fraction`
- `--liquid-water-amount-threshold-mm MM` - absolute liquid-water threshold for `liquid_water_amount`
- `--write-fraction` - Write LWC fraction rasters
- `--min-depth-mm MM` - Minimum snow depth (default: 5)

**Output:**
- Per member: `results/wet_snow/wet_snow_mask_*.tif`
- Optional: `results/wet_snow/lwc_fraction_*.tif`

---

### oa-da-wetsnow

**Sentinel-1 wet snow summary**

Processes Sentinel-1 WSM rasters into setup summary.

```bash
oa-da-wetsnow \
  --input-dir PATH \
  --project-label LABEL \
  [--setup-dir PATH] \
  [OPTIONS]
```

Similar to `oa-da-snowcover` but for Sentinel-1 wet snow masks (categorical classes configured via `obs.wetsnow.classes` in project YAML).

---

### oa-da-wetsnow-uncertainty

**Generate wet-snow uncertainty companion rasters**

Creates `*_uncertainty.tif` files next to source wet-snow GeoTIFFs using the
project YAML block `data_assimilation.uncertainty.wet_snow`.

```bash
oa-da-wetsnow-uncertainty \
  --setup-dir PATH \
  --project-label LABEL \
  [--overwrite]
```

Output is written next to the wet-snow source rasters (same filename stem plus
`_uncertainty.tif` suffix).

---

### oa-da-wetsnow-project

**Per-step wet-snow observation CSV generation**

Copies selected rows from `wet_snow_summary.csv` into per-step `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv` and `obs_wet_snow_line_<PRODUCT>_YYYYMMDD.csv` files based on project assimilation events. If `--summary-csv` is provided, the path is recorded in `obs.wetsnow.summary_csv` so later maps and WSF benchmarks use the same source; WSLA benchmarks read `wet_snow_line_diagnostics.csv` from the same directory unless `obs.wetsnow.wet_snow_line_diagnostics_csv` is configured explicitly.

---

## Visualization

### oa-da-plot-result-overview

**Setup result overview**

Plots the combined setup result overview: SCF, WSF, WSLA, ROI mean SWE, and ROI mean snow depth. The ROI SWE and snow-depth panels use the full ROI footprint, keep `open_loop` separate, and derive the 5-95% band from ensemble members only. If `<project-dir>/plots.yml` exists, the pipeline additionally writes `result_overview_custom.png` with the configured panel list. Custom panel configs support `WSF`, `WSLA`, `scores-crpss`, `scores-ner`, and station-only `scores-zskill` to embed the benchmark score panels individually.

```bash
oa-da-plot-result-overview \
  --project-dir PATH \
  [--setup-dir PATH] \
  [OPTIONS]
```

**Output:**
- `results/plots/results/result_overview.png`
- `results/plots/results/result_overview_custom.png` when `<project-dir>/plots.yml` is present

---

### oa-da-plot-project-maps

**Publication-style project maps**

Renders generated DA-event maps plus optional custom project maps from the compact project summary grid, setup grids, ROI, stations, and project observation summaries. By default the command generates one `da_*` map per assimilation event from the project YAML. `<project-dir>/maps.yml` is now reserved for custom maps such as `setup_overview`, and those custom maps are rendered together with the generated DA-event set in one command. The same workflow is also used for best-effort post-run pipeline rendering.
Map panels use the example-map visual grammar by default: boxed axes, coordinate ticks and grid lines, subplot labels like `(a)`, and attached vertical colorbars. Snow-depth model maps use the fixed reference palette with a shared linear colorbar scale per render run, `cm` tick labels, and transparent cells below `1 cm`. Increment maps use a signed diverging palette with negative changes in red and positive changes in blue.

Typical custom `maps.yml` files still use this panel catalog:

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
# - wet_snow_line            # source: open_loop | prior_probability | posterior_probability | posterior
# - wet_snow_elevation_fraction # source: open_loop | prior_probability | posterior_probability
# - legend
# - colorbar
# Optional panel keys:
# - title, name, date, legend, show_colorbar, show_scalebar, show_grid, show_hillshade, hillshade_extent
# - show_roi, show_station_marker, show_stations_name, show_stations_elev
```

Generated DA-event maps use four columns: `open loop`, `prior`, `posterior`, and `reference`. Snow-state reference panels show `analysis_increment` (`posterior - prior`); FSC and wet-snow reference panels show the satellite observation. WSLA lines are panel-local and observation WSLA is drawn only in the observation/reference panel. If an event's resampling manifest has `skipped: true`, the generated map title includes `resampling skipped`.

```bash
oa-da-plot-project-maps \
  --project-dir PATH \
  [--config PATH] \
  [--name RECIPE_NAME] \
  [--max-workers N]
```

**Output:**
- generated DA-event maps under `results/maps/da_events/*.png`
- custom YAML maps under `results/maps/*.png`

Generated `wet_snow` and `wet_snow_line` DA-event maps include a generated-only spatial elevation-band WSF row below the primary wet-snow row. Each panel keeps the map footprint, but every valid cell is colored by the raw wet snow fraction of its elevation band on a fixed white-to-black `0-100%` scale for open loop, posterior, and observation columns.

Static context panels (`hillshade`, `dem`, `svf`, `srf`, `landcover`) render the full raster coverage inside the map extent. Model and observation panels remain ROI-masked. When `show_hillshade: true`, `hillshade_extent: roi` limits the hillshade to the ROI mask and `hillshade_extent: full` draws it across the full panel. In the supported Docker workflow, omitted `--max-workers` uses automatic recipe-level multicore rendering with the effective worker count clamped to `min(visible CPUs, selected recipes)`; pass `--max-workers 1` to keep rendering sequential. If one or more maps fail because supporting data are missing, the pipeline logs a rerun command and continues. After changing shipped or local static grids, rerender the full local project-map catalog so mixed gallery outputs do not keep stale static panels.

Overview panels use setup-local GISCO GeoJSONs under `<setup>/env/` for country boundaries, regions, and labels. If those files are missing, the overview renderer downloads them once into that directory automatically.

### oa-da-plot-project-plots

**Recreate all project plots from existing outputs**

Runs the same post-run plot orchestration used by the project pipeline, but without rerunning the DA workflow itself. The command expects an already finished project with populated `steps/step_*/.../results` outputs. Before plotting, it rebuilds the ROI fraction envelopes in `results/misc/`, then renders forcing plots, setup point-result plots, assimilation weights, ESS timeline, and the result overview panels.

```bash
oa-da-plot-project-plots \
  --project-dir PATH \
  [--plot-workers N] \
  [--max-workers N]
```

**Output:**
- regenerated plot products under `results/plots/**`
- refreshed fraction envelopes under `results/misc/point_*_roi_envelope.csv`

Use this when you changed plotting code, `plots.yml`, or map-independent styling and want a clean plot rerender without executing `oa-da-project` again.

### oa-da-project-pdf

**Assemble a DIN A4 project plots/maps PDF**

Collects a compact report summary page, the curated project overview outputs, and DA-event maps into one DIN A4 portrait PDF. The command does not rerun plots or maps. It fails fast with a complete missing-file list when the required result overview, setup map, setup weights overview, or generated DA maps are missing.

```bash
oa-da-project-pdf \
  --project-dir PATH \
  [--output PATH]
```

**Output:**
- `results/reports/project_plots_maps_collection.pdf` by default

The PDF starts with a generated one-page project report containing key YAML settings, DA-event counts, and computing-cost stats from project logs and `results/plots/perf/project_perf_metrics.csv` when available. It then includes `result_overview.png`, optional `result_overview_custom.png`, `setup_overview.png`, all `setup_weights_overview*.png` pages, and one DIN A4 page per generated DA-event map under `results/maps/da_events/da_<n>.png`. Standalone per-event weights plots and other remaining plot/map PNGs are not included.

### oa-da-fetch-overview-geojson

**Prefetch setup-local overview GeoJSONs**

Downloads the GISCO overview boundaries, regions, and labels into `<setup>/env/` so overview-map rendering does not need a first-use network fetch.

```bash
oa-da-fetch-overview-geojson --setup-dir PATH
# OR
oa-da-fetch-overview-geojson --project-dir PATH
```

---

### oa-da-plot-weights

**Particle weight plots**

Visualizes particle weights and residuals for an assimilation date.

```bash
oa-da-plot-weights WEIGHTS_CSV [OPTIONS]
```

**Example:**
```bash
oa-da-plot-weights \
  /data/projects/project_2019-2020/steps/step_01_*/assim/weights_scf_20191122.csv
```

**Output:**
- `plots/assim/weights/step_XX_weights.png`

---

### oa-da-plot-ess

**ESS timeline plots**

Plots effective sample size evolution across the setup.

```bash
oa-da-plot-ess --step-dir PATH  # Per-step
# OR
oa-da-plot-ess --setup-dir PATH  # Setup-wide
```

**Output:**
- Per-step: `plots/assim/ess/step_XX_ess.png`
- Setup: `results/plots/assim/ess/setup_ess_timeline_{setup}.png`

---

## Utilities

### oa-da-perf-monitor

**Performance monitoring**

Standalone performance monitor (CPU/RAM% + setup disk size) that can attach to a running setup.

```bash
oa-da-perf-monitor \
  --setup-dir PATH \
  [--sample-interval SEC] \
  [--plot-interval SEC]
```

Suggested intervals: sample every 5–10 seconds; refresh the plot every 30–60 seconds.

**Output:**
- `results/plots/perf/project_perf_metrics.csv`
- `results/plots/perf/project_perf.png` (CPU/RAM% left axis, disk GB right axis)

---

### oa-da-model-scf-project-daily

**Backfill model SCF**

Computes daily ROI-mean model SCF for all members (retroactive).

```bash
oa-da-model-scf-project-daily \
  --project-dir PATH \
  --setup-dir PATH \
  --roi PATH \
  [--max-workers N]
```

Use this to add model SCF time series to an already-completed setup.

---

### oa-da-model-wet-snow-project-daily

**Backfill wet snow**

Similar to `oa-da-model-scf-project-daily` but for wet snow classification.

---

## Sub-domain Mode

### oa-da-subdomain

Split a setup into non-overlapping sub-domains, run one independent data assimilation project per sub-domain, then merge compact outputs.

Common workflows:

```bash
# Prepare per-sub-domain setups
oa-da-subdomain prepare \
  --setup-dir /data/rofental \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --roi /data/regions.gpkg \
  --id-field id

# Run all sub-domains (parallel)
oa-da-subdomain run \
  --project-dir /data/rofental/projects/project_2022_2023

# Merge grids and points
oa-da-subdomain merge \
  --project-dir /data/rofental/projects/project_2022_2023

# Plot station comparisons
oa-da-subdomain plot \
  --project-dir /data/rofental/projects/project_2022_2023

# One-shot pipeline (prepare -> run -> merge -> plot)
oa-da-subdomain pipeline \
  --setup-dir /data/rofental \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --roi /data/regions.gpkg
```

Defaults & tips:
- If `--subdomain-root` is omitted, `<project>/subdomains` is used.
- If `--manifest` is omitted in run/merge/plot, it resolves to `<subdomain_root>/subdomain_manifest.json`.
- If `--roi` is omitted in prepare/pipeline, `<setup>/env/subdomains.gpkg` is preferred and `<setup>/env/roi.gpkg` is the fallback.
- `--id-field` must exist in the regions file; there is no automatic fallback to another field.
- Sub-domain mode requires at least two polygons in the ROI file.
- Sub-domain runs fail fast if configured assimilation events are not available in local sub-domain summaries.
- openAMUNDSEN-DA requires `grids/roi_<domain>_<resolution>.asc`; it is generated silently from ROI/regions vector input when missing.
- Use `--max-workers` to control parallelism; BLAS/OMP threads are pinned to 1 inside the image.
- Merge is hard mosaic only (no interpolation/blending).
- Visible breaks at sub-domain boundaries are expected and intentional.
- Merge writes `results/grids/da_output_grids.nc` as the compact data assimilation grid product.
- Compact retention is the default (`data_assimilation.output.retention: compact`); set `full` to keep all member grid artifacts.
- Sub-domain mode keeps point outputs and point plots inside each sub-domain project (no project-root point merge).

Inputs/outputs:
- `--setup-dir` points to the setup root; `--project-dir` points to one project under `setup/projects`.
- Prepared sub-domain runs live under `<subdomain_root>/<subdomain_id>/`.
- Project-level outputs are written under `<project>/results/`.
- Sub-domain point outputs and plots stay under each sub-domain project directory.
- Repository example: `examples/subdomains` with regions in `env/subdomains.gpkg`.

---

## Common Options

Most commands support:

| Option | Description |
|:-------|:------------|
| `--log-level {DEBUG,INFO,WARNING,ERROR}` | Logging verbosity |
| `--overwrite` | Overwrite existing outputs |
| `--help` | Show command help |

---

## Docker Usage

All commands can be run via Docker Compose:

```bash
docker compose run --rm oa <COMMAND> [OPTIONS]
```

**Examples:**

```bash
# Setup pipeline
docker compose run --rm oa oa-da-project \
  --setup-dir /data \
  --project-dir /data/projects/project_2019-2020

# Snow-cover summary (GeoTIFF/NetCDF; MODIS after HDF→GeoTIFF with classes set in project.yml)
docker compose run --rm oa oa-da-snowcover \
  --input-dir /data/obs/snowcover \
  --project-label project_2019-2020 \
  --setup-dir /data
```

---

## Next Steps

- [Configuration Guide]({{ site.baseurl }}{% link guides/configuration.md %}) - Configure commands via YAML
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments/index.md %}) - End-to-end workflow
- [Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}) - Common issues

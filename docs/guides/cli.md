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

The package provides **20 CLI entry points** for workflow automation, organized into 5 categories:

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

**Main project pipeline orchestrator**

Runs the complete project data assimilation cycle: prior forcing → ensemble run → assimilation → resampling → rejuvenation.
The pipeline now also runs the scientific benchmark stage automatically at the end of every project, then attempts to assemble the project PDF report.

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

**Report output attempted by default:**
- `results/reports/project_report.pdf`

Report generation is best-effort in `oa-da-project`: missing plots/maps or other PDF assembly errors are logged with a manual rerun command and do not fail the completed model run.

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

### oa-da-merge-project-grids

**Merge completed project DA summary NetCDFs**

Concatenates multiple completed project `results/grids/da_output_grids.nc` files along their time-like dimensions and writes one normal DA summary NetCDF. This is intended for adjacent annual projects that share the same domain, grid, CRS, variables and compact NetCDF encoding.

Use setup-relative project names:

```bash
oa-da-merge-project-grids \
  --setup /data/rofental \
  --project project_2020_2021 \
  --project project_2021_2022 \
  --output-nc /data/rofental/results/grids/da_output_grids_2020_2022.nc
```

Or pass project directories directly:

```bash
oa-da-merge-project-grids \
  --project-dir /data/rofental/projects/project_2020_2021 \
  --project-dir /data/rofental/projects/project_2021_2022 \
  --output-nc /data/rofental/results/grids/da_output_grids_2020_2022.nc
```

**Required Arguments:**
- `--output-nc PATH` - Merged output NetCDF path
- Either `--setup PATH` with repeated `--project NAME`, or repeated `--project-dir PATH`

**Optional Arguments:**
- `--overwrite` - Replace an existing output file
- `--log-level LEVEL` - Logging level

**Validation behavior (fail-fast):**
- input projects must already contain `results/grids/da_output_grids.nc`
- all inputs must have identical variables, static coordinates, x/y grid, CRS and compatible NetCDF encoding
- duplicate timestamps on any time-like dimension are rejected
- time-like dimensions inherited from openAMUNDSEN outputs, such as `time1` and `time2`, are concatenated independently

The output keeps normal DA summary variable names and stores merge provenance in global NetCDF attributes.

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
- `--summary-csv PATH` - Optional path to `scf_summary.csv`; when provided it is recorded in `obs.snowcover.summary_csv` so later maps and benchmarks use the same source. Without this option the command resolves `obs.snowcover.summary_csv`, then supported v1 compatibility defaults under `<setup>/obs/<project>/` and `<setup>/obs/summaries/<project>/`.
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
  --step-dir PATH \
  --ensemble prior \
  --weights PATH \
  --target posterior \
  [OPTIONS]
```

**Required Arguments:**
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

Plots the combined setup result overview: fSCA, WSF, WSLA, ROI mean SWE, and ROI mean snow depth. The ROI SWE and snow-depth panels use the full ROI footprint, keep `open_loop` separate, and derive the full finite-member range from ensemble members only. Figure legends are built from rendered plot elements, so unused observation, ensemble, open-loop, or DA-event entries are omitted. Observations are red and use redundant encodings: station observations are dashed lines, satellite observations are circles and observation X markers identify assimilated observations. Month labels are centered between month-boundary ticks. If `<project-dir>/plots.yml` exists, the configured panel list is used for the standard `result_overview.png`. Configured panels support `WSF`, `WSLA`, `scores-crpss`, `scores-ner`, and station-only `scores-zskill` to embed the benchmark score panels individually.

```bash
oa-da-plot-result-overview \
  --project-dir PATH \
  [--setup-dir PATH] \
  [OPTIONS]
```

**Output:**
- `results/plots/results/result_overview.png`

---

### oa-da-plot-project-maps

**Publication-style project maps**

Renders generated DA-event maps plus optional custom project maps from the compact project summary grid, setup grids, ROI, stations, and project observation summaries. By default the command generates one `da_*` map per assimilation event from the project YAML. `<project-dir>/maps.yml` is now reserved for custom maps such as `setup_overview`, and those custom maps are rendered together with the generated DA-event set in one command. The same workflow is also used for best-effort post-run pipeline rendering.
Map panels use the example-map visual grammar by default: boxed axes, coordinate ticks and grid lines, subplot labels like `(a)`, and attached vertical colorbars. Continuous sequential model and observation maps use the viridis palette; snow-depth maps keep a shared linear colorbar scale per render run, `cm` tick labels, and transparent cells below `1 cm`. Increment maps and SRF maps use a compact-neutral signed red-blue diverging palette, with negative increments in red and positive increments in blue.

Typical custom `maps.yml` files still use this panel catalog:

```yaml
# Available panel kinds:
# - overview                 # scale: 1000000 ; optional roi_label
# - roi
# - hillshade
# - dem
# - aspect
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
# - title, name, date, legend, legend_items, below_items
# - show_colorbar, show_scalebar, show_grid, show_hillshade, hillshade_extent
# - observation (uncertainty only), show_roi, show_station_marker, show_stations_name, show_stations_elev
# - landcover_grouping: broad | rofental_manuscript # landcover panels only; omitted/native keeps source classes
# Optional recipe-level row zoom views:
# row_views:
#   - row: 1
#     center: [643767, 5191680] # setup/project CRS by default
#     zoom: 13                 # Google/Slippy-map zoom
#     # center_crs: EPSG:4326
#     # viewport_px: [1024, 1024]
```

Generated DA-event maps use four columns: `open loop`, `prior`, `posterior`, and `reference`. Snow-state reference panels show `analysis_increment` (`posterior - prior`); FSC and wet-snow reference panels show the satellite observation. Generated WSLA maps show spatial WSF without WSLA contours in the first row and derived WSLA contours only in the elevation-band WSF row. Top-level sub-domain SCF events use a taller same-file layout with a 2x2 snow-cover block above the 2x2 snow-depth response block; exact rerendering requires retained per-sub-domain grids. If an event's resampling manifest has `skipped: true`, the generated map title includes `resampling skipped`.

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

Generated `wet_snow` and `wet_snow_line` DA-event maps include a generated-only spatial elevation-band WSF row below the primary wet-snow row. Each panel keeps the map footprint, but every valid cell is colored by the raw wet snow fraction of its elevation band on a fixed white-to-black `0-100%` scale for open loop, posterior, and observation columns. Generated WSLA maps keep the primary WSF row, the elevation-band WSF row with derived WSLA contours and the snow-depth response row.

Static context panels (`hillshade`, `dem`, `aspect`, `svf`, `srf`, `landcover`) mask raster cells outside the ROI. `aspect` is derived from the DEM at render time and shown on a continuous radian colorbar. Continuous static panels such as `srf` can set `show_hillshade: true` to draw a terrain underlay. `landcover` panels can set `landcover_grouping: broad` to merge detailed vegetation/forest classes into broad manuscript-friendly classes, and the legend only lists classes present inside the ROI. The `rofental_manuscript` grouping is reserved for the Rofental paper/tutorial setup map, not for generic land-cover maps; it merges codes 1 and 13 as `rock`, 2 and 3 as `ice`, 4--7 as `grass/shrub`, and 8--12 as `forest`. Non-legend panels can use `legend_items` with `placement: below` or `placement: inside` (`anchor: top_left|top_right|bottom_left|bottom_right`) for compact layer legends such as station symbols; legacy `below_items` remains supported. Model and observation panels remain ROI-masked. Prepared sub-domain projects automatically draw the configured sub-domain polygons from `subdomain_manifest.json` on top-level ROI-bearing map panels and overview panels. Generated DA-event maps mark a sub-domain as `no DA` only when the event was dropped locally and recorded in `results/subdomain_dropped_events.csv`; events with valid weights are not marked just because posterior resampling was skipped. When `show_hillshade: true`, `hillshade_extent: roi` limits the hillshade to the ROI mask and `hillshade_extent: full` draws it across the full panel. In the supported Docker workflow, omitted `--max-workers` uses automatic recipe-level multicore rendering with the effective worker count clamped to `min(visible CPUs, selected recipes)`; pass `--max-workers 1` to keep rendering sequential. If one or more maps fail because supporting data are missing, the pipeline logs a rerun command and continues. After changing shipped or local static grids, rerender the full local project-map catalog so mixed gallery outputs do not keep stale static panels.

The `uncertainty` panel renders GeoTIFF companion rasters named `<source>_uncertainty.tif` for `observation: scf` or `observation: wet_snow`. Values use the same `0..100 [%]` scale as uncertainty-aware preprocessing, and invalid observation pixels stay masked.

Recipe-level `row_views` assign a shared zoom extent to all panels in a row. Centers are interpreted in the setup/project CRS unless `center_crs` is provided, and `zoom` follows Google/Slippy-map semantics with a default `1024 x 1024 px` viewport.

Overview panels use setup-local GISCO GeoJSONs under `<setup>/env/` for country boundaries, regions, and labels. If those files are missing, the overview renderer downloads them once into that directory automatically.

### oa-da-plot-project-plots

**Recreate all project plots from existing outputs**

Runs the same post-run plot orchestration used by the project pipeline, but without rerunning the DA workflow itself. The command expects an already finished project with populated `steps/step_*/.../results` outputs. Before plotting, it rebuilds the ROI fraction envelopes in `results/misc/`, then renders forcing plots, setup point-result plots, assimilation weights, ESS timeline, and the result overview panels. `wet_snow_line` weights plots use meter residuals and compact `WSLA` labels, while `scf` weights plots use compact `fSCA` labels; weights residual axes use compact residual-unit labels and adaptive sigma labels. Setup weights overviews include `(a)`, `(b)`, ... panel labels in the DA-event headers and panel-local station legends; standalone per-event weights plots keep their compact bottom legend.

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

Collects a compact report summary page, the curated project overview outputs, diagnostics, and DA-event maps into one DIN A4 portrait PDF. `oa-da-project` attempts this automatically at the end of the full project pipeline; this command is for manual reruns. The command does not rerun plots or maps. It fails fast with a complete missing-file list when the required result overview, setup map, setup weights overview, or generated DA maps are missing.

```bash
oa-da-project-pdf \
  --project-dir PATH \
  [--output PATH]
```

**Output:**
- `results/reports/project_report.pdf` by default

The PDF starts with a generated one-page project report containing basic setup YAML settings, wet-snow classification and liquid-water-content settings, DA-event counts, computing-cost stats from project logs and `results/plots/perf/project_perf_metrics.csv` when available, plus a bottom `Content` table with page numbers first and section names second. It then includes `result_overview.png`, `setup_overview.png`, all `setup_weights_overview*.png` pages, station snow-depth point plots on one page, `performance_scores.png`, `project_perf.png`, and generated DA-event maps under `results/maps/da_events/da_<n>.png` in temporal order. Source PNGs are placed at their shared export-DPI size rather than scaled down to fit a page; consecutive DA maps are packed onto a page only while the reserved bottom gap is preserved. Standalone per-event weights plots and other remaining plot/map PNGs are not included.

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

Standalone performance monitor (CPU/RAM%, filesystem disk pressure, throttled project directory size, and optional CPU temperature) that can attach to a running project.

```bash
oa-da-perf-monitor \
  --project-dir PATH \
  [--sample-interval SEC] \
  [--plot-interval SEC] \
  [--disk-scan-interval SEC]
```

Suggested intervals: sample every 5–10 seconds; refresh the plot every 30–60 seconds; scan recursive project size every 300 seconds or longer for large runs.

**Output:**
- `results/plots/perf/project_perf_metrics.csv`
- `results/plots/perf/project_perf.png` (CPU/RAM/filesystem-used % left axis, project size GB right axis, and CPU temperature when available)

CPU temperature sampling is optional and fail-soft. If sensors are unavailable in a container or virtualized environment, `cpu_temp_c` stays blank and the temperature line is omitted. For Docker runs that expose host sensors at a custom path, set `OA_DA_THERMAL_SYSFS_ROOT` to the mounted `hwmon` directory.

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

Split a setup into non-overlapping sub-domains. The CLI supports the existing openAMUNDSEN-DA workflow and a plain openAMUNDSEN model workflow.
The data assimilation workflow creates independent regional DA projects and merges their compact grids; it is not a formal particle-filter localization scheme.

Data assimilation workflow:

```bash
# Prepare per-sub-domain DA setups
oa-da-subdomain prepare \
  --setup-dir /data/rofental \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --roi /data/regions.gpkg \
  --id-field id

# Run all sub-domains (parallel)
oa-da-subdomain run \
  --project-dir /data/rofental/projects/project_2022_2023

# Merge grids
oa-da-subdomain merge \
  --project-dir /data/rofental/projects/project_2022_2023

# Optional cleanup is explicit and guarded; merge alone never removes raw grid support files.
oa-da-subdomain merge \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --cleanup-compact-artifacts \
  --confirm-delete-raw-grid-support

# Plot station comparisons
oa-da-subdomain plot \
  --project-dir /data/rofental/projects/project_2022_2023

# One-shot pipeline (prepare -> run -> merge -> plot)
oa-da-subdomain pipeline \
  --setup-dir /data/rofental \
  --project-dir /data/rofental/projects/project_2022_2023 \
  --roi /data/regions.gpkg
```

Plain openAMUNDSEN model workflow:

```bash
# Prepare plain model sub-domain setups
oa-da-subdomain model-prepare \
  --setup-dir /data/subdomains \
  --regions /data/subdomains/env/subdomains.gpkg

# Run all model sub-domains in parallel
oa-da-subdomain model-run \
  --setup-dir /data/subdomains \
  --max-workers 24

# Merge model grid outputs
oa-da-subdomain model-merge \
  --setup-dir /data/subdomains

# One-shot model pipeline
oa-da-subdomain model-pipeline \
  --setup-dir /data/subdomains \
  --regions /data/subdomains/env/subdomains.gpkg \
  --max-workers 24 \
  --overwrite
```

Docker usage with a mounted setup:

```bash
# Run from the openAMUNDSEN-DA repository root.
# PROJ is the host path to the plain openAMUNDSEN setup; it is mounted as /data.
PROJ=/absolute/path/to/setup docker compose run --rm oa \
  oa-da-subdomain model-pipeline \
  --setup-dir /data \
  --regions /data/env/subdomains.gpkg \
  --max-workers 24 \
  --overwrite
```

Use `/data/...` paths in command arguments because the host setup is mounted at `/data` inside the container. The image provides both `oa-da-subdomain` and the `openamundsen` executable that `model-run` launches for each sub-domain.

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

DA defaults & tips:
- If `--subdomain-root` is omitted, `<project>/subdomains` is used.
- If `--manifest` is omitted in run/merge/plot, it resolves to `<subdomain_root>/subdomain_manifest.json`.
- If `--roi` is omitted in prepare/pipeline, `<setup>/env/subdomains.gpkg` is preferred and `<setup>/env/roi.gpkg` is the fallback.
- `--id-field` must exist in the regions file; there is no automatic fallback to another field.
- Sub-domain mode requires at least two polygons in the ROI file.
- Sub-domain runs fail fast if configured assimilation events are not available in local sub-domain summaries.
- openAMUNDSEN-DA requires `grids/roi_<domain>_<resolution>.asc`; it is generated from ROI/regions vector input when missing and then used as the canonical mask.
- Use `--max-workers` to control parallelism; BLAS/OMP threads are pinned to 1 inside the image.
- Merge is hard mosaic only (no interpolation/blending).
- Visible breaks at sub-domain boundaries are expected and intentional.
- Merge writes `results/grids/da_output_grids.nc` as the compact data assimilation grid product.
- Sub-domain DA projects default to full retention (`data_assimilation.output.retention: full`) so DA-event maps can be regenerated exactly. Set `compact` only if you knowingly allow heavy sub-domain grids to be pruned after the merged compact NetCDF is written.
- Sub-domain mode keeps point outputs and point plots inside each sub-domain project (no project-root point merge).

Model defaults & tips:
- If `--subdomain-root` is omitted, `<setup>/subdomains/model` is used.
- If `--manifest` is omitted in `model-run`/`model-merge`, it resolves to `<setup>/subdomains/model/subdomain_manifest.json`.
- If `--regions`/`--roi` is omitted in `model-prepare`/`model-pipeline`, `<setup>/env/subdomains.gpkg` is preferred and `<setup>/env/roi.gpkg` is the fallback.
- The source setup YAML must define `start_date` and `end_date`.
- Generated model sub-domain setup YAMLs remain plain openAMUNDSEN configs.
- `model-run` launches `openamundsen <subdomain_setup.yml>` once per selected sub-domain and writes `<subdomain>/run.log` plus `<subdomain>/run_manifest.json`.
- `model-merge` reads matching `.nc`, `.tif`, and `.tiff` outputs from each `<subdomain>/results/grids/`.
- Model merge is hard mosaic only; point/timeseries outputs are not merged in v1.
- For large domains, keep `--max-workers` at or below the CPU cores available to Docker/the host.

DA inputs/outputs:
- `--setup-dir` points to the setup root; `--project-dir` points to one project under `setup/projects`.
- Prepared sub-domain runs live under `<subdomain_root>/<subdomain_id>/`.
- Project-level outputs are written under `<project>/results/`.
- Sub-domain point outputs and plots stay under each sub-domain project directory.
- Repository example: `examples/subdomains` with regions in `env/subdomains.gpkg`.

Model inputs/outputs:
- `--setup-dir` points to a plain openAMUNDSEN setup root.
- Prepared model runs live under `<setup>/subdomains/model/<subdomain_id>/`.
- Per-subdomain model outputs are written under `<setup>/subdomains/model/<id>/results/`.
- Merged model grid outputs are written under `<setup>/subdomains/model/results/grids/`.

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

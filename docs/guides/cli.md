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

The package provides **17 CLI entry points** for workflow automation, organized into 5 categories:

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
- `--summary-csv PATH` - Optional path to `scf_summary.csv` (default: `<setup>/obs/<project>/scf_summary.csv`)
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

Summarizes snow-cover FSC rasters (GeoTIFF or NetCDF) to a setup-level `scf_summary.csv` with `scf`, `n_valid`, `n_snow`, and `cloud_fraction`. If SCF uncertainty is enabled in project YAML, summary also includes `unc_mean`, `unc_min`, `unc_max`, and `unc_n_valid`.

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
- `--threshold PERCENT` - LWC threshold (default: from config)
- `--write-fraction` - Write LWC fraction rasters
- `--min-depth-mm MM` - Minimum snow depth (default: 10)

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

Copies selected rows from `wet_snow_summary.csv` into per-step `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv` files based on project assimilation events.

---

## Visualization

### oa-da-plot-result-overview

**Setup result overview**

Plots the combined setup result overview: SCF, wet-snow, ROI mean SWE, and ROI mean snow depth. The ROI SWE and snow-depth panels use the full ROI footprint, keep `open_loop` separate, and derive the 5-95% band from ensemble members only. If `<project-dir>/result_overview_custom.yml` exists, the pipeline additionally writes `result_overview_custom.png` with the configured panel list.

```bash
oa-da-plot-result-overview \
  --project-dir PATH \
  [--setup-dir PATH] \
  [OPTIONS]
```

**Output:**
- `plots/results/result_overview.png`
- `plots/results/result_overview_custom.png` when `<project-dir>/result_overview_custom.yml` is present

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
- Setup: `plots/assim/ess/setup_ess_timeline_{setup}.png`

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
- `plots/perf/setup_perf_metrics.csv`
- `plots/perf/setup_perf.png` (CPU/RAM% left axis, disk GB right axis)

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

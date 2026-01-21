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

The package provides **16 CLI entry points** for workflow automation, organized into 5 categories:

1. **Core Workflow** - Main pipeline commands
2. **Data Assimilation** - DA-specific operations
3. **Wet Snow Analysis** - Wet snow processing
4. **Visualization** - Plotting commands
5. **Utilities** - Helper tools

All commands are available as:
- Python modules: `python -m openamundsen_da.MODULE.COMMAND`
- CLI scripts: `oa-da-COMMAND` (after installation)

---

## Core Workflow

### oa-da-season

**Main season pipeline orchestrator**

Runs the complete seasonal DA cycle: prior forcing → ensemble run → assimilation → resampling → rejuvenation.

```bash
oa-da-season \
  --season-dir /data/propagation/season_2019-2020 \
  [OPTIONS]
```

**Required Arguments:**
- `--season-dir PATH` - Season directory (project auto-detected if `project.yml` is in a parent)

**Optional Arguments:**
- `--project-dir PATH` - Override project root (auto-detects by walking up from `--season-dir`)
- `--max-workers N` - Maximum parallel workers (default: from .env or 4)
- `--overwrite` - Overwrite existing outputs
- `--live-plots` - Enable plotting during run (default is off; plots are generated once after completion)
- `--monitor-perf` - Enable performance monitoring
- `--perf-sample-interval SEC` - Perf sampling interval (default: 5)
- `--perf-plot-interval SEC` - Perf plotting interval (default: 30)
- `--log-level LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)

**Example:**
```bash
docker compose run --rm oa oa-da-season \
  --season-dir /data/propagation/season_2019-2020 \
  --max-workers 8 \
  --monitor-perf
```

---

### oa-da-scf

**Prepare per-step SCF observation CSVs**

Copies SCF rows from `scf_summary.csv` into per-step `obs/obs_scf_<PRODUCT>_YYYYMMDD.csv` files under each step directory.

```bash
oa-da-scf \
  --season-dir PATH \
  [--summary-csv PATH] \
  [--product SNOWCOVER] \
  [--overwrite] \
  [--log-level LEVEL]
```

**Arguments:**
- `--season-dir PATH` - Season directory (e.g., `propagation/season_2019-2020`)
- `--summary-csv PATH` - Optional path to `scf_summary.csv` (default: `<project>/obs/summaries/<season>/scf_summary.csv`)
- `--product CODE` - Product tag used in filenames (default: `SNOWCOVER`)
- `--overwrite` - Overwrite existing `obs_scf_*.csv` files

**Example:**
```bash
oa-da-scf \
  --season-dir /data/propagation/season_2019-2020 \
  --summary-csv /data/obs/summaries/season_2019-2020/scf_summary.csv \
  --overwrite
```

---

### oa-da-snowcover

Summarizes snow-cover rasters (GeoTIFF/NetCDF) into `scf_summary.csv` using class mappings from `project.yml` (`obs.snowcover.classes`) and the project land-cover mask.

```bash
oa-da-snowcover \
  --input-dir PATH \
  --season-label LABEL \
  [OPTIONS]
```

**Required Arguments:**
- `--input-dir PATH` - Directory with snow-cover rasters (`*.tif/*.tiff/*.nc`)
- `--season-label LABEL` - Season identifier (e.g., season_2019-2020)

**Optional Arguments:**
- `--project-dir PATH` - Project root (default: current directory)
- `--output-root PATH` - Override output root (default: `<project>/obs`)
- `--roi PATH` - ROI vector used for clipping (default: `<project>/env/roi.gpkg`)
- `--roi-field FIELD` - ROI identifier field (optional)
- `--recursive` - Recurse into subdirectories
- `--start-date/--end-date` - Optional date bounds
- `--overwrite` - Overwrite existing outputs
- `--log-level LEVEL` - Default: INFO

**Output:**
- Summary CSV: `obs/{season}/scf_summary.csv`

---

### oa-da-snowcover

**snow-cover Sentinel-2 FSC summarization**

Summarizes snow-cover FSC rasters (GeoTIFF or NetCDF) to a season-level `scf_summary.csv` with `scf`, `n_valid`, `n_snow`, and `cloud_fraction`.

```bash
oa-da-snowcover \
  --input-dir PATH \
  --season-label LABEL \
  [OPTIONS]
```

**Required Arguments:**
- `--input-dir PATH` - Directory with snow-cover FSC rasters (`*.tif/*.tiff/*.nc`)
- `--season-label LABEL` - Season identifier (e.g., season_2019-2020)

**Optional Arguments:**
- `--project-dir PATH` - Project root (default: current directory)
- `--aoi PATH` - ROI vector (auto-detected from `<project>/env/roi.gpkg` if omitted)
- `--roi-field FIELD` - ROI identifier field (optional)
- `--recursive` - Recurse into subdirectories
- `--output-root PATH` - Override obs root (default: `<project>/obs`)
- `--log-level LEVEL` - Logging level

**Class handling:** 0..100 = valid FSC (percent), 205 = clouds (excluded; counted in `cloud_fraction`), 210 = water (excluded), 255/_FillValue = nodata.

**Example:**
```bash
oa-da-snowcover \
  --input-dir /data/obs/snowcover \
  --season-label season_2019-2020 \
  --project-dir /data \
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
  --member-results /data/propagation/season_2019-2020/step_01_*/ensembles/prior/member_001/results \
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
  --step-dir /data/propagation/season_2019-2020/step_01_* \
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
  --step-dir /data/propagation/season_2019-2020/step_01_* \
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
  --season-dir PATH \
  [OPTIONS]
```

**Required Arguments:**
- `--season-dir PATH` - Season directory (processes all steps)
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

Processes Sentinel-1 WSM rasters into season summary.

```bash
oa-da-wetsnow \
  --input-dir PATH \
  --season-label LABEL \
  --project-dir PATH \
  [OPTIONS]
```

Similar to `oa-da-snowcover` but for Sentinel-1 wet snow masks.

---

### oa-da-wetsnow-season

**Season-wide S1 processing**

Batch processes Sentinel-1 observations for entire season.

---

## Visualization

### oa-da-plot-scf

**SCF time series plots**

Plots observed and modeled SCF time series.

```bash
oa-da-plot-scf \
  --season-dir PATH \
  --project-dir PATH \
  [OPTIONS]
```

**Output:**
- `plots/results/fraction_timeseries.png`

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
  /data/propagation/season_2019-2020/step_01_*/assim/weights_scf_20191122.csv
```

**Output:**
- `plots/assim/weights/step_XX_weights.png`

---

### oa-da-plot-ess

**ESS timeline plots**

Plots effective sample size evolution across the season.

```bash
oa-da-plot-ess --step-dir PATH  # Per-step
# OR
oa-da-plot-ess --season-dir PATH  # Season-wide
```

**Output:**
- Per-step: `plots/assim/ess/step_XX_ess.png`
- Season: `plots/assim/ess/season_ess_timeline_{season}.png`

---

## Utilities

### oa-da-perf-monitor

**Performance monitoring**

Standalone performance monitor (CPU/RAM% + season disk size) that can attach to a running season.

```bash
oa-da-perf-monitor \
  --season-dir PATH \
  [--sample-interval SEC] \
  [--plot-interval SEC]
```

Suggested intervals: sample every 5–10 seconds; refresh the plot every 30–60 seconds.

**Output:**
- `plots/perf/season_perf_metrics.csv`
- `plots/perf/season_perf.png` (CPU/RAM% left axis, disk GB right axis)

---

### oa-da-model-scf-season-daily

**Backfill model SCF**

Computes daily ROI-mean model SCF for all members (retroactive).

```bash
oa-da-model-scf-season-daily \
  --project-dir PATH \
  --season-dir PATH \
  --roi PATH \
  [--max-workers N]
```

Use this to add model SCF time series to an already-completed season.

---

### oa-da-model-wet-snow-season-daily

**Backfill wet snow**

Similar to `oa-da-model-scf-season-daily` but for wet snow classification.

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
# Season pipeline
docker compose run --rm oa oa-da-season \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020

# Snow-cover summary (GeoTIFF/NetCDF; MODIS after HDF→GeoTIFF with classes set in project.yml)
docker compose run --rm oa oa-da-snowcover \
  --input-dir /data/obs/snowcover \
  --season-label season_2019-2020 \
  --project-dir /data
```

---

## Next Steps

- [Configuration Guide]({{ site.baseurl }}{% link guides/configuration.md %}) - Configure commands via YAML
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments.md %}) - End-to-end workflow
- [Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}) - Common issues

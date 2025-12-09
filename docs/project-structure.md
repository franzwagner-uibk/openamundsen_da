---
layout: default
title: Project Structure
nav_order: 3
---

# Project Structure
{: .no_toc }

Understanding the directory layout and file organization.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Repository Structure

```
openamundsen_da/
├── openamundsen_da/          # Main Python package
│   ├── core/                 # Core functionality
│   ├── observer/             # Observation processing
│   ├── methods/              # DA methods
│   │   ├── pf/              # Particle filter
│   │   ├── h_of_x/          # Forward operators
│   │   ├── wet_snow/        # Wet snow classification
│   │   └── viz/             # Visualization
│   ├── pipeline/             # Pipeline orchestration
│   ├── util/                 # Utilities
│   └── io/                   # I/O operations
├── templates/                # Project templates
│   └── project/             # Template structure
├── docs/                     # Documentation (this site)
├── Dockerfile               # Docker container definition
├── compose.yml              # Docker Compose configuration
├── environment.yml          # Conda environment
├── pyproject.toml           # Package metadata
└── README.md                # Main README
```

---

## Package Organization

### Core Module (`core/`)

**Purpose**: Fundamental ensemble and forcing operations

| File | Description |
|:-----|:------------|
| `prior_forcing.py` | Generates perturbed meteorological forcing |
| `launch.py` | Parallel ensemble launcher |
| `runner.py` | Single member execution |
| `config.py` | Configuration loading and validation |
| `constants.py` | Package-wide constants |
| `env.py` | Environment setup |

### Observer Module (`observer/`)

**Purpose**: Satellite observation processing

| File | Description |
|:-----|:------------|
| `mod10a1_preprocess.py` | MODIS MOD10A1 HDF→GeoTIFF conversion |
| `satellite_scf.py` | SCF extraction from rasters |
| `snowflake_fsc.py` | Sentinel-2 FSC (Snowflake product) |
| `satellite_wet_snow_s1.py` | Sentinel-1 wet snow mask processing |
| `fraction_obs.py` | Generic fraction observation handling |
| `plot_fractions.py` | Observation visualization |
| `plot_scf_summary.py` | SCF summary plots |

### Methods Module (`methods/`)

**Purpose**: Data assimilation algorithms and analysis

#### Particle Filter (`methods/pf/`)

| File | Description |
|:-----|:------------|
| `assimilate_scf.py` | Likelihood weight calculation |
| `resample.py` | Systematic resampling implementation |
| `rejuvenate.py` | Ensemble perturbation after resampling |
| `plot_weights.py` | Weight visualization |
| `plot_ess_timeline.py` | ESS monitoring plots |

#### Forward Operators (`methods/h_of_x/`)

| File | Description |
|:-----|:------------|
| `model_scf.py` | Model state → SCF operator |

#### Wet Snow (`methods/wet_snow/`)

| File | Description |
|:-----|:------------|
| `classify.py` | LWC-based wet/dry classification |
| `area.py` | Wet snow area calculation |

#### Visualization (`methods/viz/`)

| File | Description |
|:-----|:------------|
| `plot_forcing_ensemble.py` | Forcing time series plots |
| `plot_results_ensemble.py` | Results time series plots |
| `plot_season_ensemble.py` | Season-wide ensemble plots |
| `plot_station_variable.py` | Single-station variable plots |
| `aggregate_fractions.py` | Fraction envelope aggregation |
| `_style.py` | Plotting style configuration |
| `_utils.py` | Plotting utilities |

### Pipeline Module (`pipeline/`)

**Purpose**: Workflow orchestration

| File | Description |
|:-----|:------------|
| `season.py` | Main season pipeline orchestrator |
| `season_skeleton.py` | Season directory structure builder |

{: .note }
> The pipeline module has no `__init__.py` - modules are executed as scripts.

### Utility Module (`util/`)

**Purpose**: Helper functions and utilities

| File | Description |
|:-----|:------------|
| `aoi.py` | Area of interest operations |
| `roi.py` | ROI polygon handling |
| `glacier_mask.py` | Glacier masking utilities |
| `da_events.py` | DA event parsing from season.yml |
| `perf_monitor.py` | Performance monitoring |
| `stats.py` | Statistical utilities |
| `ts.py` | Time series utilities |

---

## Project Data Structure

When you set up a data assimilation project, it follows this structure:

```
project/
├── env/                      # Environment data
│   ├── roi.gpkg             # Study area polygon (required)
│   └── glaciers.gpkg        # Glacier outlines (optional)
│
├── meteo/                    # Meteorological forcing
│   ├── stations.csv         # Station metadata
│   ├── station_001.csv      # Long-span forcing time series
│   ├── station_002.csv
│   └── ...
│
├── obs/                      # Observations
│   └── season_2019-2020/
│       ├── scf_summary.csv           # Season-wide SCF summary
│       ├── wet_snow_summary.csv      # Season-wide wet snow summary
│       ├── NDSI_Snow_Cover_*.tif     # Preprocessed SCF rasters
│       └── ...
│
├── propagation/              # Ensemble propagation (created by framework)
│   └── season_2019-2020/
│       ├── season.yml               # Season configuration
│       ├── step_00_init/
│       │   ├── step_00_init.yml
│       │   └── ensembles/
│       │       ├── prior/
│       │       │   ├── open_loop/
│       │       │   ├── member_001/
│       │       │   ├── member_002/
│       │       │   └── ...
│       │       └── posterior/       # Created after resampling
│       │           ├── member_001/
│       │           └── ...
│       ├── step_01_YYYYMMDD-YYYYMMDD/
│       │   ├── step_01.yml
│       │   ├── ensembles/
│       │   ├── assim/               # Assimilation outputs
│       │   │   ├── weights_scf_YYYYMMDD.csv
│       │   │   └── indices_YYYYMMDD.csv
│       │   └── obs/                 # Per-step observations
│       │       └── obs_scf_MOD10A1_YYYYMMDD.csv
│       ├── step_02_.../
│       └── plots/                   # All visualization outputs
│           ├── forcing/
│           ├── results/
│           ├── assim/
│           │   ├── weights/
│           │   └── ess/
│           └── perf/
│
└── project.yml               # Main project configuration (required)
```

---

## Configuration Files

### project.yml (Required)

Main configuration file defining:
- Model settings (timestep, domain)
- Ensemble configuration (size, perturbations)
- DA parameters (H(x), resampling, rejuvenation)
- Paths and environment variables

See [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) for details.

### season.yml (Required for each season)

Season-specific configuration:
- Start and end dates
- Assimilation dates/events
- Season-specific overrides

### step_XX.yml (Auto-generated)

Step-specific configuration:
- Step boundaries (start_date, end_date)
- Results directory
- Warm-start pointers

---

## Member Directory Structure

Each ensemble member has this structure:

```
member_001/
├── config.yml               # Merged openAMUNDSEN config
├── forcing/                 # Perturbed forcing
│   ├── station_001.csv
│   └── ...
├── results/                 # Model outputs
│   ├── grids/              # NetCDF grids
│   │   ├── snow.nc
│   │   └── meteo.nc
│   ├── point_station.csv    # Point time series
│   ├── point_scf_roi.csv    # Model SCF (when enabled)
│   └── state_YYYYMMDD_HHMMSS.nc  # Model state
└── state_pointer.json       # Points to current state file
```

---

## Data Flow

```mermaid
graph TD
    A[meteo/] --> B[Prior Forcing]
    B --> C[member_001..N/]
    C --> D[openAMUNDSEN Run]
    D --> E[results/]
    F[obs/] --> G[H(x) Forward Operator]
    E --> G
    G --> H[Particle Filter]
    H --> I[Resampling]
    I --> J[Rejuvenation]
    J --> K[Next Step Prior]
    K --> C
```

---

## File Naming Conventions

### Observations

- **SCF observations**: `obs_scf_{PRODUCT}_{YYYYMMDD}.csv`
  - Example: `obs_scf_MOD10A1_20191122.csv`
  - Example: `obs_scf_SNOWFLAKE_20200315.csv`

- **Wet snow observations**: `obs_wet_snow_{PRODUCT}_{YYYYMMDD}.csv`
  - Example: `obs_wet_snow_S1_20200401.csv`

### Assimilation Outputs

- **Weights**: `weights_{variable}_{YYYYMMDD}.csv`
  - Example: `weights_scf_20191122.csv`
  - Example: `weights_wet_snow_20200401.csv`

- **Resampling indices**: `indices_{YYYYMMDD}.csv`

### Model States

- **State files**: `state_{YYYYMMDD}_{HHMMSS}.nc`
  - Example: `state_20191122_000000.nc`

---

## Next Steps

- [Workflow Overview]({{ site.baseurl }}{% link workflow.md %}) - Understanding the DA cycle
- [Configuration Guide]({{ site.baseurl }}{% link guides/configuration.md %}) - Detailed configuration reference
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments.md %}) - Setting up your first experiment

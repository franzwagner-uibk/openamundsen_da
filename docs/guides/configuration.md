---
layout: default
title: Configuration Reference
parent: Guides
nav_order: 2
---

# Configuration Reference
{: .no_toc }

Complete YAML configuration reference for openamundsen_da.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Configuration Hierarchy

openamundsen_da uses a three-level configuration hierarchy:

1. **project.yml** - Project-wide settings (required)
2. **season.yml** - Season-specific settings (required for each season)
3. **step_XX.yml** - Step-specific settings (auto-generated)

Each level can override settings from the level above.

---

## project.yml

The main configuration file that defines all project-wide settings.

### Basic Configuration

```yaml
# Model settings
model: openamundsen
timestep: 3H  # Model timestep (1H, 3H, 24H, etc.)

# Domain settings
domain:
  aoi: env/roi.gpkg  # Area of interest (required)
  crs: EPSG:32632    # Coordinate reference system
```

### Ensemble Configuration

```yaml
ensemble:
  size: 30  # Number of ensemble members

  # Prior forcing perturbations
  prior_forcing:
    sigma_t: 1.5      # Temperature perturbation std (K)
    sigma_p: 0.20     # Precipitation perturbation std (multiplicative)
    seed: 42          # Random seed for reproducibility
    rebase: false     # Rebase mode (default: false)
```

#### Perturbation Details

**Temperature Perturbations** (`sigma_t`):
- Additive Gaussian noise: `T_perturbed = T + ε`, where `ε ~ N(0, σ_T²)`
- Typical range: 0.5-2.0 K
- Larger values → more ensemble spread
- Consider regional climate uncertainty when choosing

**Precipitation Perturbations** (`sigma_p`):
- Multiplicative log-normal noise: `P_perturbed = P × exp(ε)`, where `ε ~ N(0, σ_P²)`
- Typical range: 0.10-0.30 (10-30% uncertainty)
- Preserves zero values (no precipitation when P=0)
- Larger values → more uncertainty in precipitation amounts

**Rebase Mode**:
- `false` (default): Perturbations applied to member's own previous forcing
- `true`: Perturbations applied relative to open loop forcing

---

### Data Assimilation Configuration

```yaml
data_assimilation:
  # H(x) forward operator configuration
  h_of_x:
    variable: hs              # State variable: 'hs' (snow depth) or 'swe'
    method: logistic          # Method: 'depth_threshold' or 'logistic'
    h0: 0.05                 # Threshold (m)
    k: 50.0                  # Steepness parameter (logistic only)

  # Observation errors
  observation_error:
    scf: 0.10                # SCF observation error std
    wet_snow: 0.15           # Wet snow observation error std

  # Resampling configuration
  resampling:
    algorithm: systematic     # Algorithm: 'systematic' (only option currently)
    ess_threshold_ratio: 0.5  # Resample if ESS < ratio × N
    seed: 42                 # Random seed

  # Rejuvenation (post-resampling perturbations)
  rejuvenation:
    enabled: true
    sigma_t: 0.2             # Temperature perturbation (smaller than prior)
    sigma_p: 0.2             # Precipitation perturbation
    rebase: true             # Usually true for rejuvenation
    seed: 42

  # Glacier masking
  glacier_mask:
    enabled: true
    path: env/glaciers.gpkg  # Glacier outlines (optional)
```

#### H(x) Forward Operator Methods

**Depth Threshold** (`depth_threshold`):
```
SCF(x) = 1  if HS(x) > h0
         0  otherwise
```
- Binary step function
- Simple and fast
- Parameter: `h0` (threshold in meters)
- Typical value: 0.01-0.10 m

**Logistic** (`logistic`):
```
SCF(x) = 1 / (1 + exp(-k × (HS(x) - h0)))
```
- Smooth transition
- More realistic for coarse grids
- Parameters:
  - `h0`: Midpoint threshold (m)
  - `k`: Steepness (higher = steeper transition)
- Typical values:
  - `h0`: 0.03-0.08 m
  - `k`: 30-100

**Variable Selection**:
- `hs`: Snow depth (default, recommended)
- `swe`: Snow water equivalent

#### Resampling Parameters

**ESS Threshold**:
- `ess_threshold_ratio = 0.5`: Resample when ESS < 50% of N
- Lower values (0.3-0.4): Less frequent resampling, risk of degeneracy
- Higher values (0.6-0.7): More frequent resampling, may lose diversity

**Effective Sample Size (ESS)**:
```
ESS = 1 / Σ(w_i²)
```
- Range: [1, N]
- ESS = N: All weights equal (uniform)
- ESS = 1: One particle has all weight (degenerate)

#### Glacier Masking

When enabled, glacier-covered areas are excluded from observation-model comparisons:
- Prevents assimilating firn/ice observations into seasonal snow model
- Requires glacier outline vector (GeoPackage or Shapefile)
- Applied during H(x) computation and likelihood calculation

---

### Environment Variables

```yaml
environment:
  GDAL_DATA: /usr/share/gdal
  PROJ_LIB: /usr/share/proj
  NUMEXPR_MAX_THREADS: 8
  OMP_NUM_THREADS: 1
```

Commonly used variables:
- `GDAL_DATA`: GDAL data directory path
- `PROJ_LIB`: PROJ library data path
- `NUMEXPR_MAX_THREADS`: NumPy parallelization
- `OMP_NUM_THREADS`: OpenMP threads (set to 1 to avoid over-subscription)

---

### openAMUNDSEN Configuration

You can include openAMUNDSEN-specific configuration directly in `project.yml`:

```yaml
# openAMUNDSEN model configuration
output_data:
  grids:
    format: netcdf
    variables:
      - snow_depth
      - snow_water_equivalent
      - surface_temperature
      - albedo
      - lwc

  timeseries:
    format: csv
    variables:
      - snow_depth
      - snow_water_equivalent
```

### Key Output Variables for Data Assimilation

For DA workflows, configure these essential variables in `project.yml`:

```yaml
output_data:
  grids:
    format: netcdf
    variables:
      - var: snow.swe           # Snow water equivalent (essential for DA)
        name: swe
        freq: D                 # Daily output
      - var: snow.depth         # Snow depth (for H(x) operator)
        name: hs
        freq: D
      - var: snow.albedo        # Snow albedo
        name: albedo
        freq: D
      - var: snow.lwc           # Liquid water content (for wet snow DA)
        name: lwc
        freq: D
```

**Available aggregation options**:
- `agg: sum` - Sum over period (e.g., for snowmelt)
- `agg: mean` - Mean over period
- (empty) - Instantaneous values

**Frequency codes**:
- `D`: Daily
- `M`: Monthly
- Specific dates: `[2019-11-22, 2019-12-10]`

See [openAMUNDSEN Output Data documentation](http://doc.openamundsen.org/doc/output) for complete variable list and [Configuration documentation](http://doc.openamundsen.org/doc/configuration) for all model options.

---

## season.yml

Season-specific configuration stored in `propagation/season_YYYY-YYYY/season.yml`.

```yaml
# Season boundaries
start_date: 2019-11-01
end_date: 2020-07-31

# Assimilation dates
assimilation_dates:
  - 2019-11-22  # First SCF observation
  - 2019-12-10
  - 2020-01-15
  - 2020-02-20
  - 2020-03-18
  - 2020-04-12  # Wet snow observation
  - 2020-05-05

# Season-specific overrides
ensemble:
  size: 40  # Override project-wide ensemble size

data_assimilation:
  observation_error:
    scf: 0.08  # Override for this season
```

### Assimilation Events

Alternatively, use structured event definitions:

```yaml
assimilation_events:
  - date: 2019-11-22
    type: scf
    product: MOD10A1

  - date: 2020-04-12
    type: wet_snow
    product: S1
```

This provides more metadata for tracking observation sources.

---

## step_XX.yml

Step-specific configuration (auto-generated by season skeleton builder).

```yaml
# Step boundaries
start_date: 2019-11-22
end_date: 2019-12-10

# Results configuration
results:
  directory: step_01_20191122-20191210
  state_save: true

# Warm-start pointers (populated after run)
warm_start:
  prior:
    member_001: step_00_init/ensembles/posterior/member_001/results/state_20191122_000000.nc
    member_002: step_00_init/ensembles/posterior/member_002/results/state_20191122_000000.nc
    # ...
```

These files are usually not edited manually.

---

## Example: Complete project.yml

```yaml
# =============================================================================
# openamundsen_da Project Configuration
# =============================================================================

# Basic model settings
model: openamundsen
timestep: 3H

# Domain
domain:
  aoi: env/roi.gpkg
  crs: EPSG:32632

# Ensemble configuration
ensemble:
  size: 30
  prior_forcing:
    sigma_t: 1.5
    sigma_p: 0.20
    seed: 42

# Data assimilation
data_assimilation:
  h_of_x:
    variable: hs
    method: logistic
    h0: 0.05
    k: 50.0

  observation_error:
    scf: 0.10
    wet_snow: 0.15

  resampling:
    algorithm: systematic
    ess_threshold_ratio: 0.5
    seed: 42

  rejuvenation:
    enabled: true
    sigma_t: 0.2
    sigma_p: 0.2
    rebase: true
    seed: 42

  glacier_mask:
    enabled: true
    path: env/glaciers.gpkg

# Environment
environment:
  GDAL_DATA: /usr/share/gdal
  PROJ_LIB: /usr/share/proj
  NUMEXPR_MAX_THREADS: 8
  OMP_NUM_THREADS: 1

# openAMUNDSEN configuration
output_data:
  grids:
    format: netcdf
    variables:
      - snow_depth
      - snow_water_equivalent
      - albedo
      - lwc
  timeseries:
    format: csv
    variables:
      - snow_depth
      - snow_water_equivalent
```

---

## Configuration Validation

The framework validates configuration on load:

```python
from openamundsen_da.core.config import load_config

config = load_config('/data/project.yml')
```

**Common validation errors**:
- Missing required keys (`ensemble.size`, `data_assimilation.h_of_x`, etc.)
- Invalid timestep format (must be pandas-compatible, e.g., '3H')
- Invalid CRS (must be valid EPSG code or PROJ string)
- File paths that don't exist (ROI, glacier mask)

---

## Best Practices

### Ensemble Size

- **Small domains** (< 100 km²): 20-30 members
- **Medium domains** (100-500 km²): 30-50 members
- **Large domains** (> 500 km²): 50-100 members

Trade-off: Computational cost vs. posterior quality.

### Perturbation Magnitudes

**Conservative** (small uncertainty):
```yaml
ensemble:
  prior_forcing:
    sigma_t: 1.0
    sigma_p: 0.15
```

**Moderate** (typical):
```yaml
ensemble:
  prior_forcing:
    sigma_t: 1.5
    sigma_p: 0.20
```

**Aggressive** (high uncertainty):
```yaml
ensemble:
  prior_forcing:
    sigma_t: 2.5
    sigma_p: 0.30
```

### Rejuvenation

Use **smaller** perturbations than prior:
```yaml
data_assimilation:
  rejuvenation:
    sigma_t: 0.2   # vs. prior: 1.5
    sigma_p: 0.2   # vs. prior: 0.20
```

This maintains diversity without over-perturbing after resampling.

### Random Seeds

For reproducibility, set seeds explicitly:
```yaml
ensemble:
  prior_forcing:
    seed: 42

data_assimilation:
  resampling:
    seed: 42
  rejuvenation:
    seed: 42
```

Use different seeds for different processes if needed.

---

## Next Steps

- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) - Command-line interface
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments.md %}) - Complete workflow example
- [Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}) - Common configuration issues

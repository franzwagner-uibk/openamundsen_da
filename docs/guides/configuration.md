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

{: .note }

> This guide covers openamundsen_da-specific configuration. For openAMUNDSEN model configuration, see the [openAMUNDSEN Configuration Guide](http://doc.openamundsen.org/en/stable/configuration.html).

<details markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

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
domain: "your_domain"
resolution: 100 # spatial resolution (m)
timestep: "3H" # temporal resolution (pandas-compatible string)
crs: "epsg:32632" # CRS of the input grids
timezone: 1 # UTC offset in hours
```

### Prior Forcing Configuration

```yaml
data_assimilation:
  prior_forcing:
    ensemble_size: 20 # number of ensemble members
    random_seed: 42 # RNG seed for reproducibility
    sigma_t: 0.5 # additive temperature stddev (deg C)
    mu_p: 0.0 # log-space mean for precip factor
    sigma_p: 0.5 # log-space stddev for precip factor
```

#### Perturbation Details

**Temperature Perturbations** (`sigma_t`):

- Additive Gaussian noise: `T_perturbed = T + ε`, where `ε ~ N(0, σ_T²)`
- Typical range: 0.5-2.0 K

**Precipitation Perturbations** (`sigma_p`, `mu_p`):

- Multiplicative log-normal noise: `P_perturbed = P × exp(ε)`, where `ε ~ N(μ_P, σ_P²)`
- Typical range for sigma_p: 0.15-0.50
- mu_p is typically 0.0

---

### Data Assimilation Configuration

```yaml
data_assimilation:
  # H(x) forward operator configuration
  h_of_x:
    method: depth_threshold # or "logistic"
    variable: hs # or "swe"
    params:
      h0: 0.01
      k: 80

  # Likelihood settings
  likelihood:
    scf:
      obs_sigma: 0.10
      use_binomial: true
      sigma_floor: 0.05
      sigma_cloud_scale: 0.10
      min_sigma: 0.03
    wet_snow:
      obs_sigma: 0.15
      use_binomial: false

  # Resampling configuration
  resampling:
    algorithm: systematic
    ess_threshold_ratio: 0.5 # Resample if ESS < ratio × N

  # Rejuvenation (post-resampling perturbations)
  rejuvenation:
    sigma_t: 0.2 # Additive temperature noise (deg C)
    sigma_p: 0.2 # Lognormal sigma for precip factor (mu=0)

  # Glacier masking
  glacier_mask:
    enabled: true
    path: env/glaciers.gpkg

  # Warm start settings
  restart:
    use_state: true
    dump_state: true
    state_pattern: model_state.pickle.gz
    cleanup_after_season: true  # delete state pickle files after a successful season run
```

`cleanup_after_season` defaults to `true` and removes state pickle files after a successful full-season run to save disk space. Set it to `false` to keep state files for debugging or manual restarts.

Manual cleanup is available regardless of the toggle:

```powershell
# Single season
oa-da-clean-season --project-dir /data --season-dir /data/propagation/season_YYYY-YYYY

# All seasons under project/propagation
oa-da-clean-season --project-dir /data --all-seasons
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

You must include openAMUNDSEN-specific configuration directly in `project.yml`:

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
      - var: snow.swe # Snow water equivalent (essential for DA)
        name: swe
        freq: D # Daily output
      - var: snow.depth # Snow depth (for H(x) operator)
        name: hs
        freq: D
      - var: snow.albedo # Snow albedo
        name: albedo
        freq: D
      - var: snow.lwc # Liquid water content (for wet snow DA)
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
```

This format provides metadata about observation sources and variable types.

---

## step_XX.yml

Step-specific configuration (auto-generated by season skeleton builder).

```yaml
# Step boundaries
start_date: 2019-11-22
end_date: 2019-12-10
results_dir: results
```

These files are usually not edited manually. The framework uses `state_pointer.json` files within each member directory to track warm-start state locations.

---

## Example: Complete project.yml

```yaml
domain: "example_domain"
resolution: 100
timestep: "3H"
crs: "epsg:32632"
timezone: 1

environment:
  GDAL_DATA: "/path/to/conda/env/share/gdal"
  PROJ_LIB: "/path/to/conda/env/share/proj"

data_assimilation:
  prior_forcing:
    ensemble_size: 20
    random_seed: 42
    sigma_t: 0.5
    mu_p: 0.0
    sigma_p: 0.5

  h_of_x:
    method: depth_threshold
    variable: hs
    params:
      h0: 0.01
      k: 80

  wet_snow:
    classification_threshold_percent: 0.5

  glacier_mask:
    enabled: true
    path: env/glaciers.gpkg

  likelihood:
    scf:
      obs_sigma: 0.10
      use_binomial: true
      sigma_floor: 0.05
      sigma_cloud_scale: 0.10
      min_sigma: 0.03
    wet_snow:
      obs_sigma: 0.10
      use_binomial: false

  resampling:
    algorithm: systematic
    ess_threshold_ratio: 0.5

  rejuvenation:
    sigma_t: 0.2
    sigma_p: 0.2

  restart:
    use_state: false
    dump_state: true
    state_pattern: model_state.pickle.gz
```

---

## Configuration Validation

Configuration is checked when you run the CLI (for example `oa-da-season`, `oa-da-mod10a1`, or `oa-da-assimilate-scf`). Internally, the framework merges YAML layers and hands the merged model configuration to openAMUNDSEN for parsing.

If something is missing or inconsistent, the CLI will fail early with a descriptive error message (missing required keys, invalid timestep format, missing files like ROI/glacier masks, etc.).

---

## Best Practices

### Perturbation Magnitudes

**Prior forcing** (typical values from README):

```yaml
data_assimilation:
  prior_forcing:
    sigma_t: 0.5 # Temperature: 0.5-2.0 K typical
    sigma_p: 0.5 # Precipitation: 0.15-0.50 typical
```

**Rejuvenation** - use smaller values than prior:

```yaml
data_assimilation:
  rejuvenation:
    sigma_t: 0.2 # Usually smaller than prior
    sigma_p: 0.2
```

If rejuvenation sigmas are not set, they fall back to prior_forcing sigmas.

### Random Seeds

For reproducibility, set seeds explicitly:

```yaml
data_assimilation:
  prior_forcing:
    random_seed: 42
```

The resampling and rejuvenation use the prior_forcing seed as fallback if not specified separately.

---

## Next Steps

- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) - Command-line interface
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments.md %}) - Complete workflow example
- [Troubleshooting]({{ site.baseurl }}{% link advanced/troubleshooting.md %}) - Common configuration issues

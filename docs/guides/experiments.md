---
layout: default
title: Running Experiments
parent: Guides
nav_order: 4
---

# Running Experiments
{: .no_toc }

Complete walkthrough for setting up and running a data assimilation experiment.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

This guide walks you through a complete seasonal data assimilation experiment from scratch, covering:

1. Project setup
2. Data preparation
3. Configuration
4. Observation preprocessing
5. Running the season pipeline
6. Analysis and visualization

**Estimated time**: 2-4 hours (depending on domain size and ensemble size)

---

## Prerequisites

Before starting, ensure you have:

- ✅ Docker installed and running
- ✅ openamundsen_da repository cloned
- ✅ Docker image built (`docker build -t oa-da .`)
- ✅ `.env` file configured with your paths

See the [Installation Guide]({{ site.baseurl }}{% link installation.md %}) if you haven't completed these steps.

---

## Step 1: Project Setup

### 1.1 Copy Project Template

```bash
# Copy template to your project directory
cp -r templates/project /path/to/my_project

# Verify structure
ls /path/to/my_project
# Should see: env/ meteo/ obs/ project.yml
```

### 1.2 Define Study Area (ROI)

Create a Region of Interest (ROI) polygon in GeoPackage or Shapefile format:

**Requirements**:
- Single-feature polygon
- Valid CRS (preferably projected, e.g., UTM)
- Field: Any identifier field (e.g., "name", "id")

**Example using QGIS**:
1. Open QGIS
2. Create new GeoPackage layer (`env/roi.gpkg`)
3. Draw your study area polygon
4. Add attribute field (e.g., "name" = "study_area")
5. Save

**Example using Python**:
```python
import geopandas as gpd
from shapely.geometry import Polygon

# Define bounding box (UTM Zone 32N coordinates)
coords = [
    (650000, 5200000),
    (680000, 5200000),
    (680000, 5230000),
    (650000, 5230000),
    (650000, 5200000)
]

gdf = gpd.GeoDataFrame(
    {'name': ['study_area']},
    geometry=[Polygon(coords)],
    crs='EPSG:32632'
)

gdf.to_file('env/roi.gpkg', driver='GPKG')
```

### 1.3 (Optional) Glacier Outlines

If your domain includes glaciers, add glacier outlines:

**Sources**:
- [Randolph Glacier Inventory (RGI)](https://www.glims.org/RGI/)
- National glacier inventories

**Preparation**:
```bash
# Clip to ROI extent and reproject if needed
ogr2ogr -clipsrc env/roi.gpkg -t_srs EPSG:32632 \
  env/glaciers.gpkg \
  /path/to/rgi_region.shp
```

---

## Step 2: Data Preparation

### 2.1 Meteorological Forcing

Prepare meteorological station data in openAMUNDSEN format. See the [openAMUNDSEN Input Data documentation](http://doc.openamundsen.org/doc/input) for complete details.

**Station metadata** (`meteo/stations.csv`):
```csv
station_id,x,y,z,name
1,655000,5215000,1500,Station_A
2,672000,5220000,2100,Station_B
```

**Station time series** (`meteo/station_001.csv`, etc.):
```csv
date,temp,precip,rel_hum,wind_speed,sw_in
2019-11-01 00:00:00,275.65,0.0,75,3.2,0
2019-11-01 03:00:00,274.95,0.5,82,2.9,50
...
```

**Required variables** (CSV format):
- `date`: Timestamp (YYYY-MM-DD HH:MM)
- `temp`: Air temperature (K) - **Note: Kelvin, not Celsius**
- `precip`: Precipitation sum (kg m⁻²) - equivalent to mm per timestep
- `rel_hum`: Relative humidity (%)
- `wind_speed`: Wind speed (m s⁻¹)
- `sw_in`: Global radiation (W m⁻²)

{: .note }
> Time series must cover the entire simulation period plus buffer for spin-up. openAMUNDSEN also supports NetCDF format (CF-1.6 conventions) - see documentation for details.

### 2.2 Satellite Observations

Download satellite observations for your season.

#### MODIS MOD10A1 (Snow Cover)

**Download**:
- Source: [NASA Earthdata](https://search.earthdata.nasa.gov/)
- Product: MOD10A1 v6.1
- Format: HDF (download all tiles covering your ROI)

**Organization**:
```
obs/MOD10A1_61_HDF/
├── MOD10A1.A2019326.h18v04.061.*.hdf
├── MOD10A1.A2019327.h18v04.061.*.hdf
└── ...
```

#### Sentinel-2 FSC (Optional)

**Download**:
- Source: [Snowflake](https://www.snowflake-project.eu/) or [Theia Snow Collection](https://www.theia-land.fr/)
- Product: Fractional Snow Cover (FSC)
- Format: GeoTIFF

#### Sentinel-1 Wet Snow (Optional)

**Download**:
- Source: Custom processing or [SWI product](https://land.copernicus.eu/global/products/swi)
- Format: GeoTIFF with wet snow mask

---

## Step 3: Configuration

### 3.1 Edit project.yml

Edit `project.yml` with your settings:

```yaml
# Basic settings
model: openamundsen
timestep: 3H

# Domain
domain:
  aoi: env/roi.gpkg
  crs: EPSG:32632

# Ensemble
ensemble:
  size: 30  # Start with 30 for testing
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
    enabled: true  # Set false if no glaciers
    path: env/glaciers.gpkg

# Environment
environment:
  GDAL_DATA: /usr/share/gdal
  PROJ_LIB: /usr/share/proj

# openAMUNDSEN output
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
```

See [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) for all options.

### 3.2 Create season.yml

Create `propagation/season_2019-2020/season.yml`:

```yaml
# Season boundaries
start_date: 2019-11-01
end_date: 2020-07-31

# Assimilation dates (will be populated after preprocessing)
assimilation_dates: []
```

We'll populate `assimilation_dates` after preprocessing observations.

---

## Step 4: Observation Preprocessing

### 4.1 MODIS MOD10A1 Preprocessing

Process MODIS HDF files to GeoTIFF and generate season summary:

```bash
docker compose run --rm oa oa-da-mod10a1 \
  --input-dir /data/obs/MOD10A1_61_HDF \
  --season-label season_2019-2020 \
  --project-dir /data \
  --target-epsg 32632 \
  --resolution 500 \
  --ndsi-threshold 0.4
```

**Output**:
- `obs/season_2019-2020/NDSI_Snow_Cover_YYYYMMDD.tif` (per date)
- `obs/season_2019-2020/scf_summary.csv` (season summary)

**Inspect summary**:
```bash
head obs/season_2019-2020/scf_summary.csv
```

Should see:
```csv
date,product,scf_mean,scf_std,pixel_count
2019-11-22,MOD10A1,0.45,0.32,12500
2019-11-23,MOD10A1,0.52,0.28,12500
...
```

### 4.2 Update season.yml with Observation Dates

Extract dates from summary and update `season.yml`:

```bash
# Extract dates (excluding cloudy/low-quality dates)
awk -F, 'NR>1 && $3>0.1 {print $1}' obs/season_2019-2020/scf_summary.csv > dates.txt
```

Edit `propagation/season_2019-2020/season.yml`:

```yaml
start_date: 2019-11-01
end_date: 2020-07-31

assimilation_dates:
  - 2019-11-22
  - 2019-11-25
  - 2019-12-03
  # ... (paste dates from dates.txt)
```

{: .note }
> You can thin dates (e.g., every 7-10 days) to reduce computational cost.

---

## Step 5: Build Season Skeleton

Create the season directory structure with step folders:

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.season_skeleton \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020
```

**Output**:
```
propagation/season_2019-2020/
├── season.yml
├── step_00_init/
│   └── step_00_init.yml
├── step_01_20191122-20191125/
│   └── step_01.yml
├── step_02_20191125-20191203/
│   └── step_02.yml
└── ...
```

**Verify**:
```bash
ls -d propagation/season_2019-2020/step_*
```

---

## Step 6: Extract SCF Observations Per Step

Distribute SCF observations to step directories:

```bash
docker compose run --rm oa oa-da-scf \
  --season-dir /data/propagation/season_2019-2020 \
  --summary-csv /data/obs/season_2019-2020/scf_summary.csv \
  --overwrite
```

**Output** (per step):
```
step_01_20191122-20191125/obs/obs_scf_MOD10A1_20191122.csv
```

**Verify**:
```bash
find propagation/season_2019-2020 -name "obs_scf_*.csv" | wc -l
# Should match number of assimilation dates
```

---

## Step 7: Run Season Pipeline

### 7.1 Test Run (Single Step)

Before running the full season, test with a single step:

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.season \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020 \
  --max-workers 4 \
  --stop-after-step 1
```

**What happens**:
1. ✅ Generate prior forcing (step 00)
2. ✅ Run prior ensemble (step 00)
3. ✅ Generate prior forcing (step 01)
4. ✅ Run prior ensemble (step 01)
5. ✅ Compute model SCF
6. ✅ Assimilate → compute weights
7. ✅ Resample (if ESS < threshold)
8. ✅ Rejuvenate
9. ✅ Generate plots
10. ⏹️ Stop

**Check outputs**:
```bash
ls propagation/season_2019-2020/step_01_*/assim/
# Should see: weights_scf_YYYYMMDD.csv, indices_YYYYMMDD.csv

ls propagation/season_2019-2020/plots/assim/weights/
# Should see: step_01_weights.png
```

### 7.2 Full Season Run

If test run succeeded, run the full season:

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.season \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020 \
  --max-workers 8 \
  --monitor-perf
```

**Options**:
- `--max-workers 8`: Use 8 parallel workers (adjust based on CPU count)
- `--monitor-perf`: Enable performance monitoring
- `--overwrite`: Overwrite existing outputs (use with caution)
- `--no-live-plots`: Skip live plotting (plot at end only)

**Estimated runtime**:
- **Small domain** (< 100 km²), N=30, 10 steps: 2-4 hours
- **Medium domain** (100-500 km²), N=50, 20 steps: 6-12 hours
- **Large domain** (> 500 km²), N=100, 30 steps: 24-48 hours

{: .note }
> Pipeline progress is logged to console and season log file. You can safely interrupt (Ctrl+C) and resume later.

---

## Step 8: Analysis & Visualization

### 8.1 Inspect ESS Evolution

Check effective sample size timeline:

```bash
docker compose run --rm oa oa-da-plot-ess \
  --season-dir /data/propagation/season_2019-2020
```

**Output**: `plots/assim/ess/season_ess_timeline_season_2019-2020.png`

**Interpretation**:
- ESS near N → Low information from observations
- ESS near threshold → Resampling triggered
- ESS = 1 → Severe degeneracy (check observation error settings)

### 8.2 SCF Time Series

Plot observed vs. modeled SCF:

```bash
docker compose run --rm oa oa-da-plot-scf \
  --season-dir /data/propagation/season_2019-2020 \
  --project-dir /data
```

**Output**: `plots/results/fraction_timeseries.png`

Shows:
- Observed SCF (points)
- Prior ensemble mean ± 90% envelope
- Posterior ensemble mean ± 90% envelope
- Open loop

### 8.3 Particle Weights

Inspect weights for a specific assimilation date:

```bash
docker compose run --rm oa oa-da-plot-weights \
  propagation/season_2019-2020/step_05_*/assim/weights_scf_20200115.csv
```

**Output**: `plots/assim/weights/step_05_weights.png`

Shows:
- Normalized particle weights (bar plot)
- Observation-model residuals (histogram)
- ESS value

### 8.4 Results Ensemble Plots

Plot SWE/snow depth time series for specific locations:

```bash
docker compose run --rm oa \
  python -m openamundsen_da.methods.viz.plot_results_ensemble \
  --season-dir /data/propagation/season_2019-2020 \
  --variable snow_water_equivalent
```

**Output**: `plots/results/swe_ensemble_timeseries.png`

---

## Step 9: Model Evaluation

### 9.1 Compare with Independent Observations

If you have in-situ observations (not assimilated), compare:

```python
import pandas as pd
import xarray as xr

# Load posterior member results
posterior = xr.open_dataset('propagation/season_2019-2020/step_XX/ensembles/posterior/member_001/results/grids/snow.nc')

# Load in-situ observations
obs = pd.read_csv('path/to/insitu_swe.csv', parse_dates=['time'])

# Extract model at station location
model_swe = posterior['swe'].sel(x=station_x, y=station_y, method='nearest').to_pandas()

# Compute metrics
rmse = ((model_swe - obs['swe'])**2).mean()**0.5
bias = (model_swe - obs['swe']).mean()
```

### 9.2 Ensemble Spread Analysis

Check if ensemble spread is appropriate:

```python
import xarray as xr
import glob

# Load all posterior members for a date
members = []
for path in glob.glob('propagation/season_2019-2020/step_XX/ensembles/posterior/member_*/results/grids/snow.nc'):
    ds = xr.open_dataset(path)
    members.append(ds['swe'].sel(time='2020-03-01'))

# Stack into ensemble dimension
ensemble = xr.concat(members, dim='member')

# Compute ensemble spread
spread = ensemble.std(dim='member')
mean = ensemble.mean(dim='member')

# Coefficient of variation
cv = spread / mean
print(f"Mean CV: {cv.mean().values:.2f}")
```

**Interpretation**:
- CV < 0.2: Ensemble may be under-dispersed
- CV = 0.2-0.4: Appropriate spread
- CV > 0.5: Ensemble may be over-dispersed

---

## Troubleshooting

### Issue: ESS always near N (no resampling)

**Cause**: Observation error too large, or model-obs mismatch too small

**Solution**:
- Reduce `observation_error.scf` in `project.yml`
- Check if observations are meaningful (not all 0 or 1)
- Verify H(x) parameters (`h0`, `k`)

### Issue: ESS drops to 1 immediately

**Cause**: Observation error too small, or severe model-obs mismatch

**Solution**:
- Increase `observation_error.scf`
- Check ensemble spread (may need larger `sigma_t`, `sigma_p`)
- Verify observations are correctly preprocessed

### Issue: Pipeline crashes during ensemble run

**Cause**: openAMUNDSEN configuration error, or resource limits

**Solution**:
- Check openAMUNDSEN config in `project.yml`
- Verify forcing data quality
- Reduce `max_workers` if RAM is limited
- Check Docker resource limits (`.env`: `MEMORY=16G`)

### Issue: High memory usage

**Solution**:
- Reduce ensemble size
- Reduce domain resolution
- Limit output variables in `project.yml`
- Set `NUMEXPR_MAX_THREADS=4` in environment

See [Troubleshooting Guide]({{ site.baseurl }}{% link advanced/troubleshooting.md %}) for more issues.

---

## Next Steps

### Experiment Variations

Try different configurations:

1. **Ensemble size sensitivity**:
   - Run with N=20, 30, 50, 100
   - Compare posterior uncertainty

2. **Perturbation magnitude**:
   - Test different `sigma_t`, `sigma_p` values
   - Assess impact on ensemble spread

3. **H(x) method**:
   - Compare `depth_threshold` vs. `logistic`
   - Test different `h0`, `k` parameters

4. **Observation thinning**:
   - Assimilate every 7 days vs. every 1-2 days
   - Evaluate impact on skill

### Advanced Topics

- [Performance Tuning]({{ site.baseurl }}{% link advanced/performance.md %}) - Optimize runtime
- [API Reference]({{ site.baseurl }}{% link reference/api.md %}) - Use Python API directly
- [Custom DA Methods]({{ site.baseurl }}{% link reference/da-methods.md %}) - Implement custom algorithms

---

## Further Reading

- Margulis, S.A., et al. (2019). *A particle batch smoother approach to snow water equivalent estimation*. JGR Atmospheres.
- Griessinger, N., et al. (2019). *Assimilation of snow cover derived from MODIS into a snow model*. The Cryosphere.
- openAMUNDSEN documentation: [https://openamundsen.readthedocs.io/](https://openamundsen.readthedocs.io/)

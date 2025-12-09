---
layout: default
title: Troubleshooting
parent: Advanced Topics
nav_order: 2
---

# Troubleshooting
{: .no_toc }

Common issues and solutions for openamundsen_da.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Installation Issues

### Docker "Cannot connect to Docker daemon"

**Problem**: Error when running `docker compose` commands

**Cause**: Docker service not running

**Solution**:

**Windows/macOS**:
```bash
# Start Docker Desktop application
# Wait for "Docker Desktop is running" indicator
```

**Linux**:
```bash
sudo systemctl start docker
sudo systemctl enable docker  # Start on boot
```

### Docker "Permission denied" (Linux)

**Problem**: `docker: permission denied while trying to connect to the Docker daemon socket`

**Solution**:

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and back in, or:
newgrp docker

# Verify
docker ps
```

### GDAL Import Error

**Problem**: `ImportError: GDAL not found` or `OSError: libgdal.so: cannot open shared object file`

**Solution**:

**Docker** (recommended): Use the provided Docker image with GDAL pre-installed

**Native installation**:

```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev

# macOS
brew install gdal

# Conda (all platforms)
conda install -c conda-forge gdal

# Verify
python -c "from osgeo import gdal; print(gdal.__version__)"
```

**Environment variables**:

```bash
export GDAL_DATA=$(gdal-config --datadir)
export PROJ_LIB=/usr/share/proj  # Adjust path if needed
```

Or add to `project.yml`:

```yaml
environment:
  GDAL_DATA: /usr/share/gdal
  PROJ_LIB: /usr/share/proj
```

### openAMUNDSEN Import Error

**Problem**: `ModuleNotFoundError: No module named 'openamundsen'`

**Solution**:

openAMUNDSEN is installed automatically via pip dependencies. If missing:

```bash
# Docker
docker compose build --no-cache

# Native
pip install openamundsen
```

If specific version required:

```toml
# pyproject.toml
dependencies = [
    "openamundsen==0.3.0",  # Specify version
]
```

---

## Configuration Issues

### "ROI file not found"

**Problem**: `FileNotFoundError: env/roi.gpkg`

**Solution**:

Verify file path in `project.yml`:

```yaml
domain:
  aoi: env/roi.gpkg  # Relative to project root
```

Check file exists:

```bash
ls -l /data/env/roi.gpkg  # Inside Docker container
ls -l /path/to/project/env/roi.gpkg  # On host
```

If using Docker, verify volume mount in `compose.yml`:

```yaml
volumes:
  - ${PROJ}:/data  # PROJ env var must point to project root
```

### "Invalid CRS"

**Problem**: `CRSError: Invalid CRS: EPSG:XXXXX`

**Solution**:

Use valid EPSG code. Common codes:
- **EPSG:32632**: UTM Zone 32N (Central Europe)
- **EPSG:32633**: UTM Zone 33N (Eastern Europe)
- **EPSG:3857**: Web Mercator
- **EPSG:4326**: WGS84 (lat/lon)

Find your CRS:

```bash
# From shapefile
ogrinfo -al -so env/roi.gpkg | grep PROJCRS

# From coordinates
# Use https://epsg.io/
```

Update `project.yml`:

```yaml
domain:
  crs: EPSG:32632  # Must match ROI CRS or be valid target
```

### "Timestep format invalid"

**Problem**: `ValueError: Invalid timestep format`

**Solution**:

Use pandas-compatible frequency string:

**Valid**:
- `1H`, `3H`, `6H`, `12H`, `24H` (hours)
- `30min`, `15min` (minutes)
- `1D` (day)

**Invalid**:
- `3h` (lowercase)
- `3 hours` (spaces)
- `180min` (use `3H` instead)

```yaml
# project.yml
timestep: 3H  # Correct
```

---

## Runtime Issues

### Ensemble Run Fails Immediately

**Problem**: openAMUNDSEN crashes at start

**Cause**: Configuration error, missing forcing data, or invalid state file

**Debug**:

```bash
# Run single member manually
docker compose run --rm oa \
  python -m openamundsen_da.core.runner \
  --project-dir /data \
  --member-dir /data/propagation/season_2019-2020/step_00_init/ensembles/prior/member_001 \
  --log-level DEBUG
```

Check logs for specific error.

**Common causes**:

1. **Missing forcing data**:
   ```
   FileNotFoundError: meteo/station_001.csv
   ```
   Solution: Verify forcing files exist and match `stations.csv`

2. **Invalid date range**:
   ```
   ValueError: Start date not found in forcing data
   ```
   Solution: Ensure forcing spans `start_date` to `end_date` + buffer

3. **State file mismatch**:
   ```
   ValueError: State file dimensions don't match domain
   ```
   Solution: Delete `state_pointer.json` to start from cold start

### "ESS always near N, no resampling"

**Problem**: Effective sample size stays near ensemble size, weights are uniform

**Cause**: Observations provide no information (observation error too large, or no model-obs mismatch)

**Diagnosis**:

```bash
# Inspect weights
docker compose run --rm oa oa-da-plot-weights \
  /data/propagation/season_2019-2020/step_01_*/assim/weights_scf_*.csv
```

If weights are nearly uniform (all ~1/N), observations aren't constraining the ensemble.

**Solutions**:

1. **Reduce observation error**:
   ```yaml
   data_assimilation:
     observation_error:
       scf: 0.05  # Reduce from 0.10
   ```

2. **Check observation quality**:
   ```bash
   # Verify observations have variability
   cat step_XX/obs/obs_scf_*.csv
   # Should NOT all be 0.0 or 1.0
   ```

3. **Tune H(x) parameters**:
   ```yaml
   data_assimilation:
     h_of_x:
       h0: 0.03  # Lower threshold
       k: 80.0   # Steeper transition
   ```

### "ESS drops to 1 immediately"

**Problem**: Effective sample size becomes 1, one particle dominates

**Cause**: Observation error too small, or severe model-obs mismatch

**Diagnosis**:

```bash
# Check residuals
docker compose run --rm oa oa-da-plot-weights weights.csv
# Look at histogram: if residuals are very large, mismatch is severe
```

**Solutions**:

1. **Increase observation error**:
   ```yaml
   data_assimilation:
     observation_error:
       scf: 0.15  # Increase from 0.10
   ```

2. **Increase ensemble spread**:
   ```yaml
   ensemble:
     prior_forcing:
       sigma_t: 2.0  # Increase from 1.5
       sigma_p: 0.25 # Increase from 0.20
   ```

3. **Check for systematic model bias**:
   - Plot ensemble vs. observations
   - If model is consistently too high/low, bias correction may be needed

### High Memory Usage

**Problem**: System runs out of RAM, processes killed

**Cause**: Too many members, large domain, or insufficient Docker limits

**Solutions**:

1. **Reduce ensemble size**:
   ```yaml
   ensemble:
     size: 20  # Reduce from 50
   ```

2. **Increase Docker memory limit**:
   ```bash
   # .env
   MEMORY=32G  # Increase from 16G
   ```

   Then restart Docker.

3. **Reduce workers**:
   ```bash
   oa-da-season --max-workers 4  # Reduce from 8
   ```

4. **Limit output variables**:
   ```yaml
   output_data:
     grids:
       variables:
         - snow_depth  # Only essential variables
         - snow_water_equivalent
   ```

5. **Reduce output frequency**:
   ```yaml
   output_data:
     grids:
       frequency: 1D  # Instead of hourly
   ```

### Slow Performance

**Problem**: Season run takes very long

**Optimization strategies**:

1. **Increase workers** (if RAM allows):
   ```bash
   oa-da-season --max-workers 12
   ```

2. **Reduce domain resolution**:
   - Resample ROI to coarser grid
   - Use ~100-500m resolution for large domains

3. **Reduce timestep** (if scientifically acceptable):
   ```yaml
   timestep: 6H  # Instead of 3H
   ```

4. **Limit ensemble size**:
   - N=20-30 often sufficient for small/medium domains

5. **Use SSD for data storage**:
   - Move project to SSD drive
   - Docker volumes on SSD

See [Performance Tuning]({{ site.baseurl }}{% link advanced/performance.md %}) for detailed optimization.

---

## Data Assimilation Issues

### "No observations found for assimilation date"

**Problem**: `FileNotFoundError: obs/obs_scf_MOD10A1_YYYYMMDD.csv`

**Cause**: Observation not extracted to step directory, or date mismatch

**Solution**:

1. **Check observation summary**:
   ```bash
   grep YYYYMMDD obs/season_2019-2020/scf_summary.csv
   ```

2. **Re-extract observations**:
   ```bash
   docker compose run --rm oa oa-da-scf \
     --season-dir /data/propagation/season_2019-2020 \
     --summary-csv /data/obs/season_2019-2020/scf_summary.csv \
     --overwrite
   ```

3. **Check date in season.yml**:
   ```yaml
   assimilation_dates:
     - 2019-11-22  # Must exactly match observation date
   ```

### "Resampling indices out of bounds"

**Problem**: `IndexError: index 30 is out of bounds for axis 0 with size 30`

**Cause**: Bug in resampling code, or NaN weights

**Debug**:

```python
import pandas as pd

# Load weights
w = pd.read_csv('assim/weights_scf_YYYYMMDD.csv')

# Check for NaN or negative
print(w['weight'].describe())
print(w['weight'].isna().sum())

# Check normalization
print(w['weight'].sum())  # Should be 1.0
```

**Solution**: If weights are invalid, check:
- Observation file format (should have `scf_mean` column)
- Model SCF files exist and are valid
- No division by zero in likelihood calculation

### Glacier Mask Not Applied

**Problem**: Glacier-covered areas still affect assimilation

**Solution**:

1. **Verify mask enabled**:
   ```yaml
   data_assimilation:
     glacier_mask:
       enabled: true
       path: env/glaciers.gpkg
   ```

2. **Check mask file**:
   ```bash
   ogrinfo -al -so env/glaciers.gpkg
   # Should show valid polygon layer
   ```

3. **Verify CRS match**:
   Glacier mask must be in same CRS as ROI/model domain.

   ```bash
   # Reproject if needed
   ogr2ogr -t_srs EPSG:32632 \
     env/glaciers_reprojected.gpkg \
     env/glaciers.gpkg
   ```

---

## Visualization Issues

### Plots Not Generated

**Problem**: No plots in `plots/` directory

**Cause**: Plotting disabled, or error during plotting

**Solution**:

1. **Check if plotting is disabled**:
   ```bash
   # Season command with --no-live-plots skips plots during run
   # Manually generate plots after:
   docker compose run --rm oa oa-da-plot-scf --season-dir /data/propagation/season_2019-2020
   ```

2. **Check for plotting errors**:
   - Look in logs for matplotlib/plotting errors
   - May be missing display backend (should work headless in Docker)

3. **Manually plot specific results**:
   ```bash
   # SCF time series
   docker compose run --rm oa oa-da-plot-scf \
     --season-dir /data/propagation/season_2019-2020 \
     --project-dir /data

   # ESS timeline
   docker compose run --rm oa oa-da-plot-ess \
     --season-dir /data/propagation/season_2019-2020

   # Weights
   docker compose run --rm oa oa-da-plot-weights \
     step_XX/assim/weights_scf_YYYYMMDD.csv
   ```

### "Plotting backend error"

**Problem**: `RuntimeError: Could not find display`

**Cause**: Matplotlib trying to use GUI backend in headless Docker

**Solution**:

Force non-GUI backend:

```python
# Add to plotting scripts
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
```

Or set environment variable:

```bash
# .env or project.yml
MPLBACKEND=Agg
```

---

## File Corruption Issues

### "NetCDF file corrupted"

**Problem**: `OSError: NetCDF: HDF error` or cannot open state/results files

**Cause**: Incomplete write (e.g., killed process), or disk full

**Solution**:

1. **Delete corrupted file**:
   ```bash
   rm member_001/results/grids/snow.nc
   ```

2. **Re-run step**:
   ```bash
   docker compose run --rm oa oa-da-season \
     --project-dir /data \
     --season-dir /data/propagation/season_2019-2020 \
     --overwrite  # Re-run corrupted step
   ```

3. **Check disk space**:
   ```bash
   df -h  # Ensure sufficient space
   ```

**Prevention**: Use `--monitor-perf` to watch disk usage during run.

---

## Getting Help

If your issue isn't covered here:

1. **Check logs**:
   - Season log: `propagation/season_YYYY-YYYY/season.log`
   - Step logs: `propagation/season_YYYY-YYYY/step_XX_*/step.log`

2. **Enable debug logging**:
   ```bash
   oa-da-season --log-level DEBUG [...]
   ```

3. **Search GitHub Issues**:
   - [github.com/franzwagner-uibk/openamundsen_da/issues](https://github.com/franzwagner-uibk/openamundsen_da/issues)

4. **Report a bug**:
   - Include: OS, Docker version, error message, relevant config
   - Minimal reproducible example preferred

5. **Documentation**:
   - [Installation Guide]({{ site.baseurl }}{% link installation.md %})
   - [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %})
   - [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %})

---

## Known Limitations

1. **Windows paths**: Use forward slashes (`/`) or double backslashes (`\\`) in YAML
2. **Large ensembles**: N > 100 may require >32 GB RAM
3. **Parallel I/O**: NetCDF writing is serial (bottleneck for large grids)
4. **Rejuvenation**: Only supports additive perturbations (no multiplicative state updates)

See GitHub Issues for full list and planned improvements.

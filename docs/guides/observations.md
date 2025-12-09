---
layout: default
title: Observation Processing
parent: Guides
nav_order: 3
---

# Observation Processing
{: .no_toc }

Working with satellite snow observations for data assimilation.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Overview

openamundsen_da supports three types of satellite snow observations:

1. **MODIS MOD10A1** - Daily snow cover at 500m resolution
2. **Sentinel-2 FSC** - Fractional snow cover at 20m resolution (via Snowflake product)
3. **Sentinel-1 Wet Snow** - Wet snow detection at 20m resolution

This guide covers downloading, preprocessing, and quality control for each product.

---

## MODIS MOD10A1 Snow Cover

### Product Overview

**MODIS MOD10A1 v6.1**:
- **Sensor**: Terra/MODIS
- **Resolution**: 500m
- **Temporal**: Daily
- **Coverage**: Global
- **Latency**: 1-2 days

**Key Layers**:
- `NDSI_Snow_Cover`: Normalized Difference Snow Index (0-100)
- `NDSI_Snow_Cover_Basic_QA`: Quality flags
- `NDSI_Snow_Cover_Algorithm_Flags_QA`: Algorithm flags

### Downloading MOD10A1

#### Option 1: NASA Earthdata Search

1. Go to [Earthdata Search](https://search.earthdata.nasa.gov/)
2. Create free account if needed
3. Search for "MOD10A1"
4. Filter by:
   - Version: 061
   - Date range: Your season dates
   - Spatial extent: Draw or upload ROI
5. Select tiles covering your ROI
6. Download HDF files

#### Option 2: Python (earthaccess)

```python
import earthaccess

# Authenticate
earthaccess.login()

# Search for MOD10A1
results = earthaccess.search_data(
    short_name="MOD10A1",
    version="061",
    temporal=("2019-11-01", "2020-07-31"),
    bounding_box=(-10.5, 46.5, -9.5, 47.5)  # (W, S, E, N)
)

# Download
earthaccess.download(results, local_path="./MOD10A1_61_HDF")
```

#### Option 3: AppEEARS

For large areas or long periods, use [AppEEARS](https://appeears.earthdatacloud.nasa.gov/):
1. Submit area extraction request
2. Select MOD10A1 layers
3. Choose output format (HDF or GeoTIFF)
4. Download when ready (email notification)

### Preprocessing MOD10A1

The framework provides automated preprocessing:

```bash
docker compose run --rm oa oa-da-mod10a1 \
  --input-dir /data/obs/MOD10A1_61_HDF \
  --season-label season_2019-2020 \
  --project-dir /data \
  --target-epsg 32632 \
  --resolution 500 \
  --ndsi-threshold 0.4
```

**Processing steps**:
1. **HDF → GeoTIFF**: Extract NDSI_Snow_Cover layer
2. **QA masking**: Remove cloudy/poor-quality pixels
3. **Reprojection**: Reproject to study area CRS
4. **ROI clipping**: Extract ROI extent
5. **Binary masking**: NDSI ≥ 0.4 → snow, else no snow
6. **SCF calculation**: Mean snow fraction per ROI

**Output**:
- `obs/season_2019-2020/NDSI_Snow_Cover_YYYYMMDD.tif` (per date)
- `obs/season_2019-2020/scf_summary.csv`

### Quality Control

Inspect the summary CSV:

```bash
head -20 obs/season_2019-2020/scf_summary.csv
```

**Quality indicators**:

| Field | Description | Quality Threshold |
|:------|:------------|:------------------|
| `scf_mean` | Mean SCF over ROI | Use if > 0.0 |
| `scf_std` | SCF standard deviation | Check for spatial variability |
| `pixel_count` | Valid (cloud-free) pixels | Use if > 50% of ROI |
| `cloud_fraction` | Cloud cover (if available) | Use if < 0.3 |

**Filter low-quality dates**:

```python
import pandas as pd

# Load summary
df = pd.read_csv('obs/season_2019-2020/scf_summary.csv', parse_dates=['date'])

# Filter: valid pixels > 80% of ROI area
roi_area = 12500  # From ROI shapefile
df_clean = df[df['pixel_count'] > 0.8 * roi_area]

# Save filtered dates
df_clean['date'].dt.strftime('%Y-%m-%d').to_csv('assimilation_dates.txt', index=False, header=False)
```

### NDSI Threshold Selection

The NDSI threshold (default: 0.4) defines "snow" vs. "no snow":

```
NDSI = (Green - SWIR) / (Green + SWIR)
```

**Common thresholds**:
- **0.4** (standard): Conservative, reduces commission errors
- **0.1-0.3**: More sensitive, captures patchy snow
- **0.5+**: Very conservative, only certain snow

**Testing thresholds**:

```bash
# Generate summaries with different thresholds
for thresh in 0.3 0.4 0.5; do
  docker compose run --rm oa oa-da-mod10a1 \
    --input-dir /data/obs/MOD10A1_61_HDF \
    --season-label season_2019-2020_ndsi${thresh} \
    --project-dir /data \
    --ndsi-threshold $thresh
done

# Compare
head obs/season_2019-2020_ndsi*/scf_summary.csv
```

---

## Sentinel-2 FSC (Snowflake)

### Product Overview

**Sentinel-2 FSC (via Snowflake)**:
- **Sensor**: Sentinel-2 MSI
- **Resolution**: 20m
- **Temporal**: 5-day revisit (cloud-dependent)
- **Coverage**: Europe
- **Latency**: Near real-time

**Output**: Fractional Snow Cover (0-100%)

### Downloading Sentinel-2 FSC

#### Snowflake Product

The [Snowflake project](https://www.snowflake-project.eu/) provides Sentinel-2 FSC for the Alps:

1. Go to [Snowflake Data Portal](https://data.snowflake-project.eu/)
2. Select region and date range
3. Download GeoTIFF files

**Filename format**: `FSC_YYYYMMDD_Txxxx_*.tif`

#### Theia Snow Collection

Alternative source: [Theia Snow Collection](https://www.theia-land.fr/)

```bash
# Download using Theia downloader
python theia_download.py \
  --collection SENTINEL2 \
  --level LEVEL2A \
  --start 2019-11-01 \
  --end 2020-07-31 \
  --tile 32TMS \
  --product SEB
```

### Preprocessing Sentinel-2 FSC

```bash
docker compose run --rm oa \
  python -m openamundsen_da.observer.snowflake_fsc \
  --input-dir /data/obs/FSC_snowflake \
  --season-label season_2019-2020 \
  --project-dir /data \
  --resolution 100 \
  --fsc-threshold 10
```

**Parameters**:
- `--resolution`: Resample to coarser resolution (optional, default: keep 20m)
- `--fsc-threshold`: Minimum FSC to consider as snow (%, default: 10)

**Output**: Same as MOD10A1 (GeoTIFFs + summary CSV)

### Quality Control

Sentinel-2 FSC includes quality layers:

**Quality flags** (if available in product):
- 0: High quality
- 1: Medium quality
- 2: Low quality (cloud shadow, topographic shadow)
- 3: Invalid (cloud, outside mask)

**Filter**:

```bash
# Keep only high/medium quality pixels during processing
docker compose run --rm oa \
  python -m openamundsen_da.observer.snowflake_fsc \
  --input-dir /data/obs/FSC_snowflake \
  --season-label season_2019-2020 \
  --project-dir /data \
  --quality-threshold 1  # 0-1: high/medium
```

### MOD10A1 vs. Sentinel-2 FSC

| Aspect | MOD10A1 | Sentinel-2 FSC |
|:-------|:--------|:---------------|
| **Resolution** | 500m | 20m |
| **Temporal** | Daily | 5 days |
| **Cloud gaps** | Frequent | More frequent |
| **Accuracy** | Binary (0/1) | Fractional (0-1) |
| **Coverage** | Global | Regional |
| **Best for** | Large domains, daily DA | Small domains, high detail |

**Recommendation**: Use MOD10A1 for primary assimilation, Sentinel-2 FSC for validation or targeted periods.

---

## Sentinel-1 Wet Snow

### Product Overview

**Sentinel-1 Wet Snow Mask**:
- **Sensor**: Sentinel-1 SAR (C-band)
- **Resolution**: 20-30m
- **Temporal**: 6-12 day revisit
- **Coverage**: Global
- **Latency**: 1-3 days

**Detection**: Wet snow reduces backscatter significantly → detectable via change detection.

### Downloading Sentinel-1

#### Option 1: Copernicus Data Space

1. Go to [Copernicus Data Space](https://dataspace.copernicus.eu/)
2. Search for Sentinel-1 GRD products
3. Filter by:
   - Polarization: VV or VH
   - Pass direction: Ascending or Descending
   - Date range
4. Download

#### Option 2: Google Earth Engine

```python
import ee
ee.Initialize()

# Define ROI
roi = ee.Geometry.Rectangle([10.0, 46.5, 11.0, 47.5])

# Get Sentinel-1 GRD collection
s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
  .filterBounds(roi) \
  .filterDate('2019-11-01', '2020-07-31') \
  .filter(ee.Filter.eq('instrumentMode', 'IW')) \
  .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING')) \
  .select('VH')

# Export
for img in s1.toList(s1.size()).getInfo():
    # Export each image
    pass
```

### Wet Snow Detection

The framework expects **pre-processed wet snow masks** (not raw SAR):

**WSM Classes**:
- **110**: Wet snow
- **125**: Dry snow or no snow
- **200**: Radar shadow (masked out)
- **210**: Water (masked out)

**Generate WSM** (external processing required):

Use tools like:
- [SNAP](https://step.esa.int/main/download/snap-download/) (ESA Sentinel Application Platform)
- [SWI algorithm](https://www.mdpi.com/2072-4292/12/12/1959) (Nagler et al., 2016)
- Custom change detection scripts

**Example SNAP workflow**:
1. Apply orbit file
2. Thermal noise removal
3. Calibration (σ0)
4. Terrain correction
5. Change detection (reference vs. current)
6. Threshold: ΔσVH < -3 dB → wet snow

### Preprocessing Sentinel-1 WSM

```bash
docker compose run --rm oa \
  python -m openamundsen_da.observer.satellite_wet_snow_s1 \
  --input-dir /data/obs/WSM_S1 \
  --season-label season_2019-2020 \
  --project-dir /data
```

**Input**: GeoTIFFs with wet snow mask (class 110 = wet snow)

**Output**:
- Reprojected/clipped GeoTIFFs
- `obs/season_2019-2020/wet_snow_summary.csv`

### Quality Control

Sentinel-1 has limitations:

**Exclusions**:
- **Radar shadow** (class 200): No signal, exclude
- **Steep slopes** (> 30°): Layover/foreshortening, unreliable
- **Forested areas**: Volume scattering, reduced sensitivity

**Filter by terrain**:

```python
import rasterio
import numpy as np

# Load DEM slope
with rasterio.open('path/to/slope.tif') as src:
    slope = src.read(1)

# Load WSM
with rasterio.open('obs/season_2019-2020/WSM_S1_20200415.tif') as src:
    wsm = src.read(1)

# Mask steep slopes
wsm[(slope > 30) | (wsm == 200) | (wsm == 210)] = 255  # NoData

# Write masked WSM
with rasterio.open('obs/season_2019-2020/WSM_S1_20200415_masked.tif', 'w', **src.meta) as dst:
    dst.write(wsm, 1)
```

---

## Multi-Sensor Fusion

### Combining MOD10A1 and Sentinel-2

Use MOD10A1 for daily coverage, Sentinel-2 for cloud-free high-resolution:

```yaml
# season.yml
assimilation_events:
  - date: 2019-11-22
    type: scf
    product: MOD10A1
    weight: 1.0

  - date: 2019-11-25
    type: scf
    product: SNOWFLAKE
    weight: 1.5  # Higher weight for higher resolution
```

**Implementation** (custom):

```python
# In assimilation loop
if event['product'] == 'SNOWFLAKE':
    obs_error = 0.05  # Lower error for S2
else:
    obs_error = 0.10  # Standard for MODIS
```

### Combining SCF and Wet Snow

Assimilate both snow cover and wet snow state:

```yaml
assimilation_events:
  - date: 2020-03-15
    type: scf
    product: MOD10A1

  - date: 2020-04-12
    type: wet_snow
    product: S1
```

The framework handles both observation types:

```bash
# Automatic detection based on obs file prefix
# obs_scf_*.csv → SCF assimilation
# obs_wet_snow_*.csv → Wet snow assimilation
```

---

## Observation Operators (H(x))

### SCF Forward Operator

Maps model snow depth/SWE to snow cover fraction:

**Depth Threshold**:
```
H(x) = 1  if HS > h0
       0  otherwise
```

**Logistic** (recommended):
```
H(x) = 1 / (1 + exp(-k × (HS - h0)))
```

See [Configuration → H(x)]({{ site.baseurl }}{% link guides/configuration.md %}#hx-forward-operator-methods) for details.

### Wet Snow Forward Operator

Maps model liquid water content (LWC) to wet/dry classification:

```
Wet snow = LWC > threshold (e.g., 1-3% of SWE)
```

**Configuration**:

```yaml
data_assimilation:
  wet_snow:
    lwc_threshold: 0.02  # 2% of SWE
    min_depth: 0.01      # Minimum snow depth (m)
```

---

## Best Practices

### Observation Thinning

**Too many observations** → computational cost, redundancy

**Strategies**:
1. **Temporal thinning**: Every 7-10 days instead of daily
2. **Spatial thinning**: Aggregate to coarser resolution
3. **Quality filtering**: High-quality obs only

**Example**:

```python
import pandas as pd

df = pd.read_csv('obs/season_2019-2020/scf_summary.csv', parse_dates=['date'])

# Keep every 7 days with high quality
df_thin = df.resample('7D', on='date').first()
df_thin = df_thin[df_thin['pixel_count'] > 0.8 * roi_area]
```

### Observation Error Tuning

**Too small** → particle degeneracy (ESS → 1)
**Too large** → no weight update (ESS → N)

**Starting values**:
- MOD10A1 SCF: σ_obs = 0.10-0.15
- Sentinel-2 FSC: σ_obs = 0.05-0.10
- Sentinel-1 Wet Snow: σ_obs = 0.15-0.20

**Tuning approach**:

1. Run DA with default values
2. Inspect ESS timeline
3. Adjust:
   - ESS consistently near N → reduce σ_obs
   - ESS drops to 1 frequently → increase σ_obs

### Glacier Masking

**Why mask glaciers?**

Seasonal snow models (like openAMUNDSEN) simulate seasonal snow only. Satellite observations include firn/ice on glaciers. Assimilating glacier observations into a seasonal model causes mismatch.

**Enable glacier masking**:

```yaml
data_assimilation:
  glacier_mask:
    enabled: true
    path: env/glaciers.gpkg
```

Glacier-covered pixels are excluded from:
- H(x) computation
- Likelihood calculation
- SCF mean/statistics

---

## Troubleshooting

### Issue: All SCF values are 0 or 1

**Cause**: NDSI threshold too high/low, or binary H(x)

**Solution**:
- Test different NDSI thresholds (0.3-0.5)
- Use `logistic` H(x) instead of `depth_threshold`

### Issue: No observations found for some dates

**Cause**: Cloud cover, or preprocessing failed

**Solution**:
- Check raw HDF/GeoTIFF files for those dates
- Inspect preprocessing log for errors
- Accept that some dates have no observations (common for optical sensors)

### Issue: Wet snow observations have poor quality

**Cause**: Radar shadow, steep slopes, forest

**Solution**:
- Mask steep slopes (> 30°)
- Exclude forested areas
- Use only high-confidence wet snow pixels

### Issue: Model-obs mismatch is large

**Cause**: H(x) parameters, timing mismatch, or model bias

**Solution**:
- Check observation time of day vs. model output time
- Tune H(x) parameters (`h0`, `k`)
- Verify glacier masking is enabled
- Inspect ensemble spread (may need larger σ_T, σ_P)

---

## Next Steps

- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) - Configure observation errors and H(x)
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments.md %}) - End-to-end workflow
- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) - Observation preprocessing commands

---

## References

### MODIS MOD10A1
- Hall, D.K., Riggs, G.A. (2016). *MODIS/Terra Snow Cover Daily L3 Global 500m Grid, Version 6*. NASA NSIDC DAAC. [https://doi.org/10.5067/MODIS/MOD10A1.006](https://doi.org/10.5067/MODIS/MOD10A1.006)

### Sentinel-2 FSC
- Gascoin, S., et al. (2019). *Theia Snow collection: high-resolution operational snow cover maps from Sentinel-2 and Landsat-8 data*. Earth System Science Data, 11(2), 493-514. [https://doi.org/10.5194/essd-11-493-2019](https://doi.org/10.5194/essd-11-493-2019)

### Sentinel-1 Wet Snow
- Nagler, T., et al. (2016). *Retrieval of wet snow by means of multitemporal SAR data*. Remote Sensing, 8(9), 754. [https://doi.org/10.3390/rs8090754](https://doi.org/10.3390/rs8090754)

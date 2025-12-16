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

{: .note }
> MOD10A1 data must be obtained from NASA Earthdata. The framework expects HDF files as input for preprocessing.

### Preprocessing MOD10A1

The framework provides automated preprocessing:

```bash
docker compose run --rm oa oa-da-mod10a1 \
  --input-dir /data/obs/MOD10A1_61_HDF \
  --season-label season_2019-2020 \
  --project-dir /data \
  --target-epsg 32632 \
  --resolution 500 \
  --ndsi-threshold 40
```

**Processing steps**:
1. **HDF → GeoTIFF**: Extract NDSI_Snow_Cover layer
2. **QA masking**: Remove cloudy/poor-quality pixels
3. **Reprojection**: Reproject to study area CRS
4. **ROI clipping**: Extract ROI extent
5. **Binary masking**: NDSI ≥ 40 → snow, else no snow
6. **SCF calculation**: Mean snow fraction per ROI

**Output**:
- `obs/season_2019-2020/NDSI_Snow_Cover_YYYYMMDD.tif` (per date)
- `obs/season_2019-2020/scf_summary.csv`

### Creating per-step observation CSVs (for assimilation)

After `scf_summary.csv` is created, generate per-step one-row observation CSVs (the season pipeline expects these under each step's `obs/` directory):

```bash
docker compose run --rm oa oa-da-scf \\
  --season-dir /data/propagation/season_2019-2020 \\
  --summary-csv /data/obs/season_2019-2020/scf_summary.csv \\
  --product MOD10A1 \\
  --overwrite
```


### Quality Control

Inspect the summary CSV:

```bash
head -20 obs/season_2019-2020/scf_summary.csv
```

`scf_summary.csv` contains (per date): `date`, `region_id`, `n_valid`, `n_snow`, `scf`, `cloud_fraction`, `source`.

A simple filter example:

```python
import pandas as pd

df = pd.read_csv('obs/season_2019-2020/scf_summary.csv', parse_dates=['date'])

# Example: keep dates with at least some valid pixels and limited cloud cover
df_clean = df[(df['n_valid'] > 0) & (df['cloud_fraction'] <= 0.3)]

df_clean['date'].dt.strftime('%Y-%m-%d').to_csv('assimilation_dates.txt', index=False, header=False)
```

### NDSI Threshold Selection

The MOD10A1 `NDSI_Snow_Cover` layer uses values in the range **0..100**. A threshold of **40** corresponds to an NDSI of **0.40**.

**Common thresholds (MOD10A1 band units)**:
- **30**: More sensitive (captures patchy snow, may increase false positives)
- **40** (default): Typical starting point
- **50+**: Conservative (reduces commission errors)

**Testing thresholds**:

```bash
# Generate summaries with different thresholds
for thresh in 30 40 50; do
  docker compose run --rm oa oa-da-mod10a1 \\
    --input-dir /data/obs/MOD10A1_61_HDF \\
    --season-label season_2019-2020_ndsi${thresh} \\
    --project-dir /data \\
    --ndsi-threshold $thresh

done

# Compare
head obs/season_2019-2020_ndsi*/scf_summary.csv
```

---

## Sentinel-2 FSC (Snowflake)

### Product Overview

**Sentinel-2 FSC (Snowflake)**:
- **Sensor**: Sentinel-2 MSI
- **Resolution**: Product-dependent (often 20m)
- **Temporal**: ~5-day revisit (cloud-dependent)

**Input**: GeoTIFF FSC rasters with values in **0..100 (%)** (NoData for invalid/cloud pixels).

### Summarizing to `scf_summary.csv`

The framework summarizes each FSC raster to a single ROI-mean `scf` value and appends/updates it in `obs/<season-label>/scf_summary.csv`:

```bash
docker compose run --rm oa \\
  python -m openamundsen_da.observer.snowflake_fsc \\
  --input-dir /data/obs/FSC_snowflake \\
  --season-label season_2019-2020 \\
  --project-dir /data
```

**Notes**:
- The ROI is auto-detected from `/data/env/roi.gpkg` unless you pass `--aoi`.
- The acquisition date is parsed from the filename as `YYYY_MM_DD` (e.g. `..._2019_10_01.tif`).
- Use `--recursive` if your rasters are in subfolders.

### Creating per-step observation CSVs (for assimilation)

After `scf_summary.csv` exists, generate per-step one-row observation CSVs:

```bash
docker compose run --rm oa oa-da-scf \\
  --season-dir /data/propagation/season_2019-2020 \\
  --summary-csv /data/obs/season_2019-2020/scf_summary.csv \\
  --product SNOWFLAKE \\
  --overwrite
```

---

## Sentinel-1 Wet Snow

### Product Overview

**Sentinel-1 Wet Snow Mask (WSM)**:
- **Sensor**: Sentinel-1 SAR (C-band)
- **Resolution**: product-dependent (often 20-30m)
- **Temporal**: ~6-12 day revisit

{: .note }
> The framework expects **pre-processed wet-snow mask rasters** (not raw SAR).

**WSM Classes** (expected by the summarizer):
- **110**: Wet snow
- **125**: Dry snow or no snow
- **200**: Radar shadow (excluded)
- **210**: Water (excluded)

### Summarizing WSM to `wet_snow_summary.csv`

First summarize Sentinel-1 WSM rasters into a season table:

```bash
docker compose run --rm oa oa-da-wet-snow-s1 \\
  --project-dir /data \\
  --output /data/obs/season_2019-2020/wet_snow_summary.csv
```

With `--project-dir`, the command uses these defaults:
- WSM rasters: `/data/obs/WSM_S1_SAR`
- ROI: `/data/env/roi.gpkg`

Override with `--raster-dir` / `--aoi` if your paths differ.

### Creating per-step observation CSVs (for assimilation)

```bash
docker compose run --rm oa oa-da-wet-snow-s1-season \\
  --season-dir /data/propagation/season_2019-2020 \\
  --summary-csv /data/obs/season_2019-2020/wet_snow_summary.csv \\
  --overwrite
```

This writes one-row `obs_wet_snow_S1_YYYYMMDD.csv` files into each step's `obs/` directory for configured wet-snow assimilation dates.

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
df_clean = df[df['n_valid'] > 0.8 * df['n_valid'].max()]
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

### Wet Snow Dynamics and Remote Sensing

- Rottler, E., Warscher, M., Hanzer, F., and Strasser, U.: Spatio-temporal wet snow dynamics from model simulations and remote sensing: A case study from the Rofental, Austria, Hydrological Processes, 38, e15279, [https://doi.org/10.1002/hyp.15279](https://doi.org/10.1002/hyp.15279), 2024.

- Cluzet, B., Magnusson, J., Quéno, L., Mazzotti, G., Mott, R., and Jonas, T.: Exploring how Sentinel-1 wet-snow maps can inform fully distributed physically based snowpack models, The Cryosphere, 18, 5753–5767, [https://doi.org/10.5194/tc-18-5753-2024](https://doi.org/10.5194/tc-18-5753-2024), 2024.

### Snow Cover Data Assimilation

- Baba, M. W., Gascoin, S., and Hanich, L.: Assimilation of Sentinel-2 Data into a Snowpack Model in the High Atlas of Morocco, Remote Sensing, 10, 1982, [https://doi.org/10.3390/rs10121982](https://doi.org/10.3390/rs10121982), 2018.


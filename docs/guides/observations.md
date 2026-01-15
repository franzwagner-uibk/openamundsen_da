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

Use `obs/<season-label>/scf_summary.csv` for quality control and to decide which dates to assimilate (set `assimilation_dates` in `propagation/<season-label>/season.yml`).

`scf_summary.csv` contains (per date): `date`, `region_id`, `n_valid`, `n_snow`, `scf`, `cloud_fraction`, `source`. Typical filters include a minimum `n_valid` and a maximum `cloud_fraction`.

### NDSI Threshold Selection

The MOD10A1 `NDSI_Snow_Cover` layer uses values in the range **0..100**. A threshold of **40** corresponds to an NDSI of **0.40**.

**Common thresholds (MOD10A1 band units)**:

- **30**: More sensitive (captures patchy snow, may increase false positives)
- **40** (default): Typical starting point
- **50+**: Conservative (reduces commission errors)

To test thresholds, rerun `oa-da-mod10a1` with different `--ndsi-threshold` values and compare the resulting `scf_summary.csv` outputs.

---

## Sentinel-2 FSC (SnowFLAKES)

### Product Overview

**Sentinel-2 FSC (SnowFLAKES)** (Barella et al., 2022):

- **Sensor**: Sentinel-2 MSI
- **Resolution**: Product-dependent (often 20m)
- **Temporal**: ~5-day revisit (cloud-dependent)

**Input**: GeoTIFF or NetCDF FSC rasters with values in **0..100 (%)**. Class handling:

- 0..100 = valid FSC (percent)
- 205 = clouds (excluded; counted in `cloud_fraction`)
- 210 = water (excluded)
- 255 or `_FillValue` = nodata (excluded)

This guide assumes the SnowFLAKES FSC product (Barella et al., 2022).

### Summarizing to `scf_summary.csv`

The framework summarizes each FSC raster to a single ROI-mean `scf` value and appends/updates it in `obs/<season-label>/scf_summary.csv`:

```bash
docker compose run --rm oa \\
  oa-da-snowflakes-fsc \\
  --input-dir /data/obs/FSC_snowflake* \\
  --season-label season_2019-2020 \\
  --project-dir /data
```

**Notes**:

- The ROI is auto-detected from `/data/env/roi.gpkg` unless you pass `--aoi`; land-cover exclusions use `grids/lc_<domain>_<resolution>.asc` and `data_assimilation.landcover_mask.classes_to_exclude` (defaults: 2 ice, 8-12 forests/mixed, 13 built-up). Keep exactly one matching LC file per domain/resolution. A warning is logged if >50% of the ROI is excluded; 100% exclusion fails.
- The acquisition date is parsed from the filename as `YYYY_MM_DD` or `YYYYMMDD` (e.g. `SnowFLAKES_20191001_v0_*.nc`).
- Use `--recursive` if your rasters are in subfolders.
- Outputs include `cloud_fraction` along with `n_valid`, `n_snow`, and `scf`.

### Creating per-step observation CSVs (for assimilation)

After `scf_summary.csv` exists, generate per-step one-row observation CSVs:

```bash
docker compose run --rm oa oa-da-scf \\
  --season-dir /data/propagation/season_2019-2020 \\
  --summary-csv /data/obs/season_2019-2020/scf_summary.csv \\
  --product SNOWFLAKES \\
  --overwrite
```

---

## Sentinel-1 Wet Snow

### Product Overview

**Sentinel-1 Wet Snow Mask (WSM)** (Nagler et al., 2016):

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
    classification_threshold_percent: 0.5 # LWC fraction threshold (0-1 scale)
```

The classification threshold is a volumetric LWC fraction. A value of 0.5 means snow is classified as "wet" when LWC exceeds 50% of the maximum possible LWC.

---

## Best Practices

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

### Land-Cover Masking

**Why mask certain classes?**

Satellite products often miss snow below dense canopy or in built-up areas, while the model can still simulate it. Ice/glacier classes should also be excluded to avoid mismatches between seasonal snow and firn/ice signals.

**Configure land-cover masking**:

```yaml
data_assimilation:
  landcover_mask:
    # Classes: 1 rock, 2 ice, 3 water, 4 grassland, 5 shrubland, 6 farmland,
    # 7 transitional, 8 deciduous 30-60, 9 deciduous 60-100, 10 mixed,
    # 11 coniferous 30-60, 12 coniferous 60-100, 13 built-up.
    enabled: true
    classes_to_exclude: [2, 8, 9, 10, 11, 12, 13]
```

- Land-cover grid is resolved as `grids/lc_<domain>_<resolution>.asc`.
- Excluded classes are removed from both obs-side summaries and model H(x); a warning is logged if >50% of the ROI would be excluded, and masking fails if 100% would be removed.

---



## Next Steps

- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) - Configure observation errors and H(x)
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments.md %}) - End-to-end workflow
- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) - Observation preprocessing commands

---

## References

- Barella, R., Marin, C., Gianinetto, M., and Notarnicola, C. (2022). A novel approach to high resolution snow cover fraction retrieval in mountainous regions. IGARSS 2022 - IEEE International Geoscience and Remote Sensing Symposium, 3856-3859. https://doi.org/10.1109/IGARSS46834.2022.9884177.
- Nagler, T., Rott, H., Ripper, E., Bippus, G., and Hetzenecker, M. (2016). Advancements for snowmelt monitoring by means of Sentinel-1 SAR. Remote Sensing, 8(4), 348. https://doi.org/10.3390/rs8040348.

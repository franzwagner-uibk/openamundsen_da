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

1. **Snow-cover** - Daily snow cover at 500m resolution
2. **Sentinel-2 FSC** - Fractional snow cover at 20m resolution (via Snowflake product)
3. **Sentinel-1 Wet Snow** - Wet snow detection at 20m resolution

This guide covers downloading, preprocessing, and quality control for each product.

Uncertainty handling overview:

- You can run without uncertainty layers (`data_assimilation.uncertainty.<product>.enabled: false`) and keep sigma from the formula mode.
- You can ingest externally provided uncertainty layers (`enabled: true`) from NetCDF same-file variables or GeoTIFF sidecars (`<stem>_uncertainty.tif`).
- You can generate sidecar uncertainty rasters with openAMUNDSEN-DA preprocessors and then ingest them in the same way.

In all uncertainty-enabled paths, values are expected in `0..100`, and preprocessing is fail-fast on missing/invalid uncertainty inputs.

---

## Snow-cover Snow Cover

### Product Overview

Snow-cover rasters (GeoTIFF/NetCDF) encoded as 0..100% with configurable cloud/water/nodata classes (see `project.yml` under `obs.snowcover.classes`).

### Creating per-step observation CSVs (for assimilation)

After `scf_summary.csv` is created, generate per-step one-row observation CSVs (the project pipeline expects these under each step's `obs/` directory):

```bash
docker compose run --rm oa oa-da-scf \\
  --project-dir /data/projects/project_2019-2020 \\
  --summary-csv /data/obs/project_2019-2020/scf_summary.csv \\
  --product SNOWCOVER \\
  --overwrite
```

### Quality Control

Use `obs/<project-label>/scf_summary.csv` for quality control and to decide which dates to assimilate (set `data_assimilation.assimilation_events` in `projects/<project-label>/<project-label>.yml`).

`scf_summary.csv` contains (per date): `date`, `region_id`, `n_valid`, `n_snow`, `n_invalid`, `scf`, `cloud_fraction`, `invalid_fraction`, `source`. Typical filters include a minimum `n_valid` and a maximum `cloud_fraction` and/or `invalid_fraction`.
`invalid_fraction` is the ROI-based unusable-scene fraction and captures missing or otherwise unusable ROI pixels even when they are not encoded as explicit cloud classes.
When SCF uncertainty is enabled, additional columns are included: `unc_mean`, `unc_min`, `unc_max`, `unc_n_valid`.

## Snow-cover rasters

### Class overview

- 0..100 = valid FSC (%)
- Clouds, water, nodata classes are configurable under `obs.snowcover.classes` (defaults: cloud 205, water 210, nodata 255).

### Summarizing to `scf_summary.csv`

```bash
docker compose run --rm oa \
  oa-da-snowcover \
  --input-dir /data/obs/snowcover \
  --project-label project_2019-2020 \
  --setup-dir /data
```

Notes:

- ROI defaults to a vector auto-resolved under `/data/env` (or generated from `grids/roi_<domain>_<resolution>.asc` if no vector exists); land-cover masking uses `grids/lc_<domain>_<resolution>.asc` and `data_assimilation.landcover_mask.classes_to_exclude`.
- Acquisition date is parsed from tokens like `YYYY_MM_DD` or `YYYYMMDD`.
- Supports `.tif/.tiff/.nc`; use `--recursive` for nested folders.
- NetCDF SCF products may provide CRS metadata through a CF `crs` variable or a `spatial_ref` variable with `crs_wkt`/`spatial_ref` attributes. This supports products such as EURAC SnowFLAKES v3 where GDAL subdatasets do not expose CRS directly. NetCDF products still require explicit `data_assimilation.uncertainty.scf.ingest.scf_variable`, `uncertainty_variable` and `time_variable` names when uncertainty ingest is enabled.

### Creating per-step observation CSVs (for assimilation)

```bash
docker compose run --rm oa oa-da-scf \
  --project-dir /data/projects/project_2019-2020 \
  --summary-csv /data/obs/project_2019-2020/scf_summary.csv \
  --overwrite
```

Product tags are resolved from `project.yml` (`obs.snowcover.product_tag`) and must be configured.
Per-step preparation is fail-fast: events and step windows must match, and each configured event date must exist in the summary CSV.

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

Summarize wet-snow rasters into a setup table:

```bash
docker compose run --rm oa oa-da-wetsnow \
  --input-dir /data/obs/wetsnow \
  --setup-dir /data \
  --output-root /data/obs \
  --project-label project_2019-2020
```

Notes:

- `--input-dir` is required and should point to your wet-snow raster directory.
- ROI: auto-resolved under `/data/env` (or generated from `grids/roi_<domain>_<resolution>.asc`)
- Land-cover mask: `/data/grids/lc_<domain>_<resolution>.asc`
- When wet-snow uncertainty is enabled, `wet_snow_summary.csv` additionally contains `unc_mean`, `unc_min`, `unc_max`, `unc_n_valid`.

### Optional wet-snow uncertainty generation

```bash
docker compose run --rm oa oa-da-wetsnow-uncertainty \
  --setup-dir /data \
  --project-label project_2019-2020 \
  --overwrite
```

This creates `*_uncertainty.tif` companions next to each source wet-snow raster.

### Creating per-step observation CSVs (for assimilation)

```bash
docker compose run --rm oa oa-da-wetsnow-project \\
  --project-dir /data/projects/project_2019-2020 \\
  --summary-csv /data/obs/project_2019-2020/wet_snow_summary.csv \\
  --overwrite
```

This writes one-row `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv` files into each step's `obs/` directory for configured wet-snow assimilation dates. Product tags come from `project.yml` (`obs.wetsnow.product_tag`) and must be configured.
Per-step preparation is fail-fast: events and step windows must match, and each configured event date must exist in the summary CSV.

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
H(x) = 1 / (1 + exp(-k * (HS - h0)))
```

See [Configuration -> H(x)]({{ site.baseurl }}{% link guides/configuration.md %}#hx-forward-operator-methods) for details.

### Wet Snow Forward Operator

Maps model liquid water content (LWC) to wet/dry classification:

```
Wet snow = LWC > threshold (e.g., 1-3% of SWE)
```

**Configuration**:

```yaml
data_assimilation:
  wet_snow:
    classification_method: liquid_water_fraction
    classification_threshold_percent: 0.5 # threshold in percent (0.5 = 0.5%)
```

The classification threshold is interpreted in percent and converted internally to a fraction (`percent / 100`).
Alternatively, `classification_method: liquid_water_amount` classifies model cells by summed snowpack liquid water in mm using `liquid_water_amount_threshold_mm` (default `5.0`).

---

## Best Practices

### Observation Error Tuning

**Too small** -> particle degeneracy (ESS -> 1)
**Too large** -> no weight update (ESS -> N)

**Starting values**:

- SNOWCOVER SCF: Ïƒ_obs = 0.10-0.15
- Sentinel-2 FSC: Ïƒ_obs = 0.05-0.10
- Sentinel-1 Wet Snow: Ïƒ_obs = 0.15-0.20

**Tuning approach**:

1. Run data assimilation with default values
2. Inspect ESS timeline
3. Adjust:
   - ESS consistently near N -> reduce sigma_obs
   - ESS drops to 1 frequently -> increase sigma_obs

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
- Best-practice split: keep exclusion for truly unusable classes (for example ice/water/urban), and represent usable-but-uncertain classes (for example forest/shadow) via uncertainty penalties.
- For data assimilation consistency, treat clouds as data gaps (masked) rather than 100%-penalty uncertainty pixels.

---



## Next Steps

- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) - Configure observation errors and H(x)
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments/index.md %}) - End-to-end workflow
- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) - Observation preprocessing commands

---

## References

- Barella, R., Marin, C., Gianinetto, M., and Notarnicola, C. (2022). A novel approach to high resolution snow cover fraction retrieval in mountainous regions. IGARSS 2022 - IEEE International Geoscience and Remote Sensing Symposium, 3856-3859. https://doi.org/10.1109/IGARSS46834.2022.9884177.
- Nagler, T., Rott, H., Ripper, E., Bippus, G., and Hetzenecker, M. (2016). Advancements for snowmelt monitoring by means of Sentinel-1 SAR. Remote Sensing, 8(4), 348. https://doi.org/10.3390/rs8040348.

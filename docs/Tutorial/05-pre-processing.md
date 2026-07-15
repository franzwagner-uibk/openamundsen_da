---
layout: default
title: 4. Preprocess Observations
parent: How to Use
nav_order: 4
permalink: /tutorial/pre-processing/
---

# 4. Preprocess Observations

This chapter converts the bundled snow-cover and wet-snow raster products into
the project summary tables used during preparation. The Rofental example already
contains the frozen baseline summaries, so you can either inspect them and move
on or reproduce them from the configured products.

From this point on, the tutorial assumes that the container shell is open at
`/data/rofental` and uses this project directory:

```bash
PROJECT_DIR=/data/rofental/projects/project_2022_2023
```

## Inputs and configuration

The example provides `obs/snowcover/`, `obs/wetsnow/` and `obs/stations/`.
Baseline fractional snow covered area (fSCA) and wet-snow summaries live under
`obs/summaries/project_2022_2023/`. Station observations and
`stations_da_metadata.csv` are read during project preparation and do not require
a raster-summary command.

All observation paths, formats, product tags, class mappings, summary paths and
uncertainty settings come from
`projects/project_2022_2023/project_2022_2023.yml`. The commands do not guess a
product or rewrite the YAML.

```yaml
obs:
  stations:
    dir: obs/stations
  snowcover:
    dir: obs/snowcover
    format: geotiff
    product_tag: SNOWCOVER
    summary_csv: obs/summaries/project_2022_2023/scf_summary.csv
    classes:
      valid: [0, 1, 2, 3, 4, 5]
      cloud: [205]
      water: [210]
      nodata: [255]
  wetsnow:
    dir: obs/wetsnow
    format: geotiff
    product_tag: WETSNOW
    summary_csv: obs/summaries/project_2022_2023/wet_snow_summary.csv
    classes:
      wet: [110]
      valid: [110, 125, 200, 210]
      exclude: [200, 210]

data_assimilation:
  uncertainty:
    scf:
      enabled: true
      assimilation:
        sigma_mode: formula
        aggregate_metric: unc_mean
    wet_snow:
      enabled: true
      assimilation:
        sigma_mode: formula
        aggregate_metric: unc_mean
```

The abbreviated `valid` list above is for orientation; keep the complete class
mapping in the shipped project YAML. The bundled GeoTIFF products include the
required `<stem>_uncertainty.tif` sidecars. Uncertainty uses the configured 0–100
scale. Clouds and nodata remain missing evidence rather than being converted to
high-uncertainty observations.

The project map helps check ROI coverage, station locations, land-cover context
and the snow redistribution factor:

![Rofental tutorial setup overview map with DEM, forcing stations, land cover, aspect and SRF]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/setup_overview.png?v=20260703)

The uncertainty example below shows a continuous layer over valid fSCA pixels.
The lower row zooms into a watershed detail; clouds remain masked.

![Rofental fSCA uncertainty example with land-cover context]({{ site.baseurl }}/assets/images/tutorial/rofental_uncertainty.png)

## 1. Summarize snow-cover products

Purpose: validate and aggregate every configured snow-cover product inside the
ROI and write `obs/summaries/project_2022_2023/scf_summary.csv`.

The positional argument is the project directory. `--overwrite` permits the
existing frozen summary and its completed preprocessing manifest to be replaced;
omit the option when you only want hash-identical work to be reused. `--json`
is available for machine-readable automation but is not needed in the tutorial.

```bash
openamundsen-da observations snow-cover "$PROJECT_DIR" --overwrite
```

Before using the summary, confirm that expected acquisition dates are present,
valid support is sufficient and the fSCA, cloud and invalid fractions are
plausible. With uncertainty enabled, also inspect `unc_mean`, `unc_min`,
`unc_max` and `unc_n_valid`.

Reference rows from `scf_summary.csv`:

| date | n_valid | n_snow | scf | cloud_fraction | invalid_fraction | source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2022-10-03 | 89927 | 46001 | 0.51 | 0.00 | 0.36 | s2_fsc_snowflake_rofental_2022_10_03.tif |
| 2022-10-05 | 80183 | 23973 | 0.30 | 0.00 | 0.41 | s2_fsc_snowflake_rofental_2022_10_05.tif |
| 2022-10-08 | 35064 | 1078 | 0.03 | 0.00 | 0.77 | s2_fsc_snowflake_rofental_2022_10_08.tif |
| 2022-10-13 | 13648 | 306 | 0.02 | 0.00 | 0.93 | s2_fsc_snowflake_rofental_2022_10_13.tif |
| 2022-10-18 | 73805 | 2255 | 0.03 | 0.00 | 0.48 | s2_fsc_snowflake_rofental_2022_10_18.tif |

## 2. Summarize wet-snow products

Purpose: validate and aggregate the configured wet-snow products and write
`obs/summaries/project_2022_2023/wet_snow_summary.csv` plus the configured wet
snow line diagnostics.

The positional argument and options have the same meaning as for snow cover.
Use `--overwrite` here because the tutorial intentionally reproduces the frozen
summary from its source products.

```bash
openamundsen-da observations wet-snow "$PROJECT_DIR" --overwrite
```

Confirm that the selected dates, wet-snow fractions and support counts are
plausible before scheduling events.

Reference rows from `wet_snow_summary.csv`:

| date | wet_snow_fraction | n_valid | n_wet | source |
| --- | ---: | ---: | ---: | --- |
| 2023-03-12 | 0.02 | 156982 | 3453 | WSM_S1A_SAR_track117_2023_03_12_17_07_24.tif |
| 2023-03-16 | 0.03 | 158953 | 4667 | WSM_S1A_SAR_track168_2023_03_16_05_27_37.tif |
| 2023-03-24 | 0.47 | 156982 | 73218 | WSM_S1A_SAR_track117_2023_03_24_17_07_24.tif |
| 2023-03-28 | 0.27 | 158953 | 42301 | WSM_S1A_SAR_track168_2023_03_28_05_27_38.tif |
| 2023-04-05 | 0.06 | 156982 | 9750 | WSM_S1A_SAR_track117_2023_04_05_17_07_24.tif |

## When preprocessing fails

A mixed GeoTIFF/NetCDF directory, missing configured format, invalid class
mapping, absent uncertainty sidecar or escaping path fails before writing a new
summary. If a completed manifest exists but the input hash changed, rerun with
`--overwrite` only after reviewing the changed product or configuration.

For the general input contract, see [Input Data]({{ site.baseurl }}{% link guides/observations.md %}).
Continue with [5. Prepare the Project]({{ site.baseurl }}{% link Tutorial/05-prepare-project.md %}).

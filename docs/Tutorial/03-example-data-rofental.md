---
layout: default
title: "3. Example Data: Rofental"
parent: How to Use
nav_order: 3
permalink: /tutorial/example-data-rofental/
---

# 3. Example Data: Rofental

The tutorial uses the bundled Rofental setup as one concrete, reproducible example.
This page describes that example once so later chapters do not need to repeat the same
context.

The shipped Rofental forcing is a prepared illustrative baseline. Station precipitation
has been deliberately reduced to create a controlled mismatch, so the tutorial can show
particle weighting, resampling and posterior updates clearly.

![Rofental tutorial setup overview map with DEM, forcing stations, land cover, aspect and SRF]({{ site.baseurl }}/assets/images/tutorial/rofental_2022_2023_es30/setup_overview.png?v=20260703)

*Rofental tutorial setup map rendered from `projects/project_2022_2023/maps.yml`, showing the DEM with forcing stations, the Rofental manuscript land-cover grouping, aspect and the snow redistribution factor (SRF).*

## What Is Included

The example setup contains:

- model grids and ROI files under `grids/` and `env/`
- meteorological forcing inputs under `meteo/`
- snow-cover, wet-snow, and station observations under `obs/`
- one project configuration under `projects/project_2022_2023/`

Directory overview:

```text
/data/rofental/
  rofental.yml
  env/
  grids/
  meteo/
  obs/
    snowcover/
    wetsnow/
    stations/
  projects/
    project_2022_2023/
      project_2022_2023.yml
```

## Available Resolutions

The bundled example contains three spatial resolutions:

- `100 m`
- `250 m`
- `500 m`

The active resolution is selected in the setup YAML:

File: `/data/rofental/rofental.yml`

```yaml
domain: "rofental"
resolution: 100
timestep: "3H"
```

The corresponding grids are named with the same pattern, for example:

- `grids/dem_rofental_100.asc`
- `grids/dem_rofental_250.asc`
- `grids/dem_rofental_500.asc`
- `grids/lc_rofental_100.asc`
- `grids/svf_rofental_100.asc`
- `grids/srf_rofental_100.asc`

## Observation Data In The Example

Three observation groups are bundled:

- `obs/snowcover/`: fractional snow covered area (fSCA) rasters
- `obs/wetsnow/`: wet-snow mask rasters
- `obs/stations/`: station snow observations for result plots and validation

For the snow-cover and wet-snow GeoTIFF inputs, the example also ships matching
uncertainty sidecars named `<stem>_uncertainty.tif` next to the source rasters.

These observation raster products are the starting point for the preprocessing chapter.
They are not yet in the tabular form consumed by the step-wise project pipeline, but the
uncertainty layers they need for uncertainty-aware preprocessing are already included.

## Project Configuration Used In The Tutorial

The main project file is:

- `/data/rofental/projects/project_2022_2023/project_2022_2023.yml`

This file defines:

- the project period
- observation directories and product tags
- class mappings
- data assimilation events
- likelihood, resampling, rejuvenation, uncertainty, and output settings

The setup file `/data/rofental/rofental.yml` remains pure openAMUNDSEN configuration,
while the project YAML contains the data assimilation-specific configuration.

Continue with [4. Preprocessing]({{ site.baseurl }}{% link Tutorial/05-pre-processing.md %}).

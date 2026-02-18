---
layout: default
title: 4. Framework
parent: Tutorial
nav_order: 4
permalink: /tutorial/framework/
---

# 4. Framework

## Setups

Create a local tutorial workspace and copy the bundled Rofental setup from the container image.

```bash
mkdir -p openamundsen-da
cd openamundsen-da

docker run --rm -v "$(pwd):/data" \
  ghcr.io/franzwagner-uibk/openamundsen_da:latest \
  cp -a /workspace/examples/rofental /data/rofental
```

The copied setup contains:

- `rofental.yml` (openAMUNDSEN setup config)
- `env/roi.gpkg` (ROI vector)
- `grids/` (DEM/SVF/SRF/LC and ROI masks by resolution)
- `meteo/` (station metadata + forcing)
- `obs/` (SCF and wet-snow summaries/maps + station observations)

## Projects

A project is the DA configuration unit under `setup/projects/`.

For this tutorial:

- Setup dir: `/data/rofental`
- Project dir: `/data/rofental/projects/project_2022_2023`
- Project YAML: `project_2022_2023.yml`

Define reusable variables:

```bash
IMAGE=ghcr.io/franzwagner-uibk/openamundsen_da:latest
SETUP=/data/rofental
PROJECT=/data/rofental/projects/project_2022_2023
SCF_SUM=/data/rofental/obs/project_2022_2023/scf_summary.csv
WET_SUM=/data/rofental/obs/project_2022_2023/wet_snow_summary.csv
```

## Runs

Each project run is built from:

- `project_skeleton` (build step windows),
- observation preparation (`oa-da-scf`, `oa-da-wetsnow-project`),
- project pipeline (`oa-da-project`).

These commands are executed in the next tutorial chapters.

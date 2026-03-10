# Examples bundle

This folder contains ready-to-use example data and setup layouts for `openamundsen_da`.

## What is inside

```text
examples/rofental/
|-- rofental.yml
|-- env/
|-- grids/
|-- meteo/
|-- obs/
`-- projects/
```

### `rofental.yml`
- Setup-level, stable openAMUNDSEN configuration.
- Shared by all projects inside this setup.

### `env/`
- `roi.gpkg`: single-feature region of interest polygon used for masking and aggregation.

### `grids/`
- Static raster inputs at multiple resolutions.
- `dem_rofental_*.asc`: elevation grids.
- `lc_rofental_*.asc`: land-cover grids.
- `svf_rofental_*.asc`: sky-view factor grids.
- `srf_rofental_*.asc`: slope/relief factor grids.

### `meteo/`
- `stations.csv`: station metadata.
- `*.csv`: station forcing time series.

### `obs/`
- Observation data and prepared summaries.
- `snowcover/`: snow-cover raster products.
- `wetsnow/`: Sentinel-1 wet-snow raster products.
- `project_*/`: project-level summaries (`scf_summary.csv`, `wet_snow_summary.csv`).

### `projects/`
- Project runtime configuration/output roots.
- `project_*/project_*.yml`: project-level data assimilation configuration and time span.

This bundle mirrors the documented setup/project structure and is used by tests and examples.


# Examples bundle

This folder contains ready-to-use example data and project layouts for `openamundsen_da`.

## What is inside

Top-level layout in this bundle:

```text
examples/rofental/
├── project.yml
├── env/
├── grids/
├── meteo/
├── obs/
└── propagation/
```

### `project.yml`

- Main project configuration (domain setup, model options, DA settings, observation class mappings, and paths).

### `env/`

- `roi.gpkg`: single-feature region of interest (ROI) polygon used for masking and spatial aggregation.

### `grids/`

- Static raster inputs at multiple resolutions.
- `dem_rofental_*.asc`: digital elevation model grids.
- `lc_rofental_*.asc`: land-cover grids for masking and class filtering.
- `svf_rofental_*.asc`: sky-view factor grids.
- `srf_rofental_*.asc`: slope/relief factor grids.

### `meteo/`

- Meteorological forcing input tables.
- `stations.csv`: station metadata (IDs, coordinates, attributes).
- `*.csv` (for example `bellavista.csv`): station forcing time series.

### `obs/`

- Observation data and prepared summaries.
- `snowcover/`: snow-cover raster products (FSC-style inputs; 602 files).
- `wetsnow/`: Sentinel-1 wet-snow raster products (297 files).
- `stations/`: station-based observation exports (`latschbloder.csv`, `proviantdepot.csv`).
- `season_2019_2020/`, `season_2020_2021/`, `season_2021_2022/`, `season_2022_2023/`: season-level summary folders with `scf_summary.csv` and `wet_snow_summary.csv`.
- `summaries/all_data/`: merged summary tables (`scf_summary.csv`, `wet_snow_summary.csv`).

### `propagation/`

- Season runtime configuration/output root.
- `season_2022_2023/season.yml`: season definition and assimilation timeline.
- `season_2022_2023/point_scf_roi_envelope.csv`: SCF envelope diagnostics.
- `season_2022_2023/point_wet_snow_roi_envelope.csv`: wet-snow envelope diagnostics.

This bundle is intended as a compact reference dataset that mirrors the documented project structure.

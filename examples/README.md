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
- `project_*/maps.yml`: optional custom YAML map recipes for grid-composed project figures such as `setup_overview`. Generated DA-event maps are derived from the project assimilation-event config and written under `results/maps/da_events/`, while custom YAML maps stay at the root of `results/maps/`. Generated SCF DA-event maps now default to `open-loop snow cover`, `posterior snow-cover probability`, and `satellite FSC observation`.
- The shipped `rofental` project also enables the benchmark stage and adds `station_swe` as an extra benchmark family, so completed runs write `results/benchmark/` plus the headline DA-skill plot `results/plots/assim/scores/performance_scores.png` in addition to the usual DA outputs. Because `station_hs` is assimilated in that project, the resulting `station_swe` benchmark rows appear as `semi_independent` in the benchmark outputs. Station benchmark rows also expose sigma-aware `zSkill`, and the headline plot grows a third panel when those scores are available.

This bundle mirrors the documented setup/project structure and is used by tests and examples.

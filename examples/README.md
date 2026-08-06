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
- `summaries/project_*/`: project-level summaries (`scf_summary.csv`, `wet_snow_summary.csv`) plus optional `wet_snow_line_diagnostics.csv` and per-date WSLA profile CSVs after wet-snow summarization. In v0.9 the primary `wet_snow_line` values are 50% wet-fraction crossings, while sector-relative WSLA diagnostics are stored as companion analysis fields in the same diagnostics/profile family.

### `projects/`
- Project runtime configuration/output roots.
- `project_*/project_*.yml`: project-level data assimilation configuration and time span.
- `project_*/maps.yml`: optional custom YAML map recipes for grid-composed project figures such as `setup_overview`. Generated DA-event maps are derived from the project assimilation-event config and written under `results/maps/da_events/`, while custom YAML maps stay at the root of `results/maps/`. Generated DA-event rows use four columns: `open loop`, `prior`, `posterior`, and `reference`. Snow-state reference columns show `posterior - prior` DA increments, while FSC and wet-snow reference columns show the satellite observation. Top-level sub-domain SCF events use a taller same-file layout with snow-cover panels above the snow-depth response. Generated `wet_snow_line` DA-event maps show spatial WSF without WSLA contours in the first row and derived WSLA contours only in the elevation-band WSF row.
- Map panels with `show_station_marker: true` retain the forcing-only marker by default. Set `station_marker_mode: sources_and_roles` to classify forcing and snow-observation stations using `station_match_tolerance_m` (10 m by default). The classified mode reads coordinates and role flags from the configured `stations_da_metadata.csv`; `station_categories` adds its four-class legend. Overview panels can opt into full subdomain ID labels with `show_subdomain_labels: true`; labels moved to prevent overprinting automatically receive leader lines to their subdomains.
- `sigma_rh` is interpreted as a dew-point-temperature perturbation scale in K for both shipped projects.
- The shipped `rofental` project uses the promoted `project_2022_2023` baseline: seed `1415935400`, five station snow-depth events, two fSCA (`scf`) events on `2023-04-26` and `2023-05-26`, and one `wet_snow_line` event from the `2023-05-03T05:26:24Z` acquisition. The WSLA event uses a 200 m comparison sigma and a 0.95 finite-member gate. fSCA uses its uncertainty layer. The wet-snow summaries retain `wet_snow_fraction` as a diagnostic benchmark and identify WSLA with `uppermost_crossing_fraction`.
- The shipped `rofental` project also enables the benchmark stage and adds `station_swe` as an extra benchmark family, so completed runs write `results/benchmark/` plus the headline DA-skill plot `results/plots/assim/scores/performance_scores.png` in addition to the usual DA outputs. Because `station_hs` is assimilated in that project, the resulting `station_swe` benchmark rows appear as `semi_independent` in the benchmark outputs. Station benchmark rows also expose sigma-aware `zSkill`, and the headline plot grows a third panel when those scores are available.

This bundle mirrors the documented setup/project structure and is used by tests and examples.

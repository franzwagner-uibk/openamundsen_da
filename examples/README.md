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

examples/subdomains/
|-- subdomains.yml
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
- `summaries/project_*/`: project-level summaries (`scf_summary.csv`, `wet_snow_summary.csv`) plus optional `wet_snow_line_diagnostics.csv` and per-date WSLA profile CSVs after wet-snow summarization. In v1 the primary `wet_snow_line` values are 50% wet-fraction crossings, while sector-relative WSLA diagnostics are stored as companion analysis fields in the same diagnostics/profile family.

### `projects/`
- Project runtime configuration/output roots.
- `project_*/project_*.yml`: project-level data assimilation configuration and time span.
- `project_*/maps.yml`: optional custom YAML map recipes for grid-composed project figures such as `setup_overview`. Generated DA-event maps are derived from the project assimilation-event config and written under `results/maps/da_events/`, while custom YAML maps stay at the root of `results/maps/`. Generated DA-event rows use four columns: `open loop`, `prior`, `posterior`, and `reference`. Snow-state reference columns show `posterior - prior` DA increments, while FSC and wet-snow reference columns show the satellite observation. Top-level sub-domain SCF events use a taller same-file layout with snow-cover panels above the snow-depth response. Generated `wet_snow` and `wet_snow_line` DA-event maps include a spatial elevation-band WSF row and draw WSLA contours panel-locally, with observation WSLA only in the observation/reference panel.
- `sigma_rh` is interpreted as a dew-point-temperature perturbation scale in K for both shipped projects.
- The shipped `rofental` project uses the promoted tuned `project_2022_2023` baseline: seed `113`, five station snow-depth events, one spring `wet_snow_line` event, and two SCF events with stronger valid-pixel support. It keeps `wet_snow_fraction` as a diagnostic benchmark in the wet-snow summary outputs.
- The shipped `rofental` project also enables the benchmark stage and adds `station_swe` as an extra benchmark family, so completed runs write `results/benchmark/` plus the headline DA-skill plot `results/plots/assim/scores/performance_scores.png` in addition to the usual DA outputs. Because `station_hs` is assimilated in that project, the resulting `station_swe` benchmark rows appear as `semi_independent` in the benchmark outputs. Station benchmark rows also expose sigma-aware `zSkill`, and the headline plot grows a third panel when those scores are available.

This bundle mirrors the documented setup/project structure and is used by tests and examples.

## Subdomain Example

`examples/subdomains` is the shipped subdomain setup. It contains 8 avalanche-report subregions from the North Tyrol source data, static grids at 50, 100, 250, and 500 m, `openamundsen-v2` forcing stations selected from the ROI plus a 10 km buffer, ROI station snow-depth observations, clipped SnowFLAKES FSC NetCDF files, a generic setup-overview map recipe, and a station-free SCF/snow-depth/ESS/score custom result overview for subdomain reports.

The project YAML intentionally carries a broad list of candidate DA events. During subdomain preparation and execution, generic `data_assimilation.subdomain_event_filter` settings decide which events each subdomain can assimilate based on local observation availability, FSC cloud fraction, and active station support. Dropped events are written to subdomain manifests and `results/subdomain_dropped_events.csv`; generated DA-event maps use this file to mark affected subdomains as `no DA`.

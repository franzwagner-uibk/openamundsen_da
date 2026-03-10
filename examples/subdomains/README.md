# Sub-domain Example (Rofental)

This setup is a sub-domain mode integration example for `oa-da-subdomain`.

Layout:
- `subdomains.yml`: setup-level openAMUNDSEN config.
- `projects/project_2022_2023/project_2022_2023.yml`: project-level data assimilation config.
- `env/subdomains.gpkg`: non-overlapping ROI split into three sub-domains (`sd_01`, `sd_02`, `sd_03`).
- `grids/roi_<domain>_<resolution>.asc`: canonical ROI mask used by data assimilation (generated automatically from regions vector when missing).

Current data content:
- This setup now contains the same example data as `examples/rofental`:
  - grids: `grids/`
  - meteo: `meteo/`
  - raw snow-cover: `obs/snowcover/`
  - raw wet-snow: `obs/wetsnow/`
  - station observations: `obs/stations/`

For your own setup:
1. Replace `env/subdomains.gpkg` with your real sub-domain geometry.
2. Update paths in `subdomains.yml` to your real `grids` and `meteo` directories.
3. Adapt the project YAML under `projects/` to your data assimilation events, time span, and `obs` directories.

Default output behavior:
- `oa-da-subdomain merge` writes compact data assimilation grids to `projects/<project>/results/grids/da_output_grids.nc`.
- `data_assimilation.output.retention: compact` prunes heavy member grid artifacts after merge.
- Point outputs and point plots stay inside each sub-domain project directory.


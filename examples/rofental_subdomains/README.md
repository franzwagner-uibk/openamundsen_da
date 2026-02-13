# Rofental Sub-domain Example

This setup is a sub-domain mode integration example for `oa-da-subdomain`.

Layout:
- `rofental_subdomains.yml`: setup-level openAMUNDSEN config.
- `projects/project_2022_2023/project_2022_2023.yml`: project-level DA config.
- `env/subdomains.gpkg`: non-overlapping ROI split into three sub-domains (`sd_01`, `sd_02`, `sd_03`).

Current data content:
- This setup now contains the same example data as `examples/rofental`:
  - grids: `grids/`
  - meteo: `meteo/`
  - raw snow-cover: `obs/snowcover/`
  - raw wet-snow: `obs/wetsnow/`
  - station observations: `obs/stations/`

For your own setup:
1. Replace `env/subdomains.gpkg` with your real sub-domain geometry.
2. Update paths in `rofental_subdomains.yml` to your real `grids`, `meteo`, and `obs` directories.
3. Adapt the project YAML under `projects/` to your DA events and time span.

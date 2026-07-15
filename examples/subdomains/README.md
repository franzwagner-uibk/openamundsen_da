# Subdomain Example

This shipped example covers a larger alpine ROI as 8 avalanche-report subdomains.

- Spatial domain: `env/subdomains.gpkg` and `env/roi.gpkg`, EPSG:25832.
- Temporal domain: `2022-10-01` to `2023-06-30 21:00:00`.
- Default resolution: `100 m`; available grid resolutions: `50, 100, 250, 500 m`.
- Forcing: 158 `openamundsen-v2` stations within the ROI plus 10 km buffer, trimmed to the project window.
- Station snow depth: 35 ROI stations in `obs/stations`, with `use_for_da` and `use_for_benchmark` role flags.
- FSC: 15 clipped SnowFLAKES NetCDF files in `obs/snowcover`; configured as project-level event candidates. The subdomain event filter drops cloudy subdomain/date combinations above 20% cloud cover, except documented per-subdomain overrides in the project YAML. FSC uncertainty is calculated internally from the NetCDF `fsc` layer with `u_min: 5`, `u_max: 20`, and forest land-cover penalties.
- DA config: 30 ensemble members, ESS threshold ratio 0.7, full output retention, and four-variable forcing/rejuvenation perturbations for temperature, precipitation, humidity, and shortwave radiation.
- Maps: `projects/project_2022_2023/maps.yml` adds a setup overview. Generated DA-event maps are rendered automatically from the configured assimilation events.

The data-assimilation and plain-model subdomain modes are intentionally
separate. Run the data-assimilation example in four explicit stages:

```bash
project=examples/subdomains/projects/project_2022_2023
openamundsen-da subdomains prepare "$project" --regions examples/subdomains/env/subdomains.gpkg --station-buffer-km 10 --grid-buffer-m 10000 --overwrite
openamundsen-da subdomains run "$project" --max-workers 8 --inner-max-workers 3 --overwrite
openamundsen-da subdomains merge "$project"
openamundsen-da subdomains render "$project" --max-workers 8
```

For an ordinary openAMUNDSEN simulation without projects or assimilation, use
the distinct `openamundsen-da subdomains model prepare|run|merge` command tree.

The FRAMES-specific build logic is external to `openamundsen_da`:
`/home/franz/workspace/repos/scripts/04-openAMUNDSEN/buildSubdomainExample.py`.

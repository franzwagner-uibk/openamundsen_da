# North Tyrol Subdomain Example

This shipped example covers the North Tyrol test site as 8 avalanche-report subdomains.

- Spatial domain: `env/subdomains.gpkg` and `env/roi.gpkg`, EPSG:25832.
- Temporal domain: `2022-10-01` to `2023-06-30 21:00:00`.
- Default resolution: `100 m`; available grid resolutions: `50, 100, 250, 500 m`.
- Forcing: 158 `openamundsen-v2` stations within the ROI plus 10 km buffer, trimmed to the project window.
- Station snow depth: 35 ROI stations in `obs/stations`, with `use_for_da` and `use_for_benchmark` role flags.
- FSC: 18 clipped SnowFLAKES NetCDF files in `obs/snowcover`; selected per subdomain with at most 20% cloud cover, except documented per-subdomain overrides in the project YAML.
- Maps: `projects/project_2022_2023/maps.yml` adds a setup overview plus focused snow-depth/SWE response maps on selected DA dates. Generated DA-event maps are still rendered automatically from the configured assimilation events.

Run the example with:

```bash
oa-da-subdomain pipeline --setup-dir examples/subdomains --project-dir examples/subdomains/projects/project_2022_2023 --regions examples/subdomains/env/subdomains.gpkg --station-buffer-km 10 --grid-buffer-m 10000 --max-workers 8 --inner-max-workers 3 --overwrite
```

The FRAMES-specific build logic is external to `openamundsen_da`:
`/home/franz/workspace/repos/scripts/04-openAMUNDSEN/buildNorthTyrolSubdomainExample.py`.

# Observer Agent Notes

Inherit the parent `openamundsen_da/AGENTS.md`. This file adds observation-preprocessing rules.

- Observation preprocessing should be strict fail-fast on missing config, missing layers, invalid grids, ambiguous dates, or unsupported product assumptions.
- Keep ROI masking, product tags, and output naming deterministic; downstream assimilation expects stable contracts like `obs_scf_<PRODUCT>_YYYYMMDD.csv`.
- Treat cloud pixels as gaps, not uncertainty penalties; use land-cover exclusion for unusable classes and uncertainty penalties for usable-but-uncertain classes.
- Keep NetCDF and GeoTIFF ingest rules explicit; do not guess variable names, time variables, or sidecar paths when config should define them.
- If preprocessing schema or obs-file selection changes, update tests, docs, examples, and assimilation consumers together.

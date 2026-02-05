---
layout: default
title: API Quick Reference
parent: Reference
nav_order: 2
---

# API Quick Reference

Scripting-oriented entry points. Everything else should be used via the CLI. Unlisted symbols are internal/unstable.

## Seasonal orchestration
- `openamundsen_da.pipeline.season.cli(argv=None)` — main season driver (same as `oa-da-season`). Args mirror the CLI flags.
- `openamundsen_da.pipeline.season_skeleton.cli(argv=None)` — build step skeletons (same as `oa-da-season-skeleton`).
- `openamundsen_da.pipeline.cleanup.cli(argv=None)` — remove intermediate artifacts (same as `oa-da-clean-season`).

## Particle filter
- `openamundsen_da.methods.pf.assimilate_scf.assimilate_scf_for_date(...)` — run SCF assimilation for a single date.
- `openamundsen_da.methods.pf.resample.resample_from_weights(...)` — systematic resampling helper.
- `openamundsen_da.methods.pf.rejuvenate.rejuvenate(...)` — perturb posterior to maintain ensemble spread.

## Observation operators
- `openamundsen_da.methods.h_of_x.model_scf.compute_model_scf(...)` — derive model SCF for a step/season.
- `openamundsen_da.methods.wet_snow.area.compute_model_wet_snow_fraction(...)` — derive model wet-snow fraction.

## Observation processing
- `openamundsen_da.observer.snowcover.cli_main(argv=None)` — summarize snow-cover rasters (GeoTIFF/NetCDF).
- `openamundsen_da.observer.wetsnow.cli_main(argv=None)` — summarize wet-snow rasters.
- `openamundsen_da.observer.satellite_scf.cli_main(argv=None)` — create per-date SCF CSVs from summaries.
- `openamundsen_da.observer.satellite_wet_snow_s1.cli_main(argv=None)` — create per-date wet-snow CSVs.

## Batch processing
- `openamundsen_da.batch.pipeline.run_pipeline(...)` — prepare → run → merge → plot for subregions (same as `oa-da-batch pipeline`).
- `openamundsen_da.batch.prepare.prepare_batch(...)` — write per-subregion configs and inputs.
- `openamundsen_da.batch.run.run_batch(...)` — launch subregion runs.
- `openamundsen_da.batch.merge.merge_grids/merge_points(...)` — combine subregion outputs.
- `openamundsen_da.batch.plot.plot_station_comparisons(...)` — compare merged model vs station obs.

## Utilities (select)
- `openamundsen_da.util.stats` — weight normalization, ESS, likelihood helpers.
- `openamundsen_da.util.parallel.run_tasks_with_pool(...)` — simple process pool runner.
- `openamundsen_da.io.paths` — project/season path resolution helpers.

## Stability note

These symbols are provided for power users and may change between minor versions. Prefer CLI where possible; pin versions for reproducibility.

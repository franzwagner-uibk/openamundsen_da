---
layout: default
title: API Quick Reference
parent: Reference
nav_order: 2
---

# API Quick Reference

Scripting-oriented entry points. Everything else should be used via the CLI.

## Setup orchestration
- `openamundsen_da.pipeline.project.cli(argv=None)` -> main setup driver (same as `oa-da-project`).
- `openamundsen_da.pipeline.project_skeleton.cli(argv=None)` -> build step skeletons.
- `openamundsen_da.pipeline.cleanup.cli(argv=None)` -> remove intermediate artifacts.

## Particle filter
- `openamundsen_da.methods.pf.assimilate_fraction.assimilate_scf_for_date(...)`
- `openamundsen_da.methods.pf.assimilate_fraction.assimilate_wet_snow_for_date(...)`
- `openamundsen_da.methods.pf.assimilate_station.assimilate_station_hs_for_date(...)`
- `openamundsen_da.methods.pf.assimilate_station.assimilate_station_swe_for_date(...)`
- `openamundsen_da.methods.pf.resample.resample_from_weights(...)`
- `openamundsen_da.methods.pf.rejuvenate.rejuvenate(...)`

## Observation operators
- `openamundsen_da.methods.h_of_x.model_scf.compute_model_scf(...)`
- `openamundsen_da.methods.wet_snow.area.compute_model_wet_snow_fraction(...)`

## Observation processing
- `openamundsen_da.observer.snowcover.cli_main(argv=None)`
- `openamundsen_da.observer.wetsnow.cli_main(argv=None)`
- `openamundsen_da.observer.satellite_scf.cli_main(argv=None)`
- `openamundsen_da.observer.satellite_wet_snow_s1.cli_main(argv=None)`

## Batch processing
- `openamundsen_da.subdomain.pipeline.run_pipeline(...)`
- `openamundsen_da.subdomain.prepare.prepare_subdomains(...)`
- `openamundsen_da.subdomain.run.run_subdomains(...)`
- `openamundsen_da.subdomain.merge.merge_grids(...)`
- `openamundsen_da.subdomain.merge.merge_points(...)`
- `openamundsen_da.methods.viz.plots.subdomain.station_comparisons.plot_station_comparisons(...)`

## Utilities (selected)
- `openamundsen_da.util.stats`
- `openamundsen_da.util.parallel.run_tasks_with_pool(...)`
- `openamundsen_da.io.paths`

## Stability note
Prefer CLI usage for stable workflows. Direct Python APIs may change between minor versions.



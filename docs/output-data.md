---
layout: default
title: Output data
parent: Documentation
nav_order: 5
---

# Output data

A project is successful only after ensemble execution, assimilation, compact
grid export, benchmarking and every configured plot, map and report pass their
checks. The atomic run manifest is the authoritative completion record.

## Run manifest

`results/run_manifest.json` records configuration and input hashes, the
openAMUNDSEN-DA software version, stage/member state, output inventory and
cleanup results. The optional `image` and `image_digest` fields are populated
only when the runtime supplies `OPENAMUNDSEN_DA_IMAGE` and
`OPENAMUNDSEN_DA_IMAGE_DIGEST`; an empty value does not invalidate a run.
Interrupted or failed runs keep restart state. A mismatched configuration or
input inventory cannot resume an existing run.

## Compact gridded result

Every successful data assimilation project writes:

```text
results/grids/da_output_grids.nc
```

This compressed NetCDF contains the configured scientific variables and metrics,
including the open loop, ensemble statistics and event-weighted analysis fields.
Typical names are `open_loop_<var>`, `ens_mean_<var>`, `ens_std_<var>`,
`increment_<var>`, `analysis_mean_<var>` and `analysis_increment_<var>`.

The model propagation grids feeding this builder can be NetCDF or GeoTIFF as
selected in the setup YAML. The compact data assimilation result itself is always
NetCDF.

## Tables and diagnostics

The standard result tree includes:

```text
results/
  benchmark/
  grids/da_output_grids.nc
  maps/
  misc/
  plots/assim/
  plots/perf/
  plots/points/
  plots/results/
  reports/project_report.pdf
```

Benchmark tables and their manifest live under `results/benchmark/`. Weight,
effective sample size and score plots live under `results/plots/assim/`.
`results/plots/perf/` is refreshed during every project run. The CSV retains
raw filesystem telemetry, while the plot focuses on CPU, RAM, project-directory
size and optional CPU temperature. For successful single-domain runs, exact
project-size samples immediately before and after restart-state cleanup make the
storage reduction visible. The reviewed Rofental walkthrough explains the files in
[Results and Diagnostics]({{ site.baseurl }}{% link Tutorial/07-results-and-diagnostics.md %}).

## Cleanup and retention

After all required outputs validate, a successful single-domain run deletes only
package-owned restart pickles and invalid state pointers. The manifest records
their relative paths, count and bytes. Scientific results and member grids remain.

For an older single-domain project, preview and then apply the same safe cleanup:

```bash
openamundsen-da clean PROJECT_DIR
openamundsen-da clean PROJECT_DIR --apply
```

Single-domain cleanup does not delete scientific member grids. Subdomain compact
retention can delete merge inputs only after the merged grid, configured render
outputs and report have validated.

## Subdomain outputs

Data assimilation subdomains merge into the same compact NetCDF and project-level
summary/report tree. Plain-model subdomains merge the exact configured NetCDF or
GeoTIFF model grids. Both modes use hard mosaics: no interpolation, blending or
particle exchange occurs across boundaries.

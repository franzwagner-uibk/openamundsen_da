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

When compact variables are explicitly configured, output completeness is a
hard contract. The setup must configure the matching openAMUNDSEN grid output
name before propagation, every open-loop and member NetCDF must contain that
source and the compact file must contain every requested metric-variable pair.
The same validation is repeated for subdomain leaf files and their merged
NetCDF, so a consistently incomplete set of leaves cannot be accepted.

The model propagation grids feeding this builder can be NetCDF or GeoTIFF as
selected in the setup YAML. The compact data assimilation result itself is always
NetCDF.

Compact projects also retain two compressed, all-member time-series files:

```text
results/points/ensemble_points.nc
results/forcing/ensemble_forcing.nc
```

Their dimensions are `time`, `member`, `point` and `time`, `member`,
`station`, respectively. They contain the open loop and every ensemble member,
so standard Python and R NetCDF tooling can plot or analyze the retained time
series after member CSV cleanup. Projects with satellite observation events
also retain `results/grids/da_map_support.nc`, which contains the event fields
needed by the configured map renderer.

Weighted propagation means, standard deviations and increments use the
persistent PF prior ledger for each step. Configured ensemble minima and maxima
describe the unweighted member collection; compact grid quantiles are not
currently generated.
Event-analysis fields use the normalized pre-resampling posterior weights.
Unweighted member traces and extrema describe the materialized member
collection and are not interchangeable with the weighted analysis.

## Particle-weight and stochastic provenance

Every step stores `assim/prior_weights.csv` with `member_id`, `log_weight` and
`weight`, plus `prior_weights_manifest.json`. Event tables add
`prior_log_weight`, `prior_weight`, `log_likelihood`, posterior `log_weight` and
`weight`, prior/posterior ESS, threshold and `resampled`. Paired manifests hash
the configuration, observation/model inputs and ledger ancestry.

Resampling indices and manifests distinguish a mirrored posterior from an
actual systematic resample. `rejuvenate_manifest.json` records the `keyed-v1`
RNG contract, configured seed and event-specific perturbations. Incomplete old
chains without these contracts remain readable but cannot resume under the
corrected method.

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

After all required outputs validate, `retention: compact` removes package-owned
restart checkpoints, member point and forcing CSVs and replaceable member grids.
The atomic, versioned `results/retention_manifest.json` records every planned
and completed deletion batch, source inventory, retained-consumer inventory,
producer-manifest or completed-stage digest, final consumer and regeneration recipe. Cleanup is
contained within the project and can resume an interrupted batch only when the
current source files and retained dependencies still match their recorded size
and SHA-256. Dependencies are rechecked before every resumed deletion and a
recreated path is recorded as a new generation. The retained NetCDFs, weights, benchmarks, plots, maps,
reports, logs and scientific configuration remain.

`retention: full` skips this artifact cleanup and preserves the raw member
forcing, points, grids and restart states for reanalysis.

For an older single-domain project, preview and then apply the same safe cleanup:

```bash
openamundsen-da clean PROJECT_DIR
openamundsen-da clean PROJECT_DIR --apply
```

Single-domain and subdomain leaf projects use the same cleanup lifecycle.
Subdomain compact retention deletes leaf and parent merge inputs only after the
merged grid, configured render outputs and report have validated. An interrupted
or low-disk run resumes existing work non-destructively; rebuilding requires an
explicit overwrite request.

The current compact lifecycle is deliberately boundary-based. It shortens every
generated forcing copy to its consuming step and incrementally removes obsolete
restart checkpoints, but keeps forcing/point CSVs and raw grids until final
compaction and rendering succeed. Disk admission is checked between steps and
uses all unfinished leaves' retained growth, concurrency-bound rolling
checkpoints and the atomic parent-merge temporary. Active model members are not
killed mid-propagation. Per-step compact fragments and cooperative mid-member
disk stops remain future work. This is a safe first increment, not evidence that
the full 100 m Euregio ES50 workflow already fits on a 3.6 TB filesystem.

Compact point and forcing stores collapse overlapping timestamps from adjacent
steps by their numeric mean, matching the established raw-series readers.
Cleanup validates every retained value against that collapsed raw source before
deletion. Subdomain cleanup retains each leaf's `da_output_grids.nc`, so leaf
snow-depth and SWE maps remain rerenderable without raw member grids.
Compact grid, point, forcing and map-support writes use validated,
filesystem-synchronized same-directory temporaries. Raw grid cleanup
additionally requires complete configured metrics and member sources, while
satellite map support is checked against its ROI, probability domain and raw
source values.

## Subdomain outputs

Data assimilation subdomains merge into the same compact NetCDF and project-level
summary/report tree. Plain-model subdomains merge the exact configured NetCDF or
GeoTIFF model grids. Both modes use hard mosaics: no interpolation, blending or
particle exchange occurs across boundaries.

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
restart checkpoints, member point and forcing CSVs, replaceable member grids and
step-local forcing PNGs after their configured render completes. The compact
forcing NetCDF remains the source for rerendering project-wide forcing plots.
The atomic, versioned `results/retention_manifest.json` records explicit cleanup
generations and every planned and completed deletion batch, source inventory,
retained-consumer inventory, producer-manifest or completed-stage digest, final
consumer and regeneration recipe. A verified overwrite starts a new generation
and marks the prior one superseded. Historical inventories remain available for
audit, while resume validation uses only the active generation. A planned
generation resumes only when its complete retained-consumer and producer
inventories and the exact surviving source inventory still match; a new
overwrite generation cannot be mixed into an interrupted cleanup. Cleanup is
contained within the project and can resume an interrupted batch only when the
current source files and retained dependencies still match their recorded size
and SHA-256. Dependencies are rechecked before every resumed deletion and a
recreated path is recorded as a new generation. Completed older ledgers are
upgraded to the stronger identity contract when read. A planned ledger from the
older contract is refused because its not-yet-planned cross-class sources cannot
be reconstructed safely. The retained NetCDFs, weights, benchmarks, plots,
maps, reports, logs and scientific configuration remain.

`retention: full` skips this artifact cleanup and preserves the raw member
forcing, points, grids and restart states for reanalysis.

For an older single-domain project, preview and then apply the same safe cleanup:

```bash
openamundsen-da clean PROJECT_DIR
openamundsen-da clean PROJECT_DIR --apply
```

Single-domain and subdomain leaf projects use the same cleanup lifecycle. A
successful compact subdomain leaf is finalized after its own benchmark,
configured render outputs and report validate. Its raw member forcing, point,
grid and final restart artifacts are then removed immediately, while the leaf
compact grid, point/forcing stores, DA map support, weights and report remain.
Step-local forcing PNGs are deleted only after the compact forcing store matches
their still-present raw source and stable render-completion evidence exists. The
evidence and producing member manifests are bound into the cleanup ledger; the
report may subsequently be refreshed with the final performance snapshot
without invalidating cleanup provenance.
`leaf_finalization_manifest.json` binds those retained inputs before the next
leaf wave is admitted. An interrupted or low-disk run resumes existing work
non-destructively; rebuilding requires an explicit overwrite request.

The current compact lifecycle is deliberately boundary-based. It shortens every
generated forcing copy to its consuming step and incrementally removes obsolete
restart checkpoints, but keeps a leaf's forcing/point CSVs and raw grids until
that leaf's final compaction and rendering succeed. Subdomain admission uses
bounded waves: current filesystem usage already includes measured retained
compact leaves, while the additional reserve covers only the active wave,
rolling checkpoints, the projected compact outputs of queued leaves and
unfinished parent merge/render output. Active model members are not killed
mid-propagation. Per-step compact fragments and
cooperative mid-member disk stops remain future work. This is a safe increment,
and its audited prepared-Euregio estimate fits a clean 3.6 TB filesystem. That
estimate is not a substitute for the first complete production-run acceptance,
and unrelated resident runs must be archived before relying on the envelope.

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
Before a generation is marked complete, every active batch's source inventory,
removed-path state, retained consumers and producer evidence are revalidated.
The supported single-domain run API repeats complete-ledger validation after
the final performance/report refresh and refuses to publish a successful run
manifest when retention evidence is incomplete or invalid.
Full retention keeps raw member grids and forcing/point files. SCF and wet-snow
projects in this mode must successfully rebuild their configured event fields
from that raw render support and do not require the compact-only
`da_map_support.nc` archive.

## Subdomain outputs

Data assimilation subdomains merge into the same compact NetCDF and project-level
summary/report tree. Plain-model subdomains merge the exact configured NetCDF or
GeoTIFF model grids. Both modes use hard mosaics: no interpolation, blending or
particle exchange occurs across boundaries.

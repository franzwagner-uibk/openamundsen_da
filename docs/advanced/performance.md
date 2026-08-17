---
layout: default
title: Performance
parent: Advanced
nav_order: 3
---

# Performance

openAMUNDSEN-DA parallelizes independent model propagations and subdomains with
process workers. Numeric-library threads should remain pinned to one so worker
pools do not oversubscribe the host.

## Worker selection

For a single project, useful propagation workers are bounded by the open loop plus
ensemble members. Start below the host CPU count and watch memory and disk use.
For multiple independent projects, divide the total CPU budget across projects
rather than assigning every project the full host count.

`--max-workers` limits project or subdomain workers. Data assimilation subdomain
runs also accept `--inner-max-workers` for member propagations inside each active
subdomain.

## Monitoring

Every project run writes live metrics to:

```text
results/plots/perf/project_perf_metrics.csv
results/plots/perf/project_perf.png
```

Use these outputs to identify sustained memory pressure, project-directory
growth and CPU under-utilization before changing the worker count. The CSV keeps
raw filesystem used/free/total telemetry, but filesystem utilization is not
drawn in the plot. Temperature is recorded only when a readable host sensor is
exposed to the container.

The plot records exact project-size samples immediately before and after the
automatic restart-state cleanup of a successful single-domain run. Its summary
therefore reports `Project: peak ... GB → final ... GB`. Staged subdomain runs
do not invent an equivalent cleanup sample; interpret their project-size series
according to the configured retention stage.

CPU and RAM are sampled every 5 seconds and the live plot is refreshed every
30 seconds. The exact recursive project-directory size scan is intentionally
less frequent because it can be expensive for projects containing millions of
files; it defaults to every 150 seconds. The plot measures its project-size tick
and axis-label widths before positioning the optional temperature axis, so its
fixed figure size remains readable for both small projects and sizes of several
terabytes.

## Storage

Grid resolution, duration, ensemble size, selected model variables and retention
dominate disk use. The compact `da_output_grids.nc` is the primary post-run grid
product. Successful runs automatically remove restart pickles, but member grids
remain available unless a separate safe retention operation inventories them.

Overwrite admission treats restart checkpoints like other atomic products: an
accepted checkpoint and its complete replacement may coexist until durable
promotion. Existing checkpoint bytes remain in live filesystem use while the
full replacement generation is added to projected growth.

Compact subdomain execution admits at most one outer-worker-sized wave at a
time. Every successful leaf is compacted, its retained consumers are hash-bound
and its eligible raw member artifacts are removed before the next wave receives
disk admission. Files retained by completed leaves are already included in the
live filesystem usage; the projected-growth reserve adds the active wave plus
the compact outputs still expected from queued leaves and unfinished parent
merge, map, plot and report output. A failed leaf is not final-cleaned and
therefore retains its restartable predecessor state.

Storage estimation is separated from the step-boundary hot path. The
coordinator builds the conservative component plan before workers start and
again only when missing, stale or inconsistent recovery authority requires a
rebuild. One command-scoped source catalog hashes and parses each physical
source inode once, even when many leaf/step paths alias it. New leaf waves,
finalization, parent merge and render reuse the immutable ledger obligations
and upward-only producer accounting. Member propagation and forcing
producers attach byte/count summaries to their existing manifests. At the next
ordinary step boundary the coordinator applies those summaries, raises
observed component high-water marks when necessary and calls `disk_usage`
once. It does not scan configs, source data, accumulated artifacts or sibling
project trees. The compact ledger serialization does scale with the immutable
prepared lifecycle entries (leaves and steps), so it is bounded by the
preflight plan rather than constant size. The retained ledger reports disk-check
counts/durations plus pre-commit coordinator-request count, cumulative latency
and maximum latency. Caller-side performance tests cover the final atomic ledger
write and end-to-end request latency.

The estimator currently proves aggregate component bounds, not heterogeneous
per-step shares. For safety, propagation summaries never release that aggregate:
it remains reserved until authoritative leaf finalization and may temporarily
double-count bytes already visible in filesystem usage. This can conservatively
refuse a run that would fit. Observed producer high-water marks can only raise
the immutable obligations, and validated finalization is the only operation
that releases them. A full estimate is rebuilt only for recovery from missing,
stale or inconsistent authority.

The performance monitor performs one project-size baseline reconciliation when
it starts, then derives live growth from the reservation ledger's materialized
and removed byte counters. It does not recursively scan the growing project at
periodic samples. The terminal snapshot performs one final physical
reconciliation. PNG rendering is downsampled to a bounded history, refreshes
less often as the run grows and breaks lines across monitor gaps. The companion
`project_perf_phases.csv` and `project_perf_phases.png` distinguish preflight,
propagation, finalization and unmonitored downtime. Temperature text reports
the p50, p95 and maximum plus accumulated time above the configured reporting
levels; these observations are not evidence of hardware throttling by
themselves.

The first-run reserve for step-local forcing PNGs is calibrated at 4,400 bytes
per station, member and plotted day, then increased by the standard 25% observed
artifact margin. For the audited 4,555 station-leaf identities, ES50 and a full
leap hydrological year this is about 465 GB. Compact retention removes these
derived PNGs after the compact forcing NetCDF and stable render-completion
evidence validate, so they cannot accumulate across all queued leaves.
`retention: full` budgets and keeps them.

Queued compact-leaf estimates also retain a diagnostics, log, member-metadata
and rendered-output allowance. It is calibrated from 8.01 GB for the audited
90-leaf ES30 run, scaled by the open loop plus ensemble members and increased by
25%. For ES50 this contributes at least 16.47 GB across the prepared setup;
observed artifacts can only refit the allowance upward.

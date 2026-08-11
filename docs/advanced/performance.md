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

Compact subdomain execution admits at most one outer-worker-sized wave at a
time. Every successful leaf is compacted, its retained consumers are hash-bound
and its eligible raw member artifacts are removed before the next wave receives
disk admission. Files retained by completed leaves are already included in the
live filesystem usage; the projected-growth reserve adds the active wave plus
the compact outputs still expected from queued leaves and unfinished parent
merge, map, plot and report output. A failed leaf is not final-cleaned and
therefore retains its restartable predecessor state.

The first-run reserve for step-local forcing PNGs is calibrated at 4,400 bytes
per station, member and plotted day, then increased by the standard 25% observed
artifact margin. For the audited 4,555 station-leaf identities, ES50 and a full
leap hydrological year this is about 465 GB. Compact retention removes these
derived PNGs after the compact forcing NetCDF and leaf report validate, so they
cannot accumulate across all queued leaves. `retention: full` budgets and keeps
them.

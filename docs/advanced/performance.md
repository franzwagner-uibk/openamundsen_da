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

## Storage

Grid resolution, duration, ensemble size, selected model variables and retention
dominate disk use. The compact `da_output_grids.nc` is the primary post-run grid
product. Successful runs automatically remove restart pickles, but member grids
remain available unless a separate safe retention operation inventories them.

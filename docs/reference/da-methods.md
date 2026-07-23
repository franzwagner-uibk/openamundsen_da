---
layout: default
title: Data Assimilation Implementation
parent: Reference
nav_order: 4
---

# Data Assimilation Implementation

This page identifies software stages and artifacts. For equations, methodological
motivation and scientific interpretation, use Wagner et al. (2026), the manuscript
accompanying openAMUNDSEN-DA v0.9.

## Event sequence

For each configured event, the project runner:

1. propagates the open loop and prior members with openAMUNDSEN;
2. evaluates the configured observation operator;
3. calculates and normalizes likelihood weights;
4. records effective sample size and the resampling decision;
5. materializes the posterior and applies configured rejuvenation; and
6. uses the posterior as the next step's prior.

The current particle weights are scalar at the model-domain or subdomain scale.
Independent subdomain execution is a decomposition strategy and does not add
localization or exchange particles across boundaries.

## Observation families

The event schema supports fractional snow covered area (`scf`), wet snow fraction
(`wet_snow`), wet snow line altitude (`wet_snow_line`), station snow depth
(`station_hs`) and station snow water equivalent (`station_swe`). Each family has
an explicit operator and uncertainty configuration.

## Technical artifacts

- per-event weight CSVs and plots record member likelihood response;
- effective sample size plots record degeneracy and resampling decisions;
- `results/benchmark/` stores configured evaluation cases and scores;
- `results/grids/da_output_grids.nc` stores compact open-loop, ensemble and
  event-analysis fields; and
- `results/run_manifest.json` records hashes, stage state, provenance and outputs.

See [Output data]({{ site.baseurl }}{% link output-data.md %}) for paths and
[Configuration]({{ site.baseurl }}{% link guides/configuration.md %}) for the
event/uncertainty schema.

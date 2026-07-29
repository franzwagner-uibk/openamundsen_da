---
layout: default
title: Data Assimilation Implementation
parent: Reference
nav_order: 4
---

# Data Assimilation Implementation

This page identifies the implemented software stages and artifacts. Detailed
scientific formulation and interpretation are outside its scope.

## Event sequence

For each configured event, the project runner:

1. propagates the open loop and prior members with openAMUNDSEN;
2. evaluates the configured observation operator;
3. adds the event log likelihood to the persisted prior log weight and
   normalizes the resulting posterior weights with log-sum-exp;
4. records effective sample size and the resampling decision;
5. materializes the posterior and applies configured rejuvenation; and
6. carries the normalized weights into the next propagation when resampling is
   skipped, or initializes uniformly weighted children after actual resampling.

The event analysis is the weighted distribution before resampling. The
materialized posterior is the mirrored or resampled member collection used to
initialize propagation. Skipping resampling therefore does not skip data
assimilation: the observation changes the persistent particle weights and all
weighted diagnostics until a later resampling event resets them.

The current particle weights are scalar at the model-domain or subdomain scale.
Independent subdomain execution is a decomposition strategy and does not add
localization or exchange particles across boundaries.

## Observation families

The event schema supports fractional snow covered area (`scf`), wet snow fraction
(`wet_snow`), wet snow line altitude (`wet_snow_line`), station snow depth
(`station_hs`) and station snow water equivalent (`station_swe`). Each family has
an explicit operator and uncertainty configuration.

## Technical artifacts

- every step has `assim/prior_weights.csv` and a versioned ancestry manifest;
- per-event weight CSVs contain prior weight, event log likelihood, posterior
  weight, prior and posterior ESS, threshold and resampling status;
- versioned event, resampling and rejuvenation manifests bind resume behavior
  to inputs, configuration, ancestry and the `keyed-v1` RNG scheme;
- effective sample size plots record degeneracy and resampling decisions;
- `results/benchmark/` stores configured evaluation cases and scores;
- `results/grids/da_output_grids.nc` stores compact open-loop, ensemble and
  event-analysis fields. Ensemble summaries use persistent PF weights, while
  member minima and maxima describe the materialized member collection; and
- `results/run_manifest.json` records hashes, stage state, provenance and outputs.

See [Output data]({{ site.baseurl }}{% link output-data.md %}) for paths and
[Configuration]({{ site.baseurl }}{% link guides/configuration.md %}) for the
event/uncertainty schema.

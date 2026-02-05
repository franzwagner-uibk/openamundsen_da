---
layout: default
title: Data Assimilation Methods
parent: Reference
nav_order: 3
---

# Data Assimilation Methods

openAMUNDSEN-DA implements a bootstrap particle filter (SIR) for snow data assimilation.

## Flow (per assimilation date)
1. **Forecast/prior**: Run openAMUNDSEN ensemble → prior state.
2. **Observation operator H(x)**: Transform model SWE/depth to observation space.
   - SCF: `methods.h_of_x.model_scf.compute_model_scf`
   - Wet snow: `methods.wet_snow.area.compute_model_wet_snow_fraction`
3. **Likelihood**: Compare model vs observations; compute log weights.
4. **Normalize & ESS**: Convert to normalized weights; compute Effective Sample Size (ESS).
5. **Resample** (if ESS < threshold): `methods.pf.resample.resample_from_weights`
6. **Rejuvenate**: Add small perturbations to maintain ensemble spread (`methods.pf.rejuvenate.rejuvenate`).
7. **Propagate**: Use posterior as next prior and continue the season.

## Key modules
- `methods/pf/assimilate_scf.py` — SCF assimilation core.
- `methods/pf/resample.py` — systematic resampling.
- `methods/pf/rejuvenate.py` — posterior jittering.
- `methods/h_of_x/model_scf.py` — SCF observation operator.
- `methods/wet_snow/area.py` — wet-snow observation operator.

## Configuration hooks
- `project.yml` → `data_assimilation`: controls observation products, likelihood params, ESS threshold, rejuvenation settings, landcover mask classes, etc.
- Observation product tags: default `SNOWCOVER`, `WETSNOW` (override via `project.yml`).
- Landcover mask: `data_assimilation.landcover_mask` excludes canopy/built-up/ice classes before computing fractions and likelihoods.

## Outputs
- Assimilation weights per step: `assim/weights.csv`
- ESS timeline: `plots/assim/ess/`
- Posterior member states: `propagation/season_*/step_*/ensembles/posterior/`

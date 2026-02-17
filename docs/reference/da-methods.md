---
layout: default
title: Data Assimilation Methods
parent: Reference
nav_order: 3
---

# Data Assimilation Methods

openAMUNDSEN-DA implements a bootstrap particle filter (SIR) for snow data assimilation.

## Flow (per assimilation date)
1. Forecast/prior: run the openAMUNDSEN ensemble.
2. Observation operator H(x): map model states to observation space.
3. Likelihood: compare model vs observations and compute log weights.
4. Normalize and ESS: convert log weights to normalized weights and compute ESS.
5. Resample (if ESS below threshold): select posterior members.
6. Rejuvenate: perturb posterior members to keep ensemble spread.
7. Propagate: use posterior as next prior and continue within the setup.

## Key modules
- `openamundsen_da.methods.pf.assimilate_scf`
- `openamundsen_da.methods.pf.resample`
- `openamundsen_da.methods.pf.rejuvenate`
- `openamundsen_da.methods.h_of_x.model_scf`
- `openamundsen_da.methods.wet_snow.area`

## Configuration hooks
- `project.yml`: openAMUNDSEN base config and observation product/class mappings (for example `obs.snowcover.*`, `obs.wetsnow.*`).
- `setup.yml`: DA config under `data_assimilation`.
  - `prior_forcing`
  - `h_of_x`
  - `likelihood`
  - `resampling`
  - `rejuvenation`
  - `restart`
  - `landcover_mask`
  - `assimilation_events`

## Outputs
- Per-step weights and indices: `<setup>/steps/step_*/assim/`
- ESS timeline and assimilation plots: `<setup>/plots/assim/`
- Posterior members: `<setup>/steps/step_*/ensembles/posterior/`


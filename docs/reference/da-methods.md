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
- `openamundsen_da.methods.pf.assimilate_fraction`
- `openamundsen_da.methods.pf.assimilate_station`
- `openamundsen_da.methods.pf.resample`
- `openamundsen_da.methods.pf.rejuvenate`
- `openamundsen_da.methods.h_of_x.model_scf`
- `openamundsen_da.methods.wet_snow.area`

## Configuration hooks
- `project.yml`: openAMUNDSEN base config and observation product/class mappings (for example `obs.snowcover.*`, `obs.wetsnow.*`).
- `setup.yml`: data assimilation config under `data_assimilation`.
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
- Station DA diagnostics: `station_diagnostics_station_hs_*.csv` / `station_diagnostics_station_swe_*.csv`
- ESS timeline and assimilation plots: `<setup>/plots/assim/`
- Posterior members: `<setup>/steps/step_*/ensembles/posterior/`

## Prior, Posterior, And Increment Diagnostics
- `open_loop` is the unperturbed baseline and is not resampled by DA.
- `ens_mean_<var>` in `da_output_grids.nc` is the unweighted prior ensemble mean for the current step. It already carries effects from earlier DA steps through warm starts.
- `increment_<var>` is an open-loop departure diagnostic: `ens_mean_<var> - open_loop_<var>`. It is useful for accumulated DA-vs-baseline comparison, but it is not the increment caused by one event.
- `analysis_mean_<var>` is the event-weighted posterior mean on dates with weights.
- `analysis_increment_<var>` is the event-level DA increment: `analysis_mean_<var> - ens_mean_<var>`. Positive values mean the event added snow/water to the prior ensemble mean; negative values mean it removed snow/water.

## Observation families

- `scf` and `wet_snow` use ROI fraction observations against ROI-scale `H(x)` values.
- `station_hs` and `station_swe` use station point observations against model point outputs at station locations.

For the station method, see the dedicated [Station Assimilation]({{ site.baseurl }}{% link guides/station-assimilation.md %}) guide.

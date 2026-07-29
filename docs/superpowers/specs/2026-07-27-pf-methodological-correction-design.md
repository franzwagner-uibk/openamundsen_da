---
published: false
---

# PF methodological correction design

Status: approved by the user on 2026-07-27.

## Objective

Correct openAMUNDSEN-DA into a sequential particle filter that carries
importance weights until actual adaptive resampling, uses reproducible
event-specific process noise, evaluates satellite observations at defensible
times, and reports weighted posterior diagnostics consistently.

The Rofental case study uses its configured fSCA uncertainty layer without
tuning the uncertainty or the ESS threshold to force a resampling event.
High-ESS events remain valid Bayesian reweighting events even when systematic
resampling is unnecessary. The public manuscript term remains particle filter
(PF).

## Sequential PF state

- Every event computes
  `posterior_log_weight = prior_log_weight + log_likelihood`, followed by
  log-sum-exp normalization.
- Step zero starts with uniform weights. Actual resampling creates uniformly
  weighted children. Skipped resampling preserves the posterior weights and
  their member association through propagation.
- Every step stores `assim/prior_weights.csv` with `member_id`, `log_weight`
  and `weight`, plus a versioned ancestry manifest.
- Event weight tables contain the prior log weight and weight, event log
  likelihood, and normalized posterior log weight and weight.
- Insufficient observational support produces a neutral likelihood and an
  explicit skipped-event record. Invalid configured inputs remain hard errors.
- Standalone PF stages require an existing compatible ledger or explicit
  initialization; they cannot silently start a uniform chain in the middle of
  a project.

The canonical event analysis is the weighted distribution before resampling.
The materialized posterior is the empirical mirrored or resampled member set
used for propagation. Posterior statistics, maps, increments and benchmark
scores use canonical weights; raw deterministic traces remain available and
are labeled explicitly.

## Randomness and propagation

- Scientific randomness uses a stable keyed scheme with canonical key
  `keyed-v1 | base_seed | stage | event | member | variable` and a
  cryptographic digest, never Python hashing.
- Initial forcing, rejuvenation variables and systematic resampling use
  independent stage keys. Results are invariant to worker scheduling, member
  ordering and partial rebuilds, while different events receive different
  perturbations.
- Scientific stage seeds are required in YAML. Environment, wall-clock and
  implicit-random seed fallbacks are removed.
- Rejuvenation remains the public term but denotes a fresh process-noise
  propagation, not an MCMC resample-move step. Perturbations are rebuilt from
  unmodified forcing and use the same configured distribution parameters,
  including precipitation `mu_p`.
- The ignored `rebase_open_loop` option is removed from shipped configuration
  and rejected if supplied.

## Satellite time and uncertainty

- Assimilation events accept optional ISO-8601 `observation_time`; date-only
  selection remains valid when exactly one scene exists for that product and
  date.
- A tracked offline manifest records scene/product identity, UTC acquisition
  time, provenance source and time quality. Derived summaries and event
  artifacts inherit those fields.
- Timestamp precedence is CF coordinate, raster metadata, sidecar metadata,
  configured filename parser, tracked manifest, then midnight fallback.
  Midnight fallback emits a warning and records
  `time_quality=fallback_midnight`.
- Multiple same-date scenes require `observation_time`. Exact timestamps map
  to the unique nearest model timestep; ties and offsets beyond half a model
  timestep fail.
- Satellite operators use instantaneous snow depth and liquid-water fields at
  the same matched model time.
- Rofental fSCA uses `sigma_mode: uncertainty_layer` and
  `aggregate_metric: unc_mean`, with
  `sigma=max(min_sigma, unc_mean/100)`. This is documented as an effective,
  uncalibrated comparison-error sigma. Invalid uncertainty data fail without a
  fixed-sigma fallback.
- WSLA retains its 100 m Gaussian uncertainty and its centered three-band,
  interpolated 50 percent crossing operator. Its support coverage is computed
  on the model grid and constrained to `[0, 1]`.

## Provenance, validation and compatibility

- Resume requires versioned manifests matching configuration, inputs,
  weight-ledger ancestry, RNG scheme and relevant hashes.
- Existing v0.9.3 chains are view-only and cannot resume under the corrected
  method.
- Configuration parsing fails on missing scientific seeds, unsupported
  options and malformed likelihood settings rather than substituting hidden
  defaults.
- Public additions are optional event `observation_time`, optional product
  acquisition-manifest configuration, the versioned prior-weight ledger and
  expanded event-weight and RNG provenance.

## Verification and publication workflow

- Unit tests cover cumulative weights across skipped resampling, uniform reset
  after actual resampling, neutral support gates, numerical failures, keyed RNG
  reproducibility, satellite time resolution, strict uncertainty validation,
  weighted diagnostics and WSLA coverage.
- CI lint, unit and integration wrappers must pass before the canonical ES30
  Rofental rerun.
- The Rofental run is accepted for methodological integrity and reproducibility,
  not score improvement or the occurrence of fSCA resampling.
- Generated example contracts and manuscript assets are refreshed only from
  the completed candidate run.
- Manuscript changes require a separate exact-patch approval before editing
  `template.tex`. They will correct the recursive equation, weighted/materialized
  terminology, process-noise description, fSCA uncertainty interpretation,
  actual ESS outcomes, WSLA operator wording and acquisition-time semantics.

## Explicit exclusions

This work does not add localization, AR(1) forcing errors, parameter
perturbations, MCMC moves, multivariate likelihoods, a matched stochastic
no-DA control or a new sensitivity study. It does not change the existing
nearest-neighbor binary spatial-support approximation or add manuscript text
about that approximation. Future-work items already stated in the manuscript
remain outside scope.

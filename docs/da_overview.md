# Data Assimilation Overview (openamundsen_da)

Purpose
- Seasonal prediction of snow cover using an ensemble snow model (openAMUNDSEN) with a particle filter. Repeated prediction–update cycles assimilate satellite SCA/SCF over the season.

Prediction–Update Cycle
- Initialization: Build prior ensemble by perturbing parameters/forcing; lay out member directories and config overlays.
- Prediction: For each member, run openAMUNDSEN over the next window; produce outputs required for SCF derivation.
- Update: Observation processing (satellite SCA → SCF), model SCF derivation H(x), Gaussian likelihood per member, importance weighting (normalize), resampling (avoid degeneracy), rejuvenation (noise to parameters). Posterior → next prior.

Repository Modules (current/intended)
- core/launch.py: Orchestrates ensemble runs (fan‑out, progress, logging).
- core/runner.py: Per‑member execution (merged config, OA run, manifest, per‑member logs).
- core/config.py: Merge project/season/step YAML; inject member paths; apply CLI `log_level` to OA.
- io/paths.py: Path discovery helpers; member/results layout.
- methods/ (planned):
  - obs/: satellite SCA ingestion/QC → SCF time series for assimilation dates.
  - h_of_x/: derive model SCF from OA outputs to match observation operator.
  - pf/: likelihood, weights, resampling, rejuvenation primitives.

Configuration (expected keys)
- ensemble: size, member naming, seeds; perturbation specs for parameters/forcing; assimilation dates/windows.
- likelihood: sigma/variance, masks/quality rules, temporal/spatial aggregation options.
- resampling: effective size threshold, algorithm (multinomial/systematic/stratified).
- rejuvenation: noise model and magnitudes per parameter group.
- io: project/season/step roots; member meteo/results; observation data sources.

Data & I/O Conventions
- Member layout: `<step_dir>/ensembles/<prior|posterior>/<member_xxx>/` with `meteo/`, `logs/member.log`, `results/`.
- Manifest: `results/member_run.json` with status, timings, error for post‑mortem.
- Observation cache (planned): preprocessed SCF arrays aligned to assimilation dates.

Logging Strategy
- Parent summarizes progress only; workers write to `member.log` (one file per member). CLI `--log-level` controls both parent and OA via merged config.

Validation & QA (suggested)
- Unit tests for H(x) and likelihood on small fixtures; effective sample size monitoring; reproducibility via seeds; spot‑checks against known seasons.

Open Questions / TODOs
- Define concrete perturbation distributions and parameter sets.
- Decide on resampling algorithm default and rejuvenation noise per parameter.
- Specify observation QC and spatiotemporal aggregation rules for SCF.

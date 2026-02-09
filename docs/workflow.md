---
layout: default
title: Workflow
nav_order: 4
---

# Data Assimilation Workflow

{: .no_toc }

Understanding the particle filter data assimilation cycle.
{: .fs-6 .fw-300 }

<details markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

## Overview

The openamundsen_da framework implements a sequential particle filter for snow data assimilation. The workflow cycles through prior generation, forecast propagation, and observation-based updates.

![Data Assimilation Architecture]({{ site.baseurl }}/assets/images/DataAssimilation_Design%20_DOCS%20_Architecture.drawio.png)

**Key components**:

- **Prior Generation** (orange): Perturb meteorological forcing to create ensemble input
- **Forecast/Propagation** (purple): Run openAMUNDSEN for each ensemble member
- **Update Cycle** (blue): Observation processing, likelihood computation, resampling, rejuvenation
- **Configuration** (yellow): `project.yml` controls openAMUNDSEN and DA settings, `season.yml` defines assimilation dates

![Data Assimilation Experiment Cycle]({{ site.baseurl }}/assets/images/Particle_Filter%20_DOCS.drawio.png)

---

## Prior Ensemble Generation

### Meteorological Forcing Perturbation

**Temperature**: Additive Gaussian noise

```
T_perturbed = T_original + ε_T,  ε_T ~ N(0, σ_T²)
```

**Precipitation**: Multiplicative log-normal noise

```
P_perturbed = P_original × exp(ε_P),  ε_P ~ N(μ_P, σ_P²)
```

**Command**:

```bash
docker compose run --rm oa \
  python -m openamundsen_da.core.prior_forcing \
  --input-meteo-dir /data/meteo \
  --project-dir /data \
  --step-dir /data/propagation/season_2019-2020/step_01_*
```

Optional flags: `--overwrite`, `--log-level <LEVEL>`

---

## Model Execution

### Parallel Ensemble Runs

```bash
docker compose run --rm oa \
  python -m openamundsen_da.core.launch \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020 \
  --step-dir /data/propagation/season_2019-2020/step_01_* \
  --ensemble prior \
  --max-workers 8
```

Optional flags: `--overwrite`, `--state-pattern <glob>`, `--log-level <LEVEL>`

**Parallelism**: The `--max-workers` value is an upper bound. The launcher clamps to `min(max_workers, CPUs visible, #members)`. An `open_loop` (unperturbed, unassimilated baseline) is run alongside the ensemble and carried through all steps for reference/plots.

### State Management

Warm start uses the model state saved at the end of each step via `state_pointer.json`:

```json
{
  "path": "/abs/or/rel/path/to/model_state.pickle.gz"
}
```

---

## Observation Processing

### Snow cover (GeoTIFF/NetCDF → `scf_summary.csv`)

```bash
docker compose run --rm oa \
  oa-da-snowcover \
  --input-dir /data/obs/snowcover \
  --season-label season_2019-2020 \
  --project-dir /data
```

Classes are read from `obs.snowcover.classes` in `project.yml` (defaults: valid 0–100, cloud 205, water 210, nodata 255). The land-cover mask from `project.yml` is applied to observations automatically.

### Wet snow (categorical rasters → `wet_snow_summary.csv`)

```bash
docker compose run --rm oa \
  oa-da-wetsnow \
  --input-dir /data/obs/wetsnow \
  --season-label season_2019-2020 \
  --project-dir /data
```

Wet/valid/exclude classes come from `obs.wetsnow.classes` (defaults: wet [1,2], valid [1,2,3,4,255], exclude [5,6]). ROI and land-cover masking mirror the snow-cover workflow.

By default, both summaries are written to `/data/obs/summaries/<season-label>/scf_summary.csv` and `wet_snow_summary.csv`. Override with `--output-root` if you want a different location.

### Per-step obs CSVs

```bash
docker compose run --rm oa oa-da-scf --season-dir /data/propagation/season_2019-2020 --overwrite
docker compose run --rm oa oa-da-wetsnow-season --season-dir /data/propagation/season_2019-2020 --overwrite
```

Outputs: `step_*/obs/obs_scf_<PRODUCT>_YYYYMMDD.csv` and `obs_wet_snow_<PRODUCT>_YYYYMMDD.csv`, with product tags resolved from `project.yml` (`SNOWCOVER`/`WETSNOW` by default).

### Land-Cover Masking

Configured in `project.yml`:

```yaml
data_assimilation:
  landcover_mask:
    # Classes: 1 rock, 2 ice, 3 water, 4 grassland, 5 shrubland, 6 farmland,
    # 7 transitional, 8 deciduous 30-60, 9 deciduous 60-100, 10 mixed,
    # 11 coniferous 30-60, 12 coniferous 60-100, 13 built-up.
    enabled: true
    classes_to_exclude: [2, 8, 9, 10, 11, 12, 13]
```

Excluded land-cover classes are removed from both observations and model-derived fractions. A warning is logged if >50% of the ROI would be excluded; 100% exclusion fails.

---

## Data Assimilation

### H(x) Forward Operator

Maps model state (snow depth or SWE) to observation space (SCF).

**Methods**:

1. **Depth Threshold**:

   ```
   SCF = 1  if HS > h0
   SCF = 0  otherwise
   ```

2. **Logistic** (smooth):
   ```
   SCF = 1 / (1 + exp(-k × (HS - h0)))
   ```

**Configuration** (in `project.yml`):

```yaml
data_assimilation:
  h_of_x:
    method: depth_threshold # or "logistic"
    variable: hs # or "swe"
    params:
      h0: 0.01
      k: 80
```

### Likelihood Calculation

Gaussian likelihood function:

```
w_i ∝ exp(-0.5 × ((y_obs - H(x_i)) / σ_obs)²)
```

Weights are normalized: `w_i = w_i / Σ(w_j)`

### Effective Sample Size (ESS)

```
ESS = 1 / Σ(w_i²)
```

- ESS = N: All weights equal (no information from obs)
- ESS = 1: One particle dominates (particle degeneracy)
- ESS < threshold → Trigger resampling

---

## Ensemble Update

### Systematic Resampling

**Configuration**:

```yaml
data_assimilation:
  resampling:
    algorithm: systematic
    ess_threshold_ratio: 0.5 # Resample if ESS < 0.5 × N
    seed: 42
```

**Behavior**:

- If `ESS ≥ threshold`: Skip resampling, mirror prior → posterior
- If `ESS < threshold`: Resample

### Rejuvenation

After resampling, ensemble spread is reduced. Rejuvenation adds noise to maintain diversity.

Perturbations are always applied relative to the open loop forcing:

```
forcing_new = open_loop_forcing + new_perturbation
```

**Configuration**:

```yaml
data_assimilation:
  rejuvenation:
    sigma_t: 0.2 # Additive temperature noise (deg C)
    sigma_p: 0.2 # Lognormal sigma for precip factor (mu=0)
```

If rejuvenation sigmas are not set, they fall back to prior_forcing sigmas.

### State Propagation

Copy posterior states + perturbed forcing to next step's prior:

```
step_N/ensembles/posterior/member_i/ → step_N+1/ensembles/prior/member_j/
```

---

## Season Pipeline

The season pipeline automates all phases:

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.season \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020 \
  --max-workers 8 \
  --monitor-perf
```

Optional: `--overwrite`, `--live-plots`, `--log-level <LEVEL>` (`--live-plots` enables in-run plotting; default is off and plots run once after completion)

**Pipeline steps** (per assimilation cycle):

1. Generate prior forcing
2. Run prior ensemble
3. Compute model H(x) (SCF/wet-snow)
4. Assimilate observations → weights
5. Check ESS → resample if needed
6. Rejuvenate → next prior
7. Generate plots
8. Repeat for next step

**Outputs**:

- Per-step runs in `<step>/ensembles/{prior,posterior}`
- Weights and indices in `<step>/assim/`
- Rejuvenated next-step prior with `state_pointer.json`
- Season plots under `<season_dir>/plots/{forcing,results}`

---

## State cleanup

- Automatic: `data_assimilation.restart.cleanup_after_season: true` (default) removes member state pickle files after a successful season to save disk space.
- Manual: run the cleanup CLI to delete state files even if automatic cleanup is disabled.

Clean all seasons:

```powershell
oa-da-clean-season --project-dir /data/your_project --all-seasons --log-level INFO
```

Clean one season:

```powershell
oa-da-clean-season --project-dir /data/your_project --season-dir /data/your_project/propagation/season_YYYY-YYYY --log-level INFO
```

Only state pickle files are removed; `state_pointer.json` files are left in place.

---

## Configuration Reference

### Likelihood Settings

```yaml
data_assimilation:
  likelihood:
    scf:
      obs_sigma: 0.10
      use_binomial: true
      sigma_floor: 0.05
      sigma_cloud_scale: 0.10
      min_sigma: 0.03
    wet_snow:
      obs_sigma: 0.15
      use_binomial: false
```

### Warm Start Settings

```yaml
data_assimilation:
  restart:
    dump_state: true
    state_pattern: model_state.pickle.gz
```

---

## Next Steps

- [Configuration Guide]({{ site.baseurl }}{% link guides/configuration.md %}) - Detailed configuration reference
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments/index.md %}) - Step-by-step experiment setup
- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) - Command-line tools

---

## References

- Barella, R., Marin, C., Gianinetto, M., and Notarnicola, C. (2022). A novel approach to high resolution snow cover fraction retrieval in mountainous regions. IGARSS 2022 - IEEE International Geoscience and Remote Sensing Symposium, 3856-3859. https://doi.org/10.1109/IGARSS46834.2022.9884177.
- Nagler, T., Rott, H., Ripper, E., Bippus, G., and Hetzenecker, M. (2016). Advancements for snowmelt monitoring by means of Sentinel-1 SAR. Remote Sensing, 8(4), 348. https://doi.org/10.3390/rs8040348.

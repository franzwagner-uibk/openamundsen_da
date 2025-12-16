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

## Workflow Overview

The openamundsen_da framework implements a sequential particle filter for snow data assimilation. The workflow consists of 7 main phases that repeat for each assimilation cycle.

### Architecture

![Data Assimilation Architecture]({{ site.baseurl }}/assets/images/DataAssimilation_Design%20_DOCS%20_Architecture.drawio.png)
*Figure 1: Data assimilation architecture overview. The workflow cycles through prior generation (orange), forecast propagation (purple), and the update cycle (blue). Configuration files (yellow) control the system parameters, and warm-start enables continuous state transfer between cycles.*

**Key components**:
- **Prior Generation** (orange): Perturb meteorological forcing to create ensemble input
- **Forecast/Propagation** (purple): Run openAMUNDSEN for each ensemble member
- **Update Cycle** (blue):
  - Load and preprocess satellite observations (SCF, wet snow)
  - Apply forward operator H(x) to map model states to observation space
  - Compute Gaussian likelihood comparing model predictions to observations
  - Calculate importance weights and normalize
  - Systematic resampling to select posterior ensemble
  - Rejuvenation to re-perturb forcing and maintain ensemble spread
- **Configuration** (yellow): `project.yml` controls openAMUNDSEN and DA settings, `season.yml` defines assimilation dates

### Ensemble Update Cycle Example

![Data Assimilation Experiment Cycle]({{ site.baseurl }}/assets/images/Particle_Filter%20_DOCS.drawio.png)
*Figure 2: Ensemble evolution throughout a snow season. The upper panel shows Snow Cover Area (SCA) trajectories for each ensemble member. Satellite observation times are marked with icons. Lower panels illustrate the assimilation cycle: prior ensemble generation, propagation, observation-based correction (importance weighting, resampling, rejuvenation), and posterior state propagation.*

**Interpretation**:
- **Initialization**: Ensemble spread created by perturbing meteorological forcings
- **Propagation**: Model uncertainty grows as ensemble members diverge
- **Update** (satellite icons): Observations constrain the ensemble, reducing spread
- **Prior states** (left distribution panels): Ensemble before assimilation shows wide spread
- **Posterior states** (right distribution panels): Ensemble after assimilation is concentrated around observations
- **Cycle repeats**: Posterior becomes the prior for the next step, continuously improving the forecast

### Workflow Phases

The complete workflow consists of these phases:

```mermaid
graph TD
    A[1. Initialization & Setup] --> B[2. Prior Ensemble Generation]
    B --> C[3. Model Execution]
    C --> D[4. Observation Processing]
    D --> E[5. Data Assimilation]
    E --> F[6. Ensemble Update]
    F --> G[7. Visualization & Analysis]
    G --> H{More steps?}
    H -->|Yes| B
    H -->|No| I[End]
```

---

## Phase 1: Initialization & Setup

### Project Structure Definition

Create the required directory structure:
- `env/` - ROI and glacier masks
- `meteo/` - Meteorological forcing data
- `obs/` - Satellite observations
- `propagation/` - Ensemble runs (auto-created)

### Season Skeleton Creation

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.season_skeleton \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020
```

Creates `step_XX_*/` directories with date boundaries aligned to assimilation dates.

### Configuration

Three configuration levels:
1. **project.yml** - Project-wide settings (ensemble size, DA parameters)
2. **season.yml** - Season dates and assimilation events
3. **step_XX.yml** - Step-specific settings (auto-generated)

---

## Phase 2: Prior Ensemble Generation

### Meteorological Forcing Perturbation

**Temperature**: Additive Gaussian noise
```
T_perturbed = T_original + Îµ_T,  Îµ_T ~ N(0, Ïƒ_TÂ²)
```

**Precipitation**: Multiplicative log-normal noise
```
P_perturbed = P_original Ã— exp(Îµ_P),  Îµ_P ~ N(0, Ïƒ_PÂ²)
```

**Implementation**:
```bash
docker compose run --rm oa \
  python -m openamundsen_da.core.prior_forcing \
  --input-meteo-dir /data/meteo \
  --project-dir /data \
  --step-dir /data/propagation/season_2019-2020/step_01_*
```

### Open Loop Reference

An unperturbed baseline run using original meteorological data for comparison.

### Ensemble Members

N independent members (member_001, member_002, ..., member_N) with independent perturbations.

---

## Phase 3: Model Execution

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

**Features**:
- Parallel execution using `multiprocessing`
- Worker count: `min(max_workers, CPU_count, N_members)`
- Warm-start capability via `state_pointer.json`

### State Management

Each member saves its state at step boundaries:
```json
{
  "state_file": "results/state_20191122_000000.nc",
  "timestamp": "2019-11-22T00:00:00"
}
```

### Results Output

Per member:
- NetCDF grids (snow.nc, meteo.nc)
- Point CSV time series
- Optional model SCF time series

---

## Phase 4: Observation Processing

### MODIS MOD10A1 Preprocessing

```bash
docker compose run --rm oa \
  python -m openamundsen_da.observer.mod10a1_preprocess \
  --input-dir /data/obs/MOD10A1_61_HDF \
  --season-label season_2019-2020 \
  --project-dir /data
```

**Steps**:
1. HDF â†’ GeoTIFF conversion
2. QA masking (cloud-free pixels only)
3. Reprojection to study area CRS
4. ROI clipping
5. NDSI thresholding â†’ binary snow mask
6. SCF calculation per ROI

**Output**: `obs/season_2019-2020/scf_summary.csv`

### Sentinel-2 FSC (Snowflake)

```bash
docker compose run --rm oa \
  python -m openamundsen_da.observer.snowflake_fsc \
  --input-dir /data/obs/FSC_snowflake \
  --season-label season_2019-2020 \
  --project-dir /data
```

### Sentinel-1 Wet Snow

```bash
docker compose run --rm oa \
  python -m openamundsen_da.observer.satellite_wet_snow_s1 \
  --input-dir /data/obs/WSM_S1 \
  --season-label season_2019-2020 \
  --project-dir /data
```

**WSM Classes**:
- 110: Wet snow
- 125: Dry/no snow
- 200: Radar shadow (excluded)
- 210: Water (excluded)

### Glacier Masking

When enabled (`project.yml`):
```yaml
data_assimilation:
  glacier_mask:
    enabled: true
    path: env/glaciers.gpkg
```

Firn/ice areas are excluded from obs-model comparisons to ensure consistency (seasonal snow model vs. observations including glaciers).

---

## Phase 5: Data Assimilation (Particle Filter)

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
   SCF = 1 / (1 + exp(-k Ã— (HS - h0)))
   ```

**Configuration** (in `project.yml`):
```yaml
data_assimilation:
  h_of_x:
    variable: hs      # 'hs' or 'swe'
    method: logistic  # 'depth_threshold' or 'logistic'
    h0: 0.05         # Threshold (m)
    k: 50.0          # Steepness (logistic only)
```

### Likelihood Weight Calculation

Gaussian likelihood function:
```
w_i âˆ exp(-0.5 Ã— ((y_obs - H(x_i)) / Ïƒ_obs)Â²)
```

where:
- `y_obs`: Observed SCF
- `H(x_i)`: Model SCF for member i
- `Ïƒ_obs`: Observation error std (from config)

**Normalization**:
```
w_i = w_i / Î£(w_j)
```

### Effective Sample Size (ESS)

```
ESS = 1 / Î£(w_iÂ²)
```

- ESS = N: All weights equal (no information from obs)
- ESS = 1: One particle dominates (particle degeneracy)
- ESS < threshold â†’ Trigger resampling

---

## Phase 6: Ensemble Update

### Systematic Resampling

**Algorithm**:
1. Generate systematic samples: `u_i = (i + U) / N`, where `U ~ Uniform(0,1)`
2. Map samples to cumulative weight distribution
3. Select members according to mapped indices
4. Duplicate high-weight members, discard low-weight members

**Configuration**:
```yaml
data_assimilation:
  resampling:
    algorithm: systematic
    ess_threshold_ratio: 0.5  # Resample if ESS < 0.5 Ã— N
    seed: 42
```

**Behavior**:
- If `ESS â‰¥ threshold`: Skip resampling, mirror prior â†’ posterior
- If `ESS < threshold`: Resample

### Rejuvenation

After resampling, ensemble spread is reduced (identical states). Rejuvenation adds noise to maintain spread.

**Rebase mode** (default):
Perturbations applied relative to open loop:
```
forcing_new = open_loop_forcing + new_perturbation
```

**Configuration**:
```yaml
data_assimilation:
  rejuvenation:
    sigma_t: 0.2  # Usually smaller than prior
    sigma_p: 0.2
```

### State Propagation

Copy posterior states + perturbed forcing to next step's prior:
```
step_N/ensembles/posterior/member_i/ â†’ step_N+1/ensembles/prior/member_j/
```

where `j = indices[i]` from resampling.

---

## Phase 7: Visualization & Analysis

### Forcing Plots

Per-station temperature and precipitation time series showing:
- Open loop
- All ensemble members
- Ensemble mean Â± spread

### Results Plots

SWE, snow depth, SCF time series:
- Ensemble mean
- 90% envelope (5th-95th percentiles)
- Open loop
- Observations (when available)

### Weight & Residual Plots

Per assimilation date:
- Normalized particle weights (bar plot)
- Observation-model residuals (histogram)
- ESS value

### ESS Timeline

Season-wide ESS evolution:
- ESS vs. time
- ESS threshold (horizontal line)
- Resampling events (markers)

### Performance Monitoring

When enabled (`--monitor-perf`):
- CPU usage
- Memory (RSS, system)
- Disk usage
- ETA estimation

**Output**: `plots/perf/season_perf.png` (live-updated)

---

## Automated Season Pipeline

The season pipeline automates all phases:

```bash
docker compose run --rm oa \
  python -m openamundsen_da.pipeline.season \
  --project-dir /data \
  --season-dir /data/propagation/season_2019-2020 \
  --max-workers 8 \
  --monitor-perf
```

**Pipeline steps** (per assimilation cycle):
1. Generate prior forcing
2. Run prior ensemble
3. Compute model H(x) (SCF/wet-snow)
4. Assimilate observations â†’ weights
5. Check ESS â†’ resample if needed
6. Rejuvenate â†’ next prior
7. Generate plots
8. Repeat for next step

---

## Best Practices

### Ensemble Size

- **Small domains** (< 100 kmÂ²): 20-30 members
- **Medium domains**: 30-50 members
- **Large domains** (> 500 kmÂ²): 50-100 members

Trade-off: More members = better posterior but higher computational cost.

### Perturbation Magnitudes

**Prior**:
- `Ïƒ_T`: 1.0-2.0 K (typical)
- `Ïƒ_P`: 0.15-0.25 (15-25% uncertainty)

**Rejuvenation**: Use smaller values (0.1-0.2) to avoid over-perturbation.

### ESS Threshold

- `ess_threshold_ratio = 0.5`: Resample when ESS < 50% of ensemble size
- Lower threshold: Less frequent resampling but risk of degeneracy
- Higher threshold: More resampling but may lose diversity

### Observation Error

Depends on data source:
- MODIS MOD10A1: Ïƒ_obs â‰ˆ 0.1-0.15
- Sentinel-2 FSC: Ïƒ_obs â‰ˆ 0.05-0.10
- In-situ: Ïƒ_obs â‰ˆ 0.05

Configure in `project.yml`:
```yaml
data_assimilation:
  observation_error:
    scf: 0.10
    wet_snow: 0.15
```

---

## Next Steps

- [Configuration Guide]({{ site.baseurl }}{% link guides/configuration.md %}) - Detailed configuration reference
- [Running Experiments]({{ site.baseurl }}{% link guides/experiments.md %}) - Step-by-step experiment setup
- [CLI Reference]({{ site.baseurl }}{% link guides/cli.md %}) - Command-line tools


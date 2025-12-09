---
layout: default
title: Data Assimilation Methods
parent: Reference
nav_order: 3
---

# Data Assimilation Methods
{: .no_toc }

Detailed particle filter implementation and mathematical formulation.
{: .fs-6 .fw-300 }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Sequential Particle Filter

openamundsen_da implements a **bootstrap particle filter** (also called Sequential Importance Resampling, SIR) for snow data assimilation.

---

## Mathematical Formulation

### State Space Model

**State vector** at time `t`:
```
x_t = [HS, SWE, T_surface, LWC, ...]  # Snow state variables
```

**Evolution model** (forecast):
```
x_t = f(x_{t-1}, u_t, w_t)
```
where:
- `f`: openAMUNDSEN model
- `u_t`: Meteorological forcing (perturbed)
- `w_t`: Model error (captured by forcing perturbations)

**Observation model**:
```
y_t = H(x_t) + v_t
```
where:
- `H(x)`: Forward operator (e.g., snow depth → SCF)
- `v_t ~ N(0, R)`: Observation error

---

## Particle Representation

The ensemble represents the posterior PDF as N weighted particles:

```
P(x_t | y_{1:t}) ≈ Σ w_t^(i) δ(x_t - x_t^(i))
```

where:
- `x_t^(i)`: State of particle i at time t
- `w_t^(i)`: Normalized weight of particle i
- `Σ w_t^(i) = 1`

---

## Algorithm Steps

### 1. Initialization (t=0)

Draw N particles from prior:
```
x_0^(i) ~ P(x_0)  for i=1,...,N
w_0^(i) = 1/N
```

**Implementation**:
- Initial states from spin-up run or cold start
- Perturbed forcing creates ensemble spread

### 2. Forecast (Prior)

Propagate each particle through the model:
```
x_t^(i) = f(x_{t-1}^(i), u_t^(i), w_t^(i))
```

**Implementation**:
```python
# openamundsen_da.core.launch
for member in ensemble:
    run_openamundsen(
        state_file=previous_state[member],
        forcing=perturbed_forcing[member],
        output=results[member]
    )
```

### 3. Weight Update (Likelihood)

When observation `y_t` is available, compute likelihood for each particle:

```
w_t^(i) = w_{t-1}^(i) × p(y_t | x_t^(i))
```

For Gaussian observation error:
```
p(y_t | x_t^(i)) = (1 / √(2π σ_obs²)) × exp(-0.5 × ((y_t - H(x_t^(i))) / σ_obs)²)
```

Proportional form (sufficient for relative weights):
```
w_t^(i) ∝ exp(-0.5 × ((y_t - H(x_t^(i))) / σ_obs)²)
```

**Normalize**:
```
w_t^(i) = w_t^(i) / Σ w_t^(j)
```

**Implementation**:
```python
# openamundsen_da.methods.pf.assimilate
residuals = obs_scf - model_scf  # y_t - H(x_t^(i))
weights = np.exp(-0.5 * (residuals / sigma_obs)**2)
weights /= weights.sum()
```

### 4. Effective Sample Size (ESS)

Measure of weight degeneracy:
```
ESS = 1 / Σ (w_t^(i))²
```

**Range**: `1 ≤ ESS ≤ N`
- `ESS = N`: Uniform weights (no particle degeneracy)
- `ESS = 1`: One particle has weight 1 (complete degeneracy)

### 5. Resampling

If `ESS < threshold × N`, resample particles:

**Systematic Resampling Algorithm**:

1. Generate starting point: `u ~ Uniform(0, 1/N)`
2. Generate systematic samples:
   ```
   u_i = (i + u) / N  for i=0,1,...,N-1
   ```
3. Compute cumulative weight distribution:
   ```
   c_j = Σ_{k=0}^{j} w_t^(k)
   ```
4. Select particles: For each `u_i`, find `j` such that `c_{j-1} < u_i ≤ c_j`
5. Assign new ensemble:
   ```
   x_t^(i) ← x_t^(j[i])  # Copy selected particles
   w_t^(i) ← 1/N         # Reset weights
   ```

**Implementation**:
```python
# openamundsen_da.methods.pf.resampling
def systematic_resampling(weights, N, seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1/N)

    # Cumulative weights
    c = np.cumsum(weights)

    # Systematic samples
    u_samples = (np.arange(N) + u) / N

    # Find indices
    indices = np.searchsorted(c, u_samples)

    return indices
```

### 6. Rejuvenation (Optional)

After resampling, add jitter to restore ensemble spread:

```
u_t^(i) = u_t^(i) + ε^(i)
```

where `ε^(i)` are small perturbations to forcing.

**Implementation**: Same as prior forcing perturbation, but with smaller `sigma_t`, `sigma_p`.

---

## Forward Operators H(x)

### SCF from Snow Depth

#### Depth Threshold

Binary step function:
```
H(HS) = 1  if HS > h0
        0  otherwise
```

**Characteristics**:
- Simple, fast
- Discontinuous (can cause weight concentration)
- Typical `h0`: 0.01-0.05 m

#### Logistic Function

Smooth sigmoid transition:
```
H(HS) = 1 / (1 + exp(-k × (HS - h0)))
```

**Parameters**:
- `h0`: Midpoint threshold (m)
- `k`: Steepness (larger = steeper)

**Characteristics**:
- Differentiable (smooth)
- Realistic partial snow cover representation
- Typical values: `h0=0.05`, `k=50`

**Visualization**:

```python
import numpy as np
import matplotlib.pyplot as plt

hs = np.linspace(0, 0.15, 100)
h0, k = 0.05, 50

scf = 1 / (1 + np.exp(-k * (hs - h0)))

plt.plot(hs, scf)
plt.xlabel('Snow depth (m)')
plt.ylabel('SCF')
plt.title('Logistic H(x)')
plt.grid()
```

### Wet Snow from LWC

Binary classification:
```
H(LWC) = 1  if LWC/SWE > threshold  (wet snow)
         0  otherwise                (dry snow)
```

Typical threshold: 1-3% of SWE

---

## Likelihood Functions

### Gaussian (SCF)

For continuous observations (SCF ∈ [0,1]):

```
p(y | H(x)) = N(y | H(x), σ_obs²)
```

**Log-likelihood** (for numerical stability):
```
log p(y | H(x)) = -0.5 × log(2π σ_obs²) - 0.5 × ((y - H(x)) / σ_obs)²
```

### Bernoulli (Wet Snow)

For binary observations (wet=1, dry=0):

```
p(y | H(x)) = H(x)^y × (1 - H(x))^(1-y)
```

---

## Particle Degeneracy

### Problem

Over time, most particles have negligible weights:
- ESS → 1
- Only one particle represents posterior
- Loss of ensemble diversity

### Causes

1. **Observation error too small**: Likelihood very peaked
2. **Model-observation mismatch**: Few particles near observations
3. **Insufficient ensemble size**: Not enough particles to cover space

### Solutions

1. **Resampling**: Duplicate high-weight particles
2. **Rejuvenation**: Add noise after resampling
3. **Adaptive observation error**: Inflate `σ_obs` if ESS too low
4. **Larger ensemble**: More particles → better coverage

---

## Adaptive Strategies (Future)

### Localization

For large spatial domains, localize weight updates:
```
w(x) = w_prior(x) × [p(y | H(x))]^α(x)
```

where `α(x)` decreases with distance from observation.

### Inflation

Inflate ensemble spread if variance too small:
```
x^(i) = x̄ + λ × (x^(i) - x̄)
```

where `λ > 1` (e.g., 1.05).

---

## Computational Complexity

| Operation | Complexity | Notes |
|:----------|:-----------|:------|
| Forecast | O(N × T_model) | N ensemble runs |
| Weight update | O(N × n_obs) | Evaluate H(x) for each particle |
| ESS | O(N) | Sum of squared weights |
| Resampling | O(N log N) | Searchsorted in cumulative weights |
| Rejuvenation | O(N × n_meteo) | Perturb forcing files |

**Dominant cost**: Forecast (model runs)
- Parallelizable across N members
- Scales with domain size and timestep

---

## References

### Particle Filters

- Doucet, A., et al. (2001). *Sequential Monte Carlo Methods in Practice*. Springer.
- Arulampalam, M.S., et al. (2002). *A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking*. IEEE Transactions on Signal Processing.

### Snow Data Assimilation

- Margulis, S.A., et al. (2019). *A Landsat-Era Sierra Nevada Snow Reanalysis (1985–2015)*. Journal of Hydrometeorology.
- Griessinger, N., et al. (2019). *Assimilation of snow cover derived from MODIS into a snow model*. The Cryosphere.
- Huang, C., et al. (2017). *Assimilation of snow cover and snow depth into a snow model*. Water Resources Research.

### Forward Operators

- Aalstad, K., et al. (2018). *Ensemble-based assimilation of fractional snow-covered area satellite retrievals*. The Cryosphere.
- De Lannoy, G.J.M., et al. (2012). *Assimilating satellite-based snow depth and snow cover products*. Journal of Hydrometeorology.

---

## Next Steps

- [Package Structure]({{ site.baseurl }}{% link reference/package-structure.md %}) - Code organization
- [Configuration Reference]({{ site.baseurl }}{% link guides/configuration.md %}) - Parameter tuning
- [Workflow Guide]({{ site.baseurl }}{% link workflow.md %}) - Conceptual overview

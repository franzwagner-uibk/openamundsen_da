from __future__ import annotations
"""
openamundsen_da.util.stats

Compact statistical helpers used across DA modules.

Includes:
- Prior forcing perturbation samplers
  - Temperature offset: Normal(0, sigma_t^2)
  - Precipitation factor: LogNormal(mu_p, sigma_p^2)
- Core math utilities
  - Logistic sigmoid with numerical stability
"""

from numpy.random import Generator
import numpy as np
import pandas as pd


def sample_delta_t(rng: Generator, sigma_t: float) -> float:
    """Sample an additive temperature offset ΔT ~ N(0, sigma_t^2)."""
    return float(rng.normal(0.0, sigma_t))


def sample_precip_factor(rng: Generator, mu_p: float, sigma_p: float) -> float:
    """Sample a multiplicative precipitation factor f_p ~ LogNormal(mu_p, sigma_p^2)."""
    return float(rng.lognormal(mean=mu_p, sigma=sigma_p))


def sigmoid(x):
    """Numerically stable logistic sigmoid 1 / (1 + exp(-x)).

    Accepts numpy arrays or scalars; returns same shape.
    """
    # For large negative x, exp(-x) can overflow; use np.where split
    x = np.asarray(x)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


# ---- Likelihood and PF helpers ---------------------------------------------

def gaussian_logpdf(residual: np.ndarray, sigma: np.ndarray | float) -> np.ndarray:
    """Elementwise log N(0, sigma^2) evaluated at residual.

    residual and sigma can be broadcastable arrays or scalars.
    Returns an array of log-likelihoods.
    """
    r = np.asarray(residual, dtype=float)
    s = np.asarray(sigma, dtype=float)
    return -0.5 * (np.log(2.0 * np.pi) + 2.0 * np.log(s) + (r * r) / (s * s))


def logsumexp(a: np.ndarray) -> float:
    """Stable log-sum-exp over a 1D array."""
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))


def normalize_log_weights(logw: np.ndarray) -> np.ndarray:
    """Return normalized weights from log-weights (stable softmax)."""
    lw = np.asarray(logw, dtype=float)
    lse = logsumexp(lw)
    w = np.exp(lw - lse)
    return w / np.sum(w)


def effective_sample_size(w: np.ndarray) -> float:
    """Effective sample size ESS = 1 / sum(w^2)."""
    w = np.asarray(w, dtype=float)
    s = np.sum(w * w)
    return float(1.0 / s) if s > 0 else 0.0


def systematic_resample(rng: Generator, weights: np.ndarray, n: int | None = None) -> np.ndarray:
    """Systematic resampling; returns integer indices of selected particles.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator for the initial offset u ~ U[0, 1/n).
    weights : array-like
        Normalized weights (sum to 1).
    n : int, optional
        Number of indices to draw; default len(weights).
    """
    w = np.asarray(weights, dtype=float)
    if n is None:
        n = w.size
    # cumulative sum
    c = np.cumsum(w)
    # positions
    u0 = rng.random() / n
    u = u0 + (np.arange(n) / n)
    # walk c to pick indices
    i = 0
    idx = np.empty(n, dtype=int)
    for j, uj in enumerate(u):
        while uj > c[i]:
            i += 1
        idx[j] = i
    return idx


def compute_obs_sigma(
    y: float,
    n_valid: int | None,
    cloud_fraction: float,
    *,
    use_binomial: bool,
    sigma_floor: float,
    sigma_cloud_scale: float,
    min_sigma: float,
    obs_sigma: float | None = None,
) -> float:
    """Compute observation sigma for SCF in the linear domain.

    Combines (optional) binomial variance with a floor and cloud inflation.
    If ``use_binomial`` is False and ``obs_sigma`` is provided, returns at least
    that fixed value.
    """
    var_binom = 0.0
    if use_binomial and n_valid is not None and n_valid > 0:
        var_binom = max(0.0, float(y) * (1.0 - float(y)) / float(n_valid))
    var_floor = float(sigma_floor) ** 2
    var_cloud = float(sigma_cloud_scale) ** 2 * float(cloud_fraction) ** 2
    base = max(float(min_sigma) ** 2, var_binom + var_floor + var_cloud)
    s = float(np.sqrt(base))
    if not use_binomial and obs_sigma is not None:
        s = max(s, float(obs_sigma))
    return s


# ---- Time-series ensemble helpers ------------------------------------------

def envelope(
    series_list: list[pd.Series],
    q_low: float = 0.05,
    q_high: float = 0.95,
    *,
    min_count: int = 1,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (mean, q_low, q_high) across a list of series (union alignment).

    - Aligns on the union of timestamps (outer join) and computes row-wise
      statistics while ignoring NaNs (skipna).
    - Drops timestamps with fewer than ``min_count`` available series.
    """
    if not series_list:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    aligned = pd.concat(series_list, axis=1, join="outer")
    if aligned.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    valid = aligned.count(axis=1) >= max(1, int(min_count))
    if not valid.any():
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    sub = aligned.loc[valid]
    mean = sub.mean(axis=1, skipna=True)
    lo = sub.quantile(q_low, axis=1, numeric_only=True)
    hi = sub.quantile(q_high, axis=1, numeric_only=True)
    return mean, lo, hi

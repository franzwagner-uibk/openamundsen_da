from __future__ import annotations
"""
openamundsen_da.util.stats

Compact statistical helpers used across DA modules.

Includes:
- Prior forcing perturbation samplers
  - Temperature offset: Normal(0, sigma_t^2)
  - Humidity-state offset: Normal(0, sigma_rh^2)
  - Precipitation factor: LogNormal(mu_p, sigma_p^2)
  - Shortwave factor: LogNormal(0, sigma_sw^2)
- Core math utilities
  - Logistic sigmoid with numerical stability
"""

from numpy.random import Generator
import numpy as np
import pandas as pd


def sample_delta_t(rng: Generator, sigma_t: float) -> float:
    """Sample an additive temperature offset ΔT ~ N(0, sigma_t^2)."""
    return float(rng.normal(0.0, sigma_t))


def sample_delta_rh(rng: Generator, sigma_rh: float) -> float:
    """Sample an additive humidity-state offset from N(0, sigma_rh^2)."""
    return float(rng.normal(0.0, sigma_rh))


def sample_precip_factor(rng: Generator, mu_p: float, sigma_p: float) -> float:
    """Sample a multiplicative precipitation factor f_p ~ LogNormal(mu_p, sigma_p^2)."""
    return float(rng.lognormal(mean=mu_p, sigma=sigma_p))


def sample_shortwave_factor(rng: Generator, sigma_sw: float) -> float:
    """Sample a positive multiplicative shortwave factor f_sw ~ LogNormal(0, sigma_sw^2)."""
    return float(rng.lognormal(mean=0.0, sigma=sigma_sw))


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


def normalize_weights(weights: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Return finite, non-negative weights normalized to sum to one."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("Weights must be one-dimensional")
    if w.size == 0:
        raise ValueError("Weights must not be empty")
    if not np.all(np.isfinite(w)):
        raise ValueError("Weights must be finite")
    if np.any(w < 0.0):
        raise ValueError("Weights must be non-negative")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("Weights must sum to a positive value")
    return w / total


def weighted_mean(values: np.ndarray | list[float], weights: np.ndarray | list[float] | None = None) -> float:
    """Return mean of values using optional normalized weights."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(x)):
        raise ValueError("Values must be finite")
    if weights is None:
        return float(np.mean(x))
    w = normalize_weights(weights)
    if w.size != x.size:
        raise ValueError("Values and weights must have the same length")
    return float(np.sum(w * x))


def weighted_variance(values: np.ndarray | list[float], weights: np.ndarray | list[float] | None = None) -> float:
    """Return population variance of values using optional normalized weights."""
    x = np.asarray(values, dtype=float)
    mean = weighted_mean(x, weights=weights)
    if weights is None:
        return float(np.mean((x - mean) ** 2))
    w = normalize_weights(weights)
    if w.size != x.size:
        raise ValueError("Values and weights must have the same length")
    return float(np.sum(w * (x - mean) ** 2))


def weighted_std(values: np.ndarray | list[float], weights: np.ndarray | list[float] | None = None) -> float:
    """Return population standard deviation using optional normalized weights."""
    return float(np.sqrt(max(0.0, weighted_variance(values, weights=weights))))


def weighted_quantile(
    values: np.ndarray | list[float],
    q: float,
    weights: np.ndarray | list[float] | None = None,
) -> float:
    """Return empirical quantile using optional normalized weights."""
    if not np.isfinite(q) or q < 0.0 or q > 1.0:
        raise ValueError(f"Quantile must lie in [0, 1], got {q!r}")
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(x)):
        raise ValueError("Values must be finite")
    order = np.argsort(x)
    x_sorted = x[order]
    if weights is None:
        if x_sorted.size == 1:
            return float(x_sorted[0])
        return float(np.quantile(x_sorted, q))
    w = normalize_weights(weights)
    if w.size != x.size:
        raise ValueError("Values and weights must have the same length")
    w_sorted = w[order]
    cdf = np.cumsum(w_sorted)
    idx = int(np.searchsorted(cdf, q, side="left"))
    idx = min(max(idx, 0), x_sorted.size - 1)
    return float(x_sorted[idx])


def ensemble_crps(
    values: np.ndarray | list[float],
    observation: float,
    *,
    weights: np.ndarray | list[float] | None = None,
) -> float:
    """Return CRPS for an empirical ensemble, optionally with weights."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(x)):
        raise ValueError("Values must be finite")
    y = float(observation)
    if not np.isfinite(y):
        raise ValueError("Observation must be finite")
    if weights is None:
        w = np.full(x.size, 1.0 / x.size, dtype=float)
    else:
        w = normalize_weights(weights)
        if w.size != x.size:
            raise ValueError("Values and weights must have the same length")
    term_obs = float(np.sum(w * np.abs(x - y)))
    pairwise = np.abs(x[:, None] - x[None, :])
    term_ens = 0.5 * float(np.sum((w[:, None] * w[None, :]) * pairwise))
    return term_obs - term_ens


def midpoint_pit(
    values: np.ndarray | list[float],
    observation: float,
    *,
    weights: np.ndarray | list[float] | None = None,
) -> float:
    """Return midpoint PIT value for an empirical ensemble."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("Values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(x)):
        raise ValueError("Values must be finite")
    y = float(observation)
    if not np.isfinite(y):
        raise ValueError("Observation must be finite")
    if weights is None:
        w = np.full(x.size, 1.0 / x.size, dtype=float)
    else:
        w = normalize_weights(weights)
        if w.size != x.size:
            raise ValueError("Values and weights must have the same length")
    less = float(np.sum(w[x < y]))
    equal = float(np.sum(w[x == y]))
    pit = less + 0.5 * equal
    return float(min(max(pit, 0.0), 1.0))


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

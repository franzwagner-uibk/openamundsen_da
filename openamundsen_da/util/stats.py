from __future__ import annotations
"""
openamundsen_da.util.stats

Compact statistical helpers used across DA modules.

Currently provides sampling for prior forcing perturbations:
- Temperature offset: Normal(0, sigma_t^2)
- Precipitation factor: LogNormal(mu_p, sigma_p^2)
"""

from numpy.random import Generator
import numpy as np


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

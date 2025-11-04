from __future__ import annotations
"""
openamundsen_da.core.stats

Compact statistical helpers used across DA modules.

Currently provides sampling for prior forcing perturbations:
- Temperature offset: Normal(0, sigma_t^2)
- Precipitation factor: LogNormal(mu_p, sigma_p^2)
"""

from numpy.random import Generator


def sample_delta_t(rng: Generator, sigma_t: float) -> float:
    """Sample an additive temperature offset ΔT ~ N(0, sigma_t^2)."""
    return float(rng.normal(0.0, sigma_t))


def sample_precip_factor(rng: Generator, mu_p: float, sigma_p: float) -> float:
    """Sample a multiplicative precipitation factor f_p ~ LogNormal(mu_p, sigma_p^2)."""
    return float(rng.lognormal(mean=mu_p, sigma=sigma_p))


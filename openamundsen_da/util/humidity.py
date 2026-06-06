"""Humidity-state perturbation helpers for openAMUNDSEN station forcing."""

from __future__ import annotations

import numpy as np

_KELVIN_OFFSET = 273.15
_MAGNUS_A = 17.62
_MAGNUS_B = 243.12


def relative_humidity_to_dew_point(temp_k, rel_hum):
    """Return dew-point temperature in deg C from air temperature in K and RH in percent."""
    temp_k_arr, rh_arr = _broadcast_float_arrays(temp_k, rel_hum)
    _validate_relative_humidity(rh_arr)
    temp_c = temp_k_arr - _KELVIN_OFFSET
    gamma = np.log(rh_arr / 100.0) + (_MAGNUS_A * temp_c) / (_MAGNUS_B + temp_c)
    dew_point_c = (_MAGNUS_B * gamma) / (_MAGNUS_A - gamma)
    return np.where(_finite_pair(temp_k_arr, rh_arr), dew_point_c, np.nan)


def dew_point_to_relative_humidity(temp_k, dew_point_c):
    """Return relative humidity in percent from air temperature in K and dew point in deg C."""
    temp_k_arr, dew_point_arr = _broadcast_float_arrays(temp_k, dew_point_c)
    temp_c = temp_k_arr - _KELVIN_OFFSET
    exponent = (_MAGNUS_A * dew_point_arr) / (_MAGNUS_B + dew_point_arr)
    exponent -= (_MAGNUS_A * temp_c) / (_MAGNUS_B + temp_c)
    rh = 100.0 * np.exp(exponent)
    return np.where(_finite_pair(temp_k_arr, dew_point_arr), rh, np.nan)


def perturb_relative_humidity_via_dew_point(
    temp_k,
    rel_hum,
    delta_tdew: float,
    *,
    delta_t: float = 0.0,
):
    """Perturb RH by applying an additive offset in dew-point-temperature space."""
    temp_k_arr, rh_arr = _broadcast_float_arrays(temp_k, rel_hum)
    dew_point_c = relative_humidity_to_dew_point(temp_k_arr, rh_arr)
    perturbed_temp_k = temp_k_arr + float(delta_t)
    perturbed_temp_c = perturbed_temp_k - _KELVIN_OFFSET
    perturbed_dew_point_c = np.minimum(dew_point_c + float(delta_tdew), perturbed_temp_c)
    out = dew_point_to_relative_humidity(perturbed_temp_k, perturbed_dew_point_c)
    return np.where(np.isfinite(out), np.clip(out, 0.0, 100.0), np.nan)


def _broadcast_float_arrays(left, right) -> tuple[np.ndarray, np.ndarray]:
    left_arr = np.asarray(left, dtype="float64")
    right_arr = np.asarray(right, dtype="float64")
    return np.broadcast_arrays(left_arr, right_arr)


def _validate_relative_humidity(rel_hum: np.ndarray) -> None:
    finite = np.isfinite(rel_hum)
    invalid = finite & ((rel_hum <= 0.0) | (rel_hum > 100.0))
    if np.any(invalid):
        sample = rel_hum[invalid][:5]
        raise ValueError(
            "Relative humidity values must be within (0, 100] for dew-point "
            f"perturbation; invalid sample: {sample.tolist()}"
        )


def _finite_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.isfinite(left) & np.isfinite(right)

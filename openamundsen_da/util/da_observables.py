"""Shared metadata helpers for supported assimilation observables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class AssimilationObservableSpec:
    """Output and labeling metadata for one assimilation observable."""

    variable: str
    weight_prefix: str
    weight_title: str
    station_diagnostics_prefix: str | None = None


_SPECS = {
    "scf": AssimilationObservableSpec(
        variable="scf",
        weight_prefix="weights_scf",
        weight_title="snow cover data assimilation weights",
    ),
    "wet_snow": AssimilationObservableSpec(
        variable="wet_snow",
        weight_prefix="weights_wet_snow",
        weight_title="wet snow data assimilation weights",
    ),
    "wet_snow_line": AssimilationObservableSpec(
        variable="wet_snow_line",
        weight_prefix="weights_wet_snow_line",
        weight_title="wet snow line data assimilation weights",
    ),
    "station_hs": AssimilationObservableSpec(
        variable="station_hs",
        weight_prefix="weights_station_hs",
        weight_title="station snow depth data assimilation weights",
        station_diagnostics_prefix="station_diagnostics_station_hs",
    ),
    "station_swe": AssimilationObservableSpec(
        variable="station_swe",
        weight_prefix="weights_station_swe",
        weight_title="station swe data assimilation weights",
        station_diagnostics_prefix="station_diagnostics_station_swe",
    ),
}


def assimilation_observable_spec(variable: str) -> AssimilationObservableSpec:
    """Return metadata for one supported assimilation observable."""
    key = str(variable).strip().lower()
    if key == "wet_snow_fraction":
        key = "wet_snow"
    if key not in _SPECS:
        raise ValueError(f"Unsupported assimilation observable: {variable!r}")
    return _SPECS[key]


def weights_csv_name(variable: str, dt: datetime) -> str:
    """Return the standard weights CSV filename for one observable/date."""
    spec = assimilation_observable_spec(variable)
    return f"{spec.weight_prefix}_{dt.strftime('%Y%m%d')}.csv"


def weights_glob_pattern(variable: str) -> str:
    """Return the glob pattern for weights CSVs of one observable."""
    spec = assimilation_observable_spec(variable)
    return f"{spec.weight_prefix}_*.csv"


def station_diagnostics_csv_name(variable: str, dt: datetime) -> str:
    """Return the standard station diagnostics CSV filename for one observable/date."""
    spec = assimilation_observable_spec(variable)
    if spec.station_diagnostics_prefix is None:
        raise ValueError(f"Observable does not produce station diagnostics: {variable!r}")
    return f"{spec.station_diagnostics_prefix}_{dt.strftime('%Y%m%d')}.csv"


def station_diagnostics_glob_pattern(variable: str) -> str:
    """Return the glob pattern for station diagnostics CSVs of one observable."""
    spec = assimilation_observable_spec(variable)
    if spec.station_diagnostics_prefix is None:
        raise ValueError(f"Observable does not produce station diagnostics: {variable!r}")
    return f"{spec.station_diagnostics_prefix}_*.csv"


def weight_plot_title_from_csv_path(csv_path: Path) -> str:
    """Infer a plot title from a weights CSV filename."""
    stem = Path(csv_path).stem.lower()
    specs = sorted(_SPECS.values(), key=lambda spec: len(spec.weight_prefix), reverse=True)
    for spec in specs:
        if stem.startswith(f"{spec.weight_prefix}_"):
            return spec.weight_title
    return "Data Assimilation Weights"

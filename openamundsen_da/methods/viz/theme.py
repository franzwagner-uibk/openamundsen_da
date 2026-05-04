"""Shared visualization tokens used across plot and map renderers."""

from __future__ import annotations


TEXT_COLOR = "#000000"
EXPORT_DPI = 600

# Shared publication-reference sizing used by both result-overview plots and
# project-map figure sizing.
FIGWIDTH_OVERVIEW_PAPER = 7.2876875
FIGHEIGHT_OVERVIEW_ROW = 1.71236835

# Generic DA-variable palette shared beyond the plots subpackage. Keeping this
# in the root viz theme avoids coupling benchmarking and PF diagnostics to
# ``viz.plots.theme`` for colors alone.
_DA_VARIABLE_ALIASES = {
    "scf": "scf",
    "fsc": "scf",
    "wet_snow": "wet_snow",
    "wet_snow_fraction": "wet_snow",
    "wsf": "wet_snow",
    "wet_snow_line": "wet_snow_line",
    "wsla": "wet_snow_line",
    "station_hs": "station_hs",
    "station_sd": "station_hs",
    "sd": "station_hs",
    "snow_depth": "station_hs",
    "snowdepth": "station_hs",
    "hs": "station_hs",
    "station_swe": "station_swe",
    "swe": "station_swe",
}

DA_VARIABLE_STYLES = {
    "scf": {"fill": "#9ad1f0", "line": "#0072B2"},
    "wet_snow": {"fill": "#9de0ca", "line": "#009E73"},
    "wet_snow_line": {"fill": "#efbad4", "line": "#CC79A7"},
    "station_hs": {"fill": "#f2b38d", "line": "#D55E00"},
    "station_swe": {"fill": "#a9d8f0", "line": "#56B4E9"},
}


def canonical_da_variable(variable: str) -> str:
    token = str(variable or "").strip().lower().replace("-", "_")
    return _DA_VARIABLE_ALIASES.get(token, token)


def da_variable_style(variable: str) -> dict[str, str]:
    return DA_VARIABLE_STYLES[canonical_da_variable(variable)]


def da_variable_line_color(variable: str) -> str:
    return da_variable_style(variable)["line"]


def da_variable_fill_color(variable: str) -> str:
    return da_variable_style(variable)["fill"]


__all__ = [
    "DA_VARIABLE_STYLES",
    "EXPORT_DPI",
    "FIGHEIGHT_OVERVIEW_ROW",
    "FIGWIDTH_OVERVIEW_PAPER",
    "TEXT_COLOR",
    "canonical_da_variable",
    "da_variable_fill_color",
    "da_variable_line_color",
    "da_variable_style",
]

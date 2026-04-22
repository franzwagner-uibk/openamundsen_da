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
    "wet_snow_line": "wet_snow",
    "fws": "wet_snow",
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
    "scf": {"fill": "#9ec5ff", "line": "#2f6fb5"},
    "wet_snow": {"fill": "#9bd8bf", "line": "#2c8a64"},
    "station_hs": {"fill": "#f3c38e", "line": "#ff7f0e"},
    "station_swe": {"fill": "#ccb8f2", "line": "#9467bd"},
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

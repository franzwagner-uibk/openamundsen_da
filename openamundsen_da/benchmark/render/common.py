"""Shared helpers for benchmark rendering."""

from __future__ import annotations

from pathlib import Path

from openamundsen_da.methods.viz.theme import da_variable_style as _da_variable_style


VARIABLE_STYLES = {
    "scf": {
        **_da_variable_style("scf"),
        "title": "Snow cover fraction benchmark",
        "ylabel": "snow cover fraction",
        "label": "SCF",
    },
    "wet_snow": {
        **_da_variable_style("wet_snow"),
        "title": "Wet snow fraction (WSF) benchmark",
        "ylabel": "wet snow fraction (WSF)",
        "label": "WSF",
    },
    "wet_snow_line": {
        **_da_variable_style("wet_snow_line"),
        "title": "Wet snow line (WSLA) benchmark",
        "ylabel": "wet snow line (WSLA) [m a.s.l.]",
        "label": "WSLA",
    },
    "station_swe": {
        **_da_variable_style("station_swe"),
        "title": "Station SWE benchmark",
        "ylabel": "swe [mm]",
        "label": "station SWE",
    },
    "station_hs": {
        **_da_variable_style("station_hs"),
        "title": "Station snow-depth benchmark",
        "ylabel": "snow depth [m]",
        "label": "station HS",
    },
}


def variable_style(variable: str) -> dict[str, str]:
    return VARIABLE_STYLES[str(variable)]


def variable_label(variable: str) -> str:
    return variable_style(variable).get("label", str(variable).replace("_", " "))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

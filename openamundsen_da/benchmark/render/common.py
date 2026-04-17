"""Shared helpers for benchmark rendering."""

from __future__ import annotations

from pathlib import Path

from openamundsen_da.methods.viz.plots.theme import da_variable_style as _da_variable_style


VARIABLE_STYLES = {
    "scf": {
        **_da_variable_style("scf"),
        "title": "snow cover fraction benchmark",
        "ylabel": "snow cover fraction",
        "label": "SCF",
    },
    "wet_snow": {
        **_da_variable_style("wet_snow"),
        "title": "wet snow fraction benchmark",
        "ylabel": "wet snow fraction",
        "label": "wet snow",
    },
    "station_swe": {
        **_da_variable_style("station_swe"),
        "title": "station SWE benchmark",
        "ylabel": "swe [mm]",
        "label": "station SWE",
    },
    "station_hs": {
        **_da_variable_style("station_hs"),
        "title": "station snow-depth benchmark",
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

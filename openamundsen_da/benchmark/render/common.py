"""Shared helpers for benchmark rendering."""

from __future__ import annotations

from pathlib import Path


VARIABLE_STYLES = {
    "scf": {
        "fill": "#9ec5ff",
        "line": "#2f6fb5",
        "title": "snow cover fraction benchmark",
        "ylabel": "snow cover fraction",
        "label": "SCF",
    },
    "wet_snow": {
        "fill": "#9bd8bf",
        "line": "#2c8a64",
        "title": "wet snow fraction benchmark",
        "ylabel": "wet snow fraction",
        "label": "wet snow",
    },
    "station_swe": {
        "fill": "#ccb8f2",
        "line": "#7a58b5",
        "title": "station SWE benchmark",
        "ylabel": "swe [mm]",
        "label": "station SWE",
    },
    "station_hs": {
        "fill": "#f3c38e",
        "line": "#cf7a20",
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

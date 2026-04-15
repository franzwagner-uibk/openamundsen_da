from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import FuncNorm, LinearSegmentedColormap, ListedColormap, Normalize, to_rgba


@dataclass(frozen=True)
class VariablePreset:
    variable: str
    title: str
    unit_label: str
    sequential_cmap: str
    model_min: float
    max_step: float
    max_floor: float
    increment_step: float
    increment_floor: float


@dataclass(frozen=True)
class ColorbarStyle:
    label: str
    ticks: tuple[float, ...] = ()
    ticklabels: tuple[str, ...] = ()


VARIABLE_PRESETS = {
    "snowdepth_daily": VariablePreset(
        variable="snowdepth_daily",
        title="snow depth",
        unit_label="snow depth [m]",
        sequential_cmap="YlGnBu",
        model_min=0.0,
        max_step=0.25,
        max_floor=0.5,
        increment_step=0.10,
        increment_floor=0.25,
    ),
    "swe_daily": VariablePreset(
        variable="swe_daily",
        title="snow water equivalent",
        unit_label="SWE [mm]",
        sequential_cmap="viridis",
        model_min=0.0,
        max_step=25.0,
        max_floor=50.0,
        increment_step=10.0,
        increment_floor=25.0,
    ),
    "liquid_water_content": VariablePreset(
        variable="liquid_water_content",
        title="liquid water content",
        unit_label="liquid water content [-]",
        sequential_cmap="magma",
        model_min=0.0,
        max_step=0.005,
        max_floor=0.01,
        increment_step=0.002,
        increment_floor=0.005,
    ),
}


SNOW_DEPTH_REFERENCE_TICKS_M = (0.01, 0.10, 0.25, 0.50, 1.0, 2.0, 3.0, 4.0)
SNOW_DEPTH_REFERENCE_TICKLABELS_CM = ("1", "10", "25", "50", "100", "200", "300", "400+")
SNOW_DEPTH_REFERENCE_COLORS = (
    "#ffffb2",
    "#b0ffbc",
    "#8cffff",
    "#19cdff",
    "#1982ff",
    "#0f5abe",
    "#784bff",
    "#cd0feb",
)


LANDCOVER_COLORS = {
    1: "#b7b1a8",
    2: "#d9f2ff",
    3: "#5ba3d0",
    4: "#b7d36b",
    5: "#8db255",
    6: "#d1b45a",
    7: "#c8a16c",
    8: "#92b97b",
    9: "#608f56",
    10: "#3a6b44",
    11: "#2e5b39",
    12: "#1f4a31",
    13: "#7a6d6a",
}

LANDCOVER_LABELS = {
    1: "rock",
    2: "ice",
    3: "water",
    4: "grassland",
    5: "shrubland",
    6: "farmland",
    7: "transitional",
    8: "deciduous 30-60",
    9: "deciduous 60-100",
    10: "mixed forest",
    11: "coniferous 30-60",
    12: "coniferous 60-100",
    13: "built-up",
}

FSC_OBS_CMAP = colormaps["Blues"]
UNKNOWN_LANDCOVER_COLOR = "#d9d9d9"
WET_SNOW_COLORS = {
    110: "#cf3c2e",
    125: "#f4cf65",
    200: "#8c8c8c",
    210: "#4b79c6",
}
WET_SNOW_LABELS = {
    110: "wet",
    125: "dry / no snow",
    200: "radar shadow",
    210: "water",
}


def _snow_depth_reference_cmap() -> LinearSegmentedColormap:
    positions = tuple(idx / (len(SNOW_DEPTH_REFERENCE_COLORS) - 1) for idx in range(len(SNOW_DEPTH_REFERENCE_COLORS)))
    cmap = LinearSegmentedColormap.from_list(
        "oa_da_snow_depth_reference",
        list(zip(positions, SNOW_DEPTH_REFERENCE_COLORS)),
        N=512,
    )
    transparent = (1.0, 1.0, 1.0, 0.0)
    cmap.set_under(transparent)
    cmap.set_bad(transparent)
    cmap.set_over(to_rgba(SNOW_DEPTH_REFERENCE_COLORS[-1]))
    return cmap


SNOW_DEPTH_CMAP = _snow_depth_reference_cmap()
INCREMENT_CMAP = colormaps["RdBu"]


def _snow_depth_reference_norm() -> Normalize:
    positions = np.linspace(0.0, 1.0, len(SNOW_DEPTH_REFERENCE_TICKS_M))

    def forward(values):
        return np.interp(values, SNOW_DEPTH_REFERENCE_TICKS_M, positions)

    def inverse(values):
        return np.interp(values, positions, SNOW_DEPTH_REFERENCE_TICKS_M)

    return FuncNorm(
        (forward, inverse),
        vmin=SNOW_DEPTH_REFERENCE_TICKS_M[0],
        vmax=SNOW_DEPTH_REFERENCE_TICKS_M[-1],
        clip=False,
    )


def require_variable_preset(variable: str) -> VariablePreset:
    token = str(variable).strip()
    try:
        return VARIABLE_PRESETS[token]
    except KeyError as exc:
        supported = ", ".join(sorted(VARIABLE_PRESETS))
        raise ValueError(f"Unsupported project-map variable '{variable}'. Supported variables: {supported}") from exc


def nice_ceiling(value: float, *, step: float, minimum: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    return max(minimum, math.ceil(float(value) / step) * step)


def landcover_cmap_for_codes(codes: list[int]) -> ListedColormap:
    return ListedColormap(
        [LANDCOVER_COLORS.get(int(code), UNKNOWN_LANDCOVER_COLOR) for code in codes],
        name="oa_da_landcover",
    )


def model_map_cmap(preset: VariablePreset):
    if preset.variable == "snowdepth_daily":
        return SNOW_DEPTH_CMAP
    return colormaps[preset.sequential_cmap]


def model_map_norm(preset: VariablePreset, *, vmax: float) -> Normalize:
    if preset.variable == "snowdepth_daily":
        return _snow_depth_reference_norm()
    return Normalize(vmin=preset.model_min, vmax=vmax, clip=False)


def model_colorbar_style(preset: VariablePreset) -> ColorbarStyle:
    if preset.variable == "snowdepth_daily":
        return ColorbarStyle(
            label="snow depth [cm]",
            ticks=SNOW_DEPTH_REFERENCE_TICKS_M,
            ticklabels=SNOW_DEPTH_REFERENCE_TICKLABELS_CM,
        )
    return ColorbarStyle(label=preset.unit_label)

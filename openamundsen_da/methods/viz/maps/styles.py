from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import FuncNorm, LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm, to_rgba


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


@dataclass(frozen=True)
class StaticFieldPreset:
    field: str
    title: str
    unit_label: str
    cmap_name: str
    vmin: float | None = None
    vmax: float | None = None
    step: float | None = None
    floor: float | None = None
    center: float | None = None
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
        sequential_cmap="viridis_r",
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
        sequential_cmap="viridis_r",
        model_min=0.0,
        max_step=0.005,
        max_floor=0.01,
        increment_step=0.002,
        increment_floor=0.005,
    ),
}


STATIC_FIELD_PRESETS = {
    "dem": StaticFieldPreset(
        field="dem",
        title="digital elevation model",
        unit_label="elevation [m]",
        cmap_name="Greys_r",
        step=250.0,
        floor=500.0,
    ),
    "svf": StaticFieldPreset(
        field="svf",
        title="sky view factor",
        unit_label="SVF [-]",
        cmap_name="Greys_r",
        vmin=0.5,
        vmax=1.0,
        ticks=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    ),
    "srf": StaticFieldPreset(
        field="srf",
        title="snow redistribution factor",
        unit_label="SRF [-]",
        cmap_name="RdBu",
        vmin=0.1,
        vmax=1.9,
        center=1.0,
        ticks=(0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8),
    ),
    "landcover": StaticFieldPreset(
        field="landcover",
        title="landcover",
        unit_label="landcover",
        cmap_name="oa_da_landcover",
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

FSC_OBS_CMAP = colormaps["Greys"]
FSC_INVALID_COLOR = "#d8b3b7"
UNKNOWN_LANDCOVER_COLOR = "#d9d9d9"
WET_SNOW_COLORS = {
    110: "#000000",
    125: "#d8d8d8",
    200: "#ddb9ba",
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
SNOW_DEPTH_COLORBAR_STEPS_CM = (5.0, 10.0, 25.0, 50.0, 100.0)


def snow_depth_scale_ticks(vmax: float) -> tuple[float, ...]:
    low = float(SNOW_DEPTH_REFERENCE_TICKS_M[0])
    ref_high = float(SNOW_DEPTH_REFERENCE_TICKS_M[-1])
    high = max(float(vmax), low)
    if high == low:
        return (low,)
    positions = np.linspace(0.0, 1.0, len(SNOW_DEPTH_REFERENCE_TICKS_M))
    ticks = np.interp(positions, (0.0, 1.0), (low, high))
    reference_positions = np.interp(
        np.asarray(SNOW_DEPTH_REFERENCE_TICKS_M, dtype=float),
        (low, ref_high),
        (0.0, 1.0),
    )
    scaled = np.interp(reference_positions, positions, ticks)
    scaled[0] = low
    scaled[-1] = high
    return tuple(float(value) for value in scaled)


def snow_depth_colorbar_labels_cm(vmax: float) -> tuple[float, ...]:
    vmax_cm = max(float(vmax) * 100.0, 1.0)
    for step in SNOW_DEPTH_COLORBAR_STEPS_CM:
        labels = [1.0]
        current = step
        while current < vmax_cm - 1e-9:
            labels.append(float(current))
            current += step
        if abs(labels[-1] - vmax_cm) > 1e-9:
            labels.append(float(vmax_cm))
        if len(labels) <= 7:
            return tuple(labels)
    return (1.0, float(vmax_cm))


def snow_depth_colorbar_ticks(vmax: float) -> tuple[float, ...]:
    labels_cm = snow_depth_colorbar_labels_cm(vmax)
    if len(labels_cm) == 1:
        return (float(vmax),)
    target_positions = np.linspace(0.0, 1.0, len(labels_cm))
    anchor_positions = np.linspace(0.0, 1.0, len(SNOW_DEPTH_REFERENCE_TICKS_M))
    return tuple(
        float(value)
        for value in np.interp(
            target_positions,
            anchor_positions,
            np.asarray(snow_depth_scale_ticks(vmax), dtype=float),
        )
    )


def snow_depth_colorbar_ticklabels(vmax: float) -> tuple[str, ...]:
    return tuple(str(int(round(value))) for value in snow_depth_colorbar_labels_cm(vmax))


def _snow_depth_reference_norm(vmax: float) -> Normalize:
    ticks = snow_depth_scale_ticks(vmax)
    positions = np.linspace(0.0, 1.0, len(ticks))

    def forward(values):
        return np.interp(values, ticks, positions)

    def inverse(values):
        return np.interp(values, positions, ticks)

    return FuncNorm(
        (forward, inverse),
        vmin=ticks[0],
        vmax=ticks[-1],
        clip=False,
    )


def require_variable_preset(variable: str) -> VariablePreset:
    token = str(variable).strip()
    try:
        return VARIABLE_PRESETS[token]
    except KeyError as exc:
        supported = ", ".join(sorted(VARIABLE_PRESETS))
        raise ValueError(f"Unsupported project-map variable '{variable}'. Supported variables: {supported}") from exc


def require_static_field_preset(field: str) -> StaticFieldPreset:
    token = str(field).strip()
    try:
        return STATIC_FIELD_PRESETS[token]
    except KeyError as exc:
        supported = ", ".join(sorted(STATIC_FIELD_PRESETS))
        raise ValueError(f"Unsupported project-map static field '{field}'. Supported fields: {supported}") from exc


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
        return _snow_depth_reference_norm(vmax)
    return Normalize(vmin=preset.model_min, vmax=vmax, clip=False)


def model_colorbar_style(preset: VariablePreset, *, vmax: float | None = None) -> ColorbarStyle:
    if preset.variable == "snowdepth_daily":
        if vmax is None:
            return ColorbarStyle(
                label="snow depth [cm]",
                ticks=SNOW_DEPTH_REFERENCE_TICKS_M,
                ticklabels=SNOW_DEPTH_REFERENCE_TICKLABELS_CM,
            )
        ticks = snow_depth_colorbar_ticks(vmax if vmax is not None else SNOW_DEPTH_REFERENCE_TICKS_M[-1])
        return ColorbarStyle(
            label="snow depth [cm]",
            ticks=ticks,
            ticklabels=snow_depth_colorbar_ticklabels(vmax),
        )
    return ColorbarStyle(label=preset.unit_label)


def static_field_cmap(preset: StaticFieldPreset):
    if preset.field == "landcover":
        raise ValueError("Landcover colors are derived from present codes")
    return colormaps[preset.cmap_name]


def static_field_range(preset: StaticFieldPreset, values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        if preset.vmin is not None and preset.vmax is not None:
            return float(preset.vmin), float(preset.vmax)
        return (0.0, 1.0)

    if preset.vmin is not None:
        vmin = float(preset.vmin)
    else:
        step = float(preset.step or 1.0)
        vmin = math.floor(float(finite.min()) / step) * step

    if preset.vmax is not None:
        vmax = float(preset.vmax)
    else:
        vmax = nice_ceiling(float(finite.max()), step=float(preset.step or 1.0), minimum=float(preset.floor or 1.0))
    if vmax <= vmin:
        vmax = vmin + max(float(preset.step or 1.0), 1e-6)
    return vmin, vmax


def static_field_norm(preset: StaticFieldPreset, values: np.ndarray) -> Normalize:
    vmin, vmax = static_field_range(preset, values)
    if preset.center is not None:
        if vmin < float(preset.center) < vmax:
            return TwoSlopeNorm(vcenter=float(preset.center), vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax, clip=False)


def static_field_colorbar_style(preset: StaticFieldPreset) -> ColorbarStyle:
    return ColorbarStyle(label=preset.unit_label, ticks=preset.ticks, ticklabels=preset.ticklabels)

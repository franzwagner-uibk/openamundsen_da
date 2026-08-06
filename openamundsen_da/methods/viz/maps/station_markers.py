"""Station-source and role classification for project maps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

FORCING_STATION_COLOR = "#c21f24"
SNOW_STATION_COLOR = "#2166ac"
HOLDOUT_STATION_COLOR = "#482475"
STATION_MARKER_SIZE = 26.0
STATION_CLUSTER_RADIUS_POINTS = 5.0
LEFT_HALF_TRIANGLE = MplPath(
    [(0.0, 0.5), (-0.5, -0.5), (0.0, -0.5), (0.0, 0.5)],
    [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY],
)
RIGHT_HALF_TRIANGLE = MplPath(
    [(0.0, 0.5), (0.0, -0.5), (0.5, -0.5), (0.0, 0.5)],
    [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY],
)


@dataclass(frozen=True)
class StationMarker:
    """One classified station marker in setup coordinates."""

    kind: str
    x: float
    y: float
    station_id: str
    name: str
    alt: float | None
    forcing_id: str | None = None
    offset_x_points: float = 0.0
    offset_y_points: float = 0.0


def _require_columns(table: pd.DataFrame, columns: set[str], *, label: str) -> None:
    missing = sorted(columns - set(table.columns))
    if missing:
        raise ValueError(
            f"{label} station metadata missing required columns: {', '.join(missing)}"
        )


def _station_ids(table: pd.DataFrame, column: str, *, label: str) -> pd.Series:
    values = table[column].astype("string").fillna("").str.strip()
    if (values == "").any():
        raise ValueError(f"{label} station metadata contains an empty {column}")
    if values.duplicated().any():
        duplicates = sorted(values[values.duplicated(keep=False)].unique())
        raise ValueError(
            f"{label} station metadata contains duplicate IDs: {duplicates}"
        )
    return values


def _finite_coordinates(
    table: pd.DataFrame, *, label: str
) -> tuple[np.ndarray, np.ndarray]:
    x = pd.to_numeric(table["x"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(table["y"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(
            f"{label} station metadata contains non-finite x/y coordinates"
        )
    return x, y


def _role_flag(value: object, *, column: str, station_id: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    token = str(value).strip().lower()
    if token in {"1", "true", "t", "yes", "y"}:
        return True
    if token in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid {column} for snow station {station_id!r}: {value!r}")


def _optional_alt(row: pd.Series) -> float | None:
    if "alt" not in row or pd.isna(row["alt"]):
        return None
    value = float(row["alt"])
    return value if np.isfinite(value) else None


def _marker_name(row: pd.Series, station_id: str) -> str:
    value = row.get("name")
    return (
        str(value).strip()
        if value is not None and not pd.isna(value) and str(value).strip()
        else station_id
    )


def _cluster_offsets(count: int) -> tuple[tuple[float, float], ...]:
    if count <= 1:
        return ((0.0, 0.0),)
    return tuple(
        (
            STATION_CLUSTER_RADIUS_POINTS
            * math.cos(math.radians(90.0 - index * 360.0 / count)),
            STATION_CLUSTER_RADIUS_POINTS
            * math.sin(math.radians(90.0 - index * 360.0 / count)),
        )
        for index in range(count)
    )


def classify_station_markers(
    forcing_stations: pd.DataFrame,
    snow_stations: pd.DataFrame,
    *,
    tolerance_m: float = 10.0,
) -> tuple[StationMarker, ...]:
    """Return deterministic forcing/snow/holdout map markers.

    Every snow station is matched independently to its nearest forcing station.
    Multiple snow records may therefore share one forcing station; those records
    are retained and arranged around the forcing coordinate.
    """

    if not np.isfinite(float(tolerance_m)) or float(tolerance_m) <= 0.0:
        raise ValueError("station_match_tolerance_m must be > 0")
    _require_columns(forcing_stations, {"id", "x", "y"}, label="Forcing")
    _require_columns(
        snow_stations,
        {"station_id", "x", "y", "use_for_da", "use_for_benchmark"},
        label="Snow",
    )
    forcing = forcing_stations.reset_index(drop=True).copy()
    snow = snow_stations.reset_index(drop=True).copy()
    forcing_ids = _station_ids(forcing, "id", label="Forcing")
    snow_ids = _station_ids(snow, "station_id", label="Snow")
    forcing_x, forcing_y = _finite_coordinates(forcing, label="Forcing")
    snow_x, snow_y = _finite_coordinates(snow, label="Snow")

    matches_by_forcing: dict[int, list[int]] = {}
    unmatched_snow: list[int] = []
    for snow_index, station_id in enumerate(snow_ids):
        if forcing.empty:
            unmatched_snow.append(snow_index)
            continue
        distances = np.hypot(
            forcing_x - snow_x[snow_index], forcing_y - snow_y[snow_index]
        )
        nearest_distance = float(np.min(distances))
        candidates = np.flatnonzero(
            np.isclose(distances, nearest_distance, rtol=0.0, atol=1e-9)
        )
        forcing_index = min(candidates, key=lambda idx: str(forcing_ids.iloc[int(idx)]))
        if nearest_distance <= float(tolerance_m):
            matches_by_forcing.setdefault(int(forcing_index), []).append(snow_index)
        else:
            unmatched_snow.append(snow_index)

    markers: list[StationMarker] = []
    for forcing_index, row in forcing.iterrows():
        matched = sorted(
            matches_by_forcing.get(forcing_index, []),
            key=lambda idx: str(snow_ids.iloc[idx]),
        )
        if not matched:
            station_id = str(forcing_ids.iloc[forcing_index])
            markers.append(
                StationMarker(
                    kind="forcing",
                    x=float(forcing_x[forcing_index]),
                    y=float(forcing_y[forcing_index]),
                    station_id=station_id,
                    name=_marker_name(row, station_id),
                    alt=_optional_alt(row),
                )
            )
            continue

        for snow_index, offset in zip(
            matched, _cluster_offsets(len(matched)), strict=True
        ):
            snow_row = snow.iloc[snow_index]
            station_id = str(snow_ids.iloc[snow_index])
            use_for_da = _role_flag(
                snow_row["use_for_da"], column="use_for_da", station_id=station_id
            )
            is_holdout = _role_flag(
                snow_row["use_for_benchmark"],
                column="use_for_benchmark",
                station_id=station_id,
            )
            if use_for_da and is_holdout:
                raise ValueError(
                    f"Snow station {station_id!r} cannot be both DA-active and a holdout"
                )
            markers.append(
                StationMarker(
                    kind="holdout" if is_holdout else "both",
                    x=float(forcing_x[forcing_index]),
                    y=float(forcing_y[forcing_index]),
                    station_id=station_id,
                    name=_marker_name(snow_row, station_id),
                    alt=_optional_alt(snow_row),
                    forcing_id=str(forcing_ids.iloc[forcing_index]),
                    offset_x_points=float(offset[0]),
                    offset_y_points=float(offset[1]),
                )
            )

    for snow_index in sorted(unmatched_snow, key=lambda idx: str(snow_ids.iloc[idx])):
        row = snow.iloc[snow_index]
        station_id = str(snow_ids.iloc[snow_index])
        use_for_da = _role_flag(
            row["use_for_da"], column="use_for_da", station_id=station_id
        )
        is_holdout = _role_flag(
            row["use_for_benchmark"], column="use_for_benchmark", station_id=station_id
        )
        if use_for_da and is_holdout:
            raise ValueError(
                f"Snow station {station_id!r} cannot be both DA-active and a holdout"
            )
        markers.append(
            StationMarker(
                kind="holdout" if is_holdout else "snow",
                x=float(snow_x[snow_index]),
                y=float(snow_y[snow_index]),
                station_id=station_id,
                name=_marker_name(row, station_id),
                alt=_optional_alt(row),
            )
        )

    return tuple(markers)


__all__ = [
    "FORCING_STATION_COLOR",
    "HOLDOUT_STATION_COLOR",
    "LEFT_HALF_TRIANGLE",
    "RIGHT_HALF_TRIANGLE",
    "SNOW_STATION_COLOR",
    "STATION_CLUSTER_RADIUS_POINTS",
    "STATION_MARKER_SIZE",
    "StationMarker",
    "classify_station_markers",
]

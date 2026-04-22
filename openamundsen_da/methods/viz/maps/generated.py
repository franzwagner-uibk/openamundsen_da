from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from openamundsen_da.benchmark.extract.cases import benchmark_variable_spec
from openamundsen_da.io.paths import project_fraction_envelope_path
from openamundsen_da.methods.viz.fraction_series import load_fraction_series, load_open_loop_fraction_series
from openamundsen_da.methods.viz.fraction_series import default_fraction_obs_path
from openamundsen_da.methods.viz.maps.config import LayoutSpec, MapDefaults, MapPanelSpec, MapRecipe
from openamundsen_da.util.da_events import AssimilationEvent, load_assimilation_events


GENERATED_DA_MAPS_SUBDIR = "da_events"
_FRACTION_REFERENCE_VARIABLES = ("scf", "wet_snow")
_VARIABLE_LABELS = {
    "scf": "snow cover fraction",
    "wet_snow": "wet snow",
    "wet_snow_line": "wet snow line",
    "station_hs": "station snow depth",
    "station_swe": "station snow water equivalent",
}
_STREAM_VARIABLE_LABELS = {
    "scf": "snow cover",
}


@dataclass(frozen=True)
class GeneratedRow:
    label: str
    panels: tuple[MapPanelSpec, ...]


def _project_setup_dir(project_dir: Path) -> Path:
    return project_dir.parent.parent


def _variable_label(variable: str) -> str:
    return _VARIABLE_LABELS.get(variable, str(variable).replace("_", " "))


def _stream_row_label(variable: str, relation: str | None = None) -> str:
    base = _STREAM_VARIABLE_LABELS.get(variable, _variable_label(variable))
    if relation is None:
        return base
    return f"{base} ({relation})"


def _fraction_summary_dates(project_dir: Path, variable: str) -> set[pd.Timestamp]:
    spec = benchmark_variable_spec(variable)
    if spec.summary_filename is None:
        return set()
    setup_dir = _project_setup_dir(project_dir)
    summary_path = default_fraction_obs_path(setup_dir, Path(project_dir).name, spec.summary_filename)
    if not summary_path.is_file():
        return set()
    df = pd.read_csv(summary_path, usecols=["date"])
    if df.empty:
        return set()
    return {pd.Timestamp(value).normalize() for value in pd.to_datetime(df["date"]).tolist()}


def _event_dates_by_variable(project_dir: Path) -> dict[str, set[pd.Timestamp]]:
    by_variable: dict[str, set[pd.Timestamp]] = {}
    for event in load_assimilation_events(project_dir):
        by_variable.setdefault(event.variable, set()).add(pd.Timestamp(event.date).normalize())
    return by_variable


def _relation_for_variable(
    project_dir: Path,
    *,
    variable: str,
    date: pd.Timestamp,
    require_summary_date: bool,
) -> str | None:
    if require_summary_date:
        summary_dates = _fraction_summary_dates(project_dir, variable)
        if date not in summary_dates:
            return None
    event_dates = _event_dates_by_variable(project_dir).get(variable, set())
    if date in event_dates:
        return None
    return "semi-independent" if event_dates and min(event_dates) < date else "independent"


def _fraction_model_support_available(project_dir: Path, variable: str) -> bool:
    if not (project_dir / "steps").is_dir():
        return False
    value_col = "scf" if variable == "scf" else "wet_snow_fraction"
    member_filename = "point_scf_roi.csv" if variable == "scf" else "point_wet_snow_roi.csv"
    open_loop = load_open_loop_fraction_series(project_dir, member_filename, value_col)
    envelope = load_fraction_series(project_fraction_envelope_path(project_dir, variable), "value_mean")
    return open_loop is not None and envelope is not None


def _reference_stream(project_dir: Path, *, variable: str, date: pd.Timestamp) -> str | None:
    if variable not in _FRACTION_REFERENCE_VARIABLES:
        return None
    relation = _relation_for_variable(project_dir, variable=variable, date=date, require_summary_date=True)
    if relation is None:
        return None
    return _stream_row_label(variable, relation)


def _snow_depth_row(*, row: int, label: str) -> GeneratedRow:
    return GeneratedRow(
        label=label,
        panels=(
            MapPanelSpec(kind="snow_depth", row=row, col=0, source="open_loop", title="open loop", show_hillshade=True),
            MapPanelSpec(kind="snow_depth", row=row, col=1, source="ensemble_mean", title="ensemble mean", show_hillshade=True),
            MapPanelSpec(kind="snow_depth", row=row, col=2, source="increment", title="increment", show_hillshade=True),
        ),
    )


def _fraction_row(*, row: int, kind: str, label: str) -> GeneratedRow:
    if kind == "fsc":
        return GeneratedRow(
            label=label,
            panels=(
                MapPanelSpec(
                    kind=kind,
                    row=row,
                    col=0,
                    source="open_loop_binary",
                    title="open-loop snow cover",
                    show_hillshade=True,
                    hillshade_extent="roi",
                ),
                MapPanelSpec(
                    kind=kind,
                    row=row,
                    col=1,
                    source="posterior_probability",
                    title="ensemble snow-cover probability",
                    show_hillshade=True,
                    hillshade_extent="roi",
                ),
                MapPanelSpec(kind=kind, row=row, col=2, title="satellite FSC observation"),
            ),
        )
    return GeneratedRow(
        label=label,
        panels=(
            MapPanelSpec(kind=kind, row=row, col=0, source="open_loop", title="open loop", show_hillshade=True, hillshade_extent="roi"),
            MapPanelSpec(kind=kind, row=row, col=1, source="ensemble_mean", title="ensemble mean", show_hillshade=True, hillshade_extent="roi"),
            MapPanelSpec(kind=kind, row=row, col=2, title="observation"),
        ),
    )


def _wet_snow_line_row(*, row: int, label: str) -> GeneratedRow:
    return GeneratedRow(
        label=label,
        panels=(
            MapPanelSpec(kind="wet_snow_line", row=row, col=0, source="open_loop", title="open loop", show_hillshade=True, hillshade_extent="roi"),
            MapPanelSpec(kind="wet_snow_line", row=row, col=1, source="posterior", title="posterior", show_hillshade=True, hillshade_extent="roi"),
            MapPanelSpec(kind="wet_snow_line", row=row, col=2, title="observation"),
        ),
    )


def _generated_rows_for_event(project_dir: Path, event: AssimilationEvent) -> tuple[GeneratedRow, ...]:
    rows: list[GeneratedRow] = []
    row_index = 0
    if event.variable == "station_hs":
        rows.append(_snow_depth_row(row=row_index, label=_variable_label("station_hs")))
        row_index += 1
    elif event.variable == "station_swe":
        rows.append(_snow_depth_row(row=row_index, label=_variable_label("station_swe")))
        row_index += 1
    elif event.variable == "scf" and _fraction_model_support_available(project_dir, "scf"):
        rows.append(_fraction_row(row=row_index, kind="fsc", label=_stream_row_label("scf")))
        row_index += 1
        hs_relation = _relation_for_variable(
            project_dir,
            variable="station_hs",
            date=pd.Timestamp(event.date).normalize(),
            require_summary_date=False,
        )
        hs_label = _stream_row_label("station_hs", hs_relation or "independent")
        rows.append(_snow_depth_row(row=row_index, label=hs_label))
        row_index += 1
    elif event.variable == "wet_snow_line" and _fraction_model_support_available(project_dir, "wet_snow"):
        rows.append(_wet_snow_line_row(row=row_index, label=_stream_row_label("wet_snow_line")))
        row_index += 1
        hs_relation = _relation_for_variable(
            project_dir,
            variable="station_hs",
            date=pd.Timestamp(event.date).normalize(),
            require_summary_date=False,
        )
        hs_label = _stream_row_label("station_hs", hs_relation or "independent")
        rows.append(_snow_depth_row(row=row_index, label=hs_label))
        row_index += 1
    elif event.variable == "wet_snow" and _fraction_model_support_available(project_dir, "wet_snow"):
        rows.append(_fraction_row(row=row_index, kind="wet_snow", label=_stream_row_label("wet_snow")))
        row_index += 1
        hs_relation = _relation_for_variable(
            project_dir,
            variable="station_hs",
            date=pd.Timestamp(event.date).normalize(),
            require_summary_date=False,
        )
        hs_label = _stream_row_label("station_hs", hs_relation or "independent")
        rows.append(_snow_depth_row(row=row_index, label=hs_label))
        row_index += 1
    else:
        rows.append(_snow_depth_row(row=row_index, label=_variable_label("station_hs")))
        row_index += 1

    event_date = pd.Timestamp(event.date).normalize()
    for variable in _FRACTION_REFERENCE_VARIABLES:
        if variable == event.variable or (event.variable == "wet_snow_line" and variable == "wet_snow"):
            continue
        if not _fraction_model_support_available(project_dir, variable):
            continue
        stream_label = _reference_stream(project_dir, variable=variable, date=event_date)
        if stream_label is None:
            continue
        kind = "fsc" if variable == "scf" else "wet_snow"
        rows.append(_fraction_row(row=row_index, kind=kind, label=stream_label))
        row_index += 1
    return tuple(rows)


def _generated_recipe(index: int, event: AssimilationEvent, rows: tuple[GeneratedRow, ...]) -> MapRecipe:
    panels = tuple(
        MapPanelSpec(
            kind=panel.kind,
            row=row_idx,
            col=panel.col,
            title=panel.title,
            source=panel.source,
            show_hillshade=panel.show_hillshade,
            hillshade_extent=panel.hillshade_extent,
        )
        for row_idx, row in enumerate(rows)
        for panel in row.panels
    )
    return MapRecipe(
        name=f"da_{index}",
        title=f"da_{index}",
        figure_title=f"{event.date.isoformat()} ({_variable_label(event.variable)})",
        output_subdir=GENERATED_DA_MAPS_SUBDIR,
        layout=LayoutSpec(nrows=len(rows), ncols=3),
        row_labels=tuple(row.label for row in rows),
        defaults=MapDefaults(date=event.date.isoformat(), show_scalebar=True),
        panels=panels,
    )


def generated_da_map_recipes(project_dir: Path) -> tuple[MapRecipe, ...]:
    events = load_assimilation_events(project_dir)
    return tuple(
        _generated_recipe(index, event, _generated_rows_for_event(project_dir, event))
        for index, event in enumerate(events, start=1)
    )


def generated_da_maps_available(project_dir: Path) -> bool:
    try:
        return bool(load_assimilation_events(project_dir))
    except Exception:
        return False


def project_maps_custom_config_path(project_dir: Path) -> Path:
    return Path(project_dir) / "maps.yml"


def project_maps_custom_config_exists(project_dir: Path) -> bool:
    return project_maps_custom_config_path(project_dir).is_file()


def default_project_maps_rerun_command(project_dir: Path, *, recipe_name: str | None = None, config_path: Path | None = None) -> str:
    parts = [
        "python",
        "-m",
        "openamundsen_da.methods.viz.maps.runner",
        "--project-dir",
        str(Path(project_dir)),
    ]
    if config_path is not None and Path(config_path).is_file():
        parts.extend(["--config", str(Path(config_path))])
    if recipe_name:
        parts.extend(["--name", recipe_name])
    return " ".join(parts)


__all__ = [
    "GENERATED_DA_MAPS_SUBDIR",
    "default_project_maps_rerun_command",
    "generated_da_map_recipes",
    "generated_da_maps_available",
    "project_maps_custom_config_exists",
    "project_maps_custom_config_path",
]

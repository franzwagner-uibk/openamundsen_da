from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from openamundsen_da.benchmark.extract.cases import benchmark_variable_spec
from openamundsen_da.io.paths import project_fraction_envelope_path
from openamundsen_da.methods.viz.fraction_series import load_fraction_series, load_open_loop_fraction_series
from openamundsen_da.methods.viz.maps.config import LayoutSpec, MapDefaults, MapPanelSpec, MapRecipe
from openamundsen_da.observer.summary_paths import resolve_fraction_summary_path
from openamundsen_da.subdomain.manifest import SubdomainManifest
from openamundsen_da.util.da_events import AssimilationEvent, load_assimilation_events
from openamundsen_da.util.run_mode import read_run_mode


GENERATED_DA_MAPS_SUBDIR = "da_events"
_FRACTION_REFERENCE_VARIABLES = ("scf", "wet_snow")
_VARIABLE_LABELS = {
    "scf": "Snow cover fraction",
    "wet_snow": "Wet snow fraction (WSF)",
    "wet_snow_line": "Wet snow line (WSLA)",
    "station_hs": "Station snow depth",
    "station_swe": "Station snow water equivalent",
}
_STREAM_VARIABLE_LABELS = {
    "scf": "Snow cover",
}


@dataclass(frozen=True)
class GeneratedRow:
    label: str
    panels: tuple[MapPanelSpec, ...]


def _project_setup_dir(project_dir: Path) -> Path:
    return project_dir.parent.parent


def _variable_label(variable: str) -> str:
    return _VARIABLE_LABELS.get(variable, str(variable).replace("_", " "))


def _figure_title_variable_label(variable: str) -> str:
    if variable == "wet_snow_line":
        return "Wet snow line - WSLA"
    return _variable_label(variable)


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
    summary_path = resolve_fraction_summary_path(setup_dir, project_dir, spec.summary_filename)
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
    if _is_top_level_subdomain_project(project_dir):
        return _top_level_subdomain_fraction_support_available(project_dir, variable)
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


def _resampling_skipped(project_dir: Path, date: pd.Timestamp) -> bool:
    stamp = pd.Timestamp(date).strftime("%Y%m%d")
    steps_dir = Path(project_dir) / "steps"
    if not steps_dir.is_dir():
        return False
    for manifest_path in steps_dir.glob(f"*/assim/resample_manifest_{stamp}.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if bool(manifest.get("skipped")):
            return True
    return False


def _generated_figure_title(index: int, project_dir: Path, event: AssimilationEvent) -> str:
    title = f"DA {index} - {event.date.isoformat()} ({_figure_title_variable_label(event.variable)})"
    if _resampling_skipped(project_dir, pd.Timestamp(event.date).normalize()):
        title += " - resampling skipped"
    return title


def _snow_depth_row(*, row: int, label: str, event_variable: str) -> GeneratedRow:
    return GeneratedRow(
        label=label,
        panels=(
            MapPanelSpec(kind="snow_depth", row=row, col=0, source="open_loop", title="Open-loop snow depth", show_hillshade=True),
            MapPanelSpec(kind="snow_depth", row=row, col=1, source="ensemble_mean", title="Prior snow depth", show_hillshade=True),
            MapPanelSpec(
                kind="snow_depth",
                row=row,
                col=2,
                source="analysis_mean",
                title="Posterior snow depth",
                show_hillshade=True,
                variable=event_variable,
            ),
            MapPanelSpec(
                kind="snow_depth",
                row=row,
                col=3,
                source="analysis_increment",
                title="Snow-depth increment",
                show_hillshade=True,
                variable=event_variable,
            ),
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
                    title="Open-loop snow cover",
                    show_hillshade=True,
                    hillshade_extent="roi",
                ),
                MapPanelSpec(
                    kind=kind,
                    row=row,
                    col=1,
                    source="prior_probability",
                    title="Prior snow-cover probability",
                    show_hillshade=True,
                    hillshade_extent="roi",
                ),
                MapPanelSpec(
                    kind=kind,
                    row=row,
                    col=2,
                    source="posterior_probability",
                    title="Posterior snow-cover probability",
                    show_hillshade=True,
                    hillshade_extent="roi",
                ),
                MapPanelSpec(kind=kind, row=row, col=3, title="Satellite FSC observation"),
            ),
        )
    return GeneratedRow(
        label=label,
        panels=(
            MapPanelSpec(kind=kind, row=row, col=0, source="open_loop", title="Open-loop WSF", show_hillshade=True, hillshade_extent="roi"),
            MapPanelSpec(
                kind=kind,
                row=row,
                col=1,
                source="prior_probability",
                title="Prior WSF",
                show_hillshade=True,
                hillshade_extent="roi",
            ),
            MapPanelSpec(
                kind=kind,
                row=row,
                col=2,
                source="posterior_probability",
                title="Posterior WSF",
                show_hillshade=True,
                hillshade_extent="roi",
            ),
            MapPanelSpec(kind=kind, row=row, col=3, title="Wet-snow observation"),
        ),
    )


def _wet_snow_line_row(*, row: int, label: str) -> GeneratedRow:
    return GeneratedRow(
        label=label,
        panels=(
            MapPanelSpec(
                kind="wet_snow_line",
                row=row,
                col=0,
                source="open_loop",
                title="Open-loop wet snow line",
                show_hillshade=True,
                hillshade_extent="roi",
            ),
            MapPanelSpec(
                kind="wet_snow_line",
                row=row,
                col=1,
                source="prior_probability",
                title="Prior wet snow line",
                show_hillshade=True,
                hillshade_extent="roi",
            ),
            MapPanelSpec(
                kind="wet_snow_line",
                row=row,
                col=2,
                source="posterior_probability",
                title="Posterior wet snow line",
                show_hillshade=True,
                hillshade_extent="roi",
            ),
            MapPanelSpec(kind="wet_snow_line", row=row, col=3, title="Observed wet snow line"),
        ),
    )


def _wet_snow_elevation_fraction_row(*, row: int, variable: str) -> GeneratedRow:
    return GeneratedRow(
        label="Elevation-band WSF",
        panels=(
            MapPanelSpec(
                kind="wet_snow_elevation_fraction",
                row=row,
                col=0,
                source="open_loop",
                title="Open-loop elevation-band WSF",
                variable=variable,
            ),
            MapPanelSpec(
                kind="wet_snow_elevation_fraction",
                row=row,
                col=1,
                source="prior_probability",
                title="Prior elevation-band WSF",
                variable=variable,
            ),
            MapPanelSpec(
                kind="wet_snow_elevation_fraction",
                row=row,
                col=2,
                source="posterior_probability",
                title="Posterior elevation-band WSF",
                variable=variable,
            ),
            MapPanelSpec(kind="wet_snow_elevation_fraction", row=row, col=3, title="Observed elevation-band WSF", variable=variable),
        ),
    )


def _generated_rows_for_event(project_dir: Path, event: AssimilationEvent) -> tuple[GeneratedRow, ...]:
    rows: list[GeneratedRow] = []
    row_index = 0
    if event.variable == "station_hs":
        rows.append(_snow_depth_row(row=row_index, label=_variable_label("station_hs"), event_variable=event.variable))
        row_index += 1
    elif event.variable == "station_swe":
        rows.append(_snow_depth_row(row=row_index, label=_variable_label("station_swe"), event_variable=event.variable))
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
        rows.append(_snow_depth_row(row=row_index, label=hs_label, event_variable=event.variable))
        row_index += 1
    elif event.variable == "wet_snow_line" and _fraction_model_support_available(project_dir, "wet_snow"):
        rows.append(_wet_snow_line_row(row=row_index, label=_stream_row_label("wet_snow_line")))
        row_index += 1
        rows.append(_wet_snow_elevation_fraction_row(row=row_index, variable="wet_snow_line"))
        row_index += 1
        hs_relation = _relation_for_variable(
            project_dir,
            variable="station_hs",
            date=pd.Timestamp(event.date).normalize(),
            require_summary_date=False,
        )
        hs_label = _stream_row_label("station_hs", hs_relation or "independent")
        rows.append(_snow_depth_row(row=row_index, label=hs_label, event_variable=event.variable))
        row_index += 1
    elif event.variable == "wet_snow" and _fraction_model_support_available(project_dir, "wet_snow"):
        rows.append(_fraction_row(row=row_index, kind="wet_snow", label=_stream_row_label("wet_snow")))
        row_index += 1
        rows.append(_wet_snow_elevation_fraction_row(row=row_index, variable="wet_snow"))
        row_index += 1
        hs_relation = _relation_for_variable(
            project_dir,
            variable="station_hs",
            date=pd.Timestamp(event.date).normalize(),
            require_summary_date=False,
        )
        hs_label = _stream_row_label("station_hs", hs_relation or "independent")
        rows.append(_snow_depth_row(row=row_index, label=hs_label, event_variable=event.variable))
        row_index += 1
    else:
        rows.append(_snow_depth_row(row=row_index, label=_variable_label("station_hs"), event_variable=event.variable))
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


def _is_top_level_subdomain_project(project_dir: Path) -> bool:
    manifest_path = Path(project_dir) / "subdomains" / "subdomain_manifest.json"
    return read_run_mode(project_dir) == "subdomain" and manifest_path.is_file()


def _top_level_subdomain_fraction_support_available(project_dir: Path, variable: str) -> bool:
    if variable != "scf":
        return False
    manifest_path = Path(project_dir) / "subdomains" / "subdomain_manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = SubdomainManifest.load(manifest_path)
    except Exception:
        return False
    if str(getattr(manifest, "run_mode", "")).lower() != "subdomain":
        return False
    if not manifest.subdomains:
        return False
    for sub in manifest.subdomains.values():
        if not (sub.project_dir / "steps").is_dir():
            return False
        summary = sub.setup_dir / "obs" / sub.project_name / "scf_summary.csv"
        if not summary.is_file():
            return False
    return True


def _use_large_subdomain_snow_layout(project_dir: Path, rows: tuple[GeneratedRow, ...]) -> bool:
    if len(rows) != 1 or not _is_top_level_subdomain_project(project_dir):
        return False
    panels = rows[0].panels
    return len(panels) == 4 and all(panel.kind == "snow_depth" for panel in panels)


def _use_top_level_subdomain_scf_layout(
    project_dir: Path,
    event: AssimilationEvent,
    rows: tuple[GeneratedRow, ...],
) -> bool:
    if event.variable != "scf" or not _is_top_level_subdomain_project(project_dir):
        return False
    if len(rows) < 2:
        return False
    return (
        len(rows[0].panels) == 4
        and all(panel.kind == "fsc" for panel in rows[0].panels)
        and len(rows[1].panels) == 4
        and all(panel.kind == "snow_depth" for panel in rows[1].panels)
    )


def _generated_panel_position(
    *,
    row_idx: int,
    panel_idx: int,
    panel: MapPanelSpec,
    large_snow_layout: bool,
    top_level_scf_layout: bool,
) -> tuple[int, int]:
    if top_level_scf_layout:
        if row_idx == 0:
            return ((0, 0), (1, 0), (1, 1), (0, 1))[panel_idx]
        if row_idx == 1:
            return ((2, 0), (3, 0), (3, 1), (2, 1))[panel_idx]
        return row_idx + 2, panel_idx
    if not large_snow_layout:
        return row_idx, panel.col
    return ((0, 0), (1, 0), (1, 1), (0, 1))[panel_idx]


def _generated_recipe(index: int, project_dir: Path, event: AssimilationEvent, rows: tuple[GeneratedRow, ...]) -> MapRecipe:
    use_large_subdomain_snow_layout = _use_large_subdomain_snow_layout(project_dir, rows)
    use_top_level_subdomain_scf_layout = _use_top_level_subdomain_scf_layout(project_dir, event, rows)
    if use_top_level_subdomain_scf_layout:
        layout = LayoutSpec(nrows=4, ncols=2)
    elif use_large_subdomain_snow_layout:
        layout = LayoutSpec(nrows=2, ncols=2)
    else:
        layout = LayoutSpec(nrows=len(rows), ncols=4)
    panels = tuple(
        MapPanelSpec(
            kind=panel.kind,
            row=position[0],
            col=position[1],
            title=panel.title,
            source=panel.source,
            show_hillshade=panel.show_hillshade,
            hillshade_extent=panel.hillshade_extent,
            variable=panel.variable,
        )
        for row_idx, row in enumerate(rows)
        for panel_idx, panel in enumerate(row.panels)
        for position in (
            _generated_panel_position(
                row_idx=row_idx,
                panel_idx=panel_idx,
                panel=panel,
                large_snow_layout=use_large_subdomain_snow_layout,
                top_level_scf_layout=use_top_level_subdomain_scf_layout,
            ),
        )
    )
    return MapRecipe(
        name=f"da_{index}",
        title=f"DA {index}",
        figure_title=_generated_figure_title(index, project_dir, event),
        output_subdir=GENERATED_DA_MAPS_SUBDIR,
        layout=layout,
        row_labels=(),
        defaults=MapDefaults(date=event.date.isoformat(), show_scalebar=True),
        panels=panels,
    )


def generated_da_map_recipes(project_dir: Path) -> tuple[MapRecipe, ...]:
    events = load_assimilation_events(project_dir)
    return tuple(
        _generated_recipe(index, project_dir, event, _generated_rows_for_event(project_dir, event))
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

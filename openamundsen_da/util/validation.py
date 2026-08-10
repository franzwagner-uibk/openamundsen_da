"""Validation helpers for DA runs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import rasterio

from openamundsen_da.observer.fraction_obs import resolve_obs_product_tag
from openamundsen_da.util.station_da import (
    is_station_variable,
    load_station_assimilation_config,
    read_station_metadata,
    station_observation_csvs,
)
from openamundsen_da.util.da_events import AssimilationEvent
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml, find_setup_yaml
from openamundsen_da.observer.summary_paths import resolve_fraction_summary_path
from openamundsen_da.util.roi_grid import load_setup_roi_mask


def _configured_model_point_ids(setup_dir: Path, setup_cfg: dict) -> dict[str, str]:
    """Return case-normalized point IDs that openAMUNDSEN will produce."""

    timeseries_cfg = ((setup_cfg.get("output_data") or {}).get("timeseries") or {})
    point_ids: dict[str, str] = {}
    for point in timeseries_cfg.get("points") or []:
        if not isinstance(point, dict) or point.get("name") is None:
            continue
        original = str(point["name"]).strip()
        if original:
            point_ids[original.lower()] = original

    if not bool(timeseries_cfg.get("add_default_points", True)):
        return point_ids

    stations_path = Path(setup_dir) / "meteo" / "stations.csv"
    if not stations_path.is_file():
        raise FileNotFoundError(
            "Cannot resolve output_data.timeseries.add_default_points: "
            f"missing meteo station metadata {stations_path}"
        )
    stations = pd.read_csv(stations_path, dtype={"id": "string"})
    required_columns = {"id", "x", "y"}
    missing_columns = sorted(required_columns - set(stations.columns))
    if missing_columns:
        raise ValueError(
            f"Cannot resolve default output points from {stations_path}: "
            f"missing columns {', '.join(missing_columns)}"
        )

    roi_mask, grid_spec, _ = load_setup_roi_mask(setup_dir, ensure_grid=True)
    try:
        rows, cols = rasterio.transform.rowcol(
            grid_spec.transform,
            pd.to_numeric(stations["x"], errors="raise").to_numpy(),
            pd.to_numeric(stations["y"], errors="raise").to_numpy(),
        )
    except Exception as exc:
        raise ValueError(f"Invalid meteo station coordinates in {stations_path}") from exc

    for station_id, row, col in zip(stations["id"], rows, cols, strict=True):
        if pd.isna(station_id):
            continue
        row_idx = int(row)
        col_idx = int(col)
        if not (0 <= row_idx < grid_spec.rows and 0 <= col_idx < grid_spec.cols):
            continue
        if not bool(roi_mask[row_idx, col_idx]):
            continue
        original = str(station_id).strip()
        if original:
            point_ids[original.lower()] = original
    return point_ids


def _validate_station_identity_contract(
    *,
    setup_dir: Path,
    setup_cfg: dict,
    obs_dir: Path,
    metadata_path: Path,
) -> list[str]:
    """Return errors for active station IDs lacking observations or model points."""

    if not metadata_path.is_file():
        return [f"Station assimilation requires metadata file {metadata_path}, but it does not exist."]

    raw_metadata = pd.read_csv(metadata_path, dtype={"station_id": "string"})
    metadata = read_station_metadata(metadata_path)
    original_ids: dict[str, str] = {}
    if "station_id" in raw_metadata.columns:
        for raw_station_id in raw_metadata["station_id"]:
            if pd.isna(raw_station_id):
                continue
            original = str(raw_station_id).strip()
            if original:
                original_ids[original.lower()] = original

    active_ids = {
        str(station_id)
        for station_id, row in metadata.iterrows()
        if bool(row.get("use_for_da", True)) or bool(row.get("use_for_benchmark", True))
    }
    if not active_ids:
        return []

    observation_ids = {
        path.stem.strip().lower(): path.stem
        for path in station_observation_csvs(obs_dir)
        if path.stem.strip()
    }
    point_ids = _configured_model_point_ids(setup_dir, setup_cfg)
    errors: list[str] = []
    missing_observations = sorted(
        original_ids.get(station_id, station_id)
        for station_id in active_ids - set(observation_ids)
    )
    if missing_observations:
        errors.append(
            "Active station IDs missing same-ID observation CSVs in "
            f"{obs_dir}: {', '.join(missing_observations)}"
        )
    missing_points = sorted(
        original_ids.get(station_id, station_id)
        for station_id in active_ids - set(point_ids)
    )
    if missing_points:
        errors.append(
            "Active station IDs missing same-ID model output points: "
            + ", ".join(missing_points)
        )
    return errors


def validate_assimilation_requirements(
    setup_dir: Path,
    project_dir: Path,
    steps: list[Path],
    events: list[AssimilationEvent],
) -> None:
    """Validate required config outputs and obs files before running a project."""
    proj_cfg = _read_yaml_file(find_setup_yaml(setup_dir)) or {}
    grid_vars = ((proj_cfg.get("output_data") or {}).get("grids") or {}).get("variables") or []
    instantaneous_names: set[str] = set()
    for entry in grid_vars:
        if not isinstance(entry, dict):
            continue
        if entry.get("name"):
            name = str(entry["name"])
            if entry.get("agg") is None:
                instantaneous_names.add(name)

    errors: list[str] = []

    needs_scf = any(ev.variable == "scf" for ev in events)
    if needs_scf and "snowdepth_instantaneous" not in instantaneous_names:
        errors.append(
            "Configure instantaneous snow depth output (var: snow.depth, "
            "name: snowdepth_instantaneous, without agg) in output_data.grids for SCF assimilation."
        )

    needs_wet = any(ev.variable in {"wet_snow", "wet_snow_line"} for ev in events)
    if needs_wet:
        for name, variable in (
            ("snowdepth_instantaneous", "snow.depth"),
            ("liquid_water_content_instantaneous", "snow.liquid_water_content"),
        ):
            if name not in instantaneous_names:
                errors.append(
                    f"Configure instantaneous output (var: {variable}, name: {name}, without agg) "
                    "in output_data.grids for wet-snow assimilation."
                )

    project_cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
    da_cfg = project_cfg.get("data_assimilation") or {}
    bench_cfg = da_cfg.get("benchmark") or {}
    benchmark_variables = set()
    if isinstance(bench_cfg, dict):
        for raw in bench_cfg.get("independent_variables") or []:
            key = str(raw).strip().lower()
            benchmark_variables.add("wet_snow" if key == "wet_snow_fraction" else key)
    event_variables = {("wet_snow" if ev.variable == "wet_snow_fraction" else ev.variable) for ev in events}
    for variable, filename in (
        ("scf", "scf_summary.csv"),
        ("wet_snow", "wet_snow_summary.csv"),
        ("wet_snow_line", "wet_snow_line_diagnostics.csv"),
    ):
        if variable not in event_variables and variable not in benchmark_variables:
            continue
        summary_path = resolve_fraction_summary_path(setup_dir, project_dir, filename)
        if not summary_path.is_file():
            owner = "snowcover" if variable == "scf" else "wetsnow"
            errors.append(
                f"Missing {variable} summary CSV required by post-processing/benchmark: {summary_path}. "
                f"Set obs.{owner}.summary_csv / obs.{owner}.wet_snow_line_diagnostics_csv or run the corresponding obs prep command."
            )

    needs_station = any(is_station_variable(ev.variable) for ev in events) or any(
        is_station_variable(variable) for variable in benchmark_variables
    )
    if needs_station:
        try:
            station_cfg = load_station_assimilation_config(setup_dir=setup_dir, project_dir=project_dir)
            if not station_cfg.obs_dir.is_dir():
                errors.append(
                    f"Station assimilation requires observation directory {station_cfg.obs_dir}, "
                    "but it does not exist."
                )
            else:
                station_obs_csvs = station_observation_csvs(station_cfg.obs_dir)
                if not station_obs_csvs:
                    errors.append(
                        f"Station assimilation requires at least one station observation CSV in {station_cfg.obs_dir}"
                    )
                else:
                    errors.extend(
                        _validate_station_identity_contract(
                            setup_dir=setup_dir,
                            setup_cfg=proj_cfg,
                            obs_dir=station_cfg.obs_dir,
                            metadata_path=station_cfg.metadata_path,
                        )
                    )
        except Exception as exc:
            errors.append(str(exc))

    max_idx = min(len(events), len(steps) - 1)
    for idx in range(max_idx):
        ev = events[idx]
        step_dir = Path(steps[idx])

        if is_station_variable(ev.variable):
            continue

        prod_tag = ev.product or resolve_obs_product_tag(ev.variable, setup_dir=setup_dir, project_dir=project_dir)
        base_name = f"obs_{ev.variable}_{ev.date.strftime('%Y%m%d')}.csv"
        prod_name = f"obs_{ev.variable}_{prod_tag}_{ev.date.strftime('%Y%m%d')}.csv"
        candidates = [step_dir / "obs" / base_name, step_dir / "obs" / prod_name]
        if not any(p.is_file() for p in candidates):
            expect = " or ".join(p.name for p in candidates)
            errors.append(f"{step_dir.name}: missing obs CSV for {ev.variable} ({ev.product}) on {ev.date} -> expected {expect}")

    if errors:
        raise ValueError("Config/obs/output validation failed:\n- " + "\n- ".join(errors))

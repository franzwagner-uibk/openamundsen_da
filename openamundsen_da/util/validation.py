"""Validation helpers for DA runs."""

from __future__ import annotations

from pathlib import Path

from openamundsen_da.observer.fraction_obs import resolve_obs_product_tag
from openamundsen_da.util.station_da import (
    is_station_variable,
    load_station_assimilation_config,
    station_observation_csvs,
)
from openamundsen_da.util.da_events import AssimilationEvent
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml, find_setup_yaml
from openamundsen_da.observer.summary_paths import resolve_fraction_summary_path


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

    if any(is_station_variable(ev.variable) for ev in events):
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

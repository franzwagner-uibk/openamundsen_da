"""Validation helpers for DA runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from openamundsen_da.observer.fraction_obs import resolve_obs_product_tag
from openamundsen_da.util.da_events import AssimilationEvent
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml, list_member_dirs


def _has_output_pattern(base_dirs: Iterable[Path], patterns: list[str]) -> bool:
    for root in base_dirs:
        for patt in patterns:
            if list(root.glob(patt)):
                return True
    return False


def validate_assimilation_requirements(
    project_dir: Path,
    season_dir: Path,
    steps: list[Path],
    events: list[AssimilationEvent],
) -> None:
    """Validate required config outputs and obs files before running a season."""
    proj_cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
    grid_vars = ((proj_cfg.get("output_data") or {}).get("grids") or {}).get("variables") or []
    names: set[str] = set()
    vars_: set[str] = set()
    for entry in grid_vars:
        if not isinstance(entry, dict):
            continue
        if entry.get("name"):
            names.add(str(entry["name"]))
        if entry.get("var"):
            vars_.add(str(entry["var"]))

    errors: list[str] = []

    needs_scf = any(ev.variable == "scf" for ev in events)
    if needs_scf and not ({"snowdepth_daily"} & names or {"snow.depth"} & vars_):
        errors.append("Configure snow depth daily output (var: snow.depth, name: snowdepth_daily) in output_data.grids for SCF assimilation.")

    needs_wet = any(ev.variable == "wet_snow" for ev in events)
    if needs_wet and not ({"liquid_water_content"} & names or {"snow.liquid_water_content"} & vars_):
        errors.append("Configure liquid water content output (var: snow.liquid_water_content, name: liquid_water_content) in output_data.grids for wet-snow assimilation.")

    max_idx = min(len(events), len(steps) - 1)
    for idx in range(max_idx):
        ev = events[idx]
        step_dir = Path(steps[idx])

        prod_tag = ev.product or resolve_obs_product_tag(ev.variable, project_dir=project_dir)
        base_name = f"obs_{ev.variable}_{ev.date.strftime('%Y%m%d')}.csv"
        prod_name = f"obs_{ev.variable}_{prod_tag}_{ev.date.strftime('%Y%m%d')}.csv"
        candidates = [step_dir / "obs" / base_name, step_dir / "obs" / prod_name]
        if not any(p.is_file() for p in candidates):
            expect = " or ".join(p.name for p in candidates)
            errors.append(f"{step_dir.name}: missing obs CSV for {ev.variable} ({ev.product}) on {ev.date} -> expected {expect}")

    if errors:
        raise ValueError("Config/obs/output validation failed:\n- " + "\n- ".join(errors))

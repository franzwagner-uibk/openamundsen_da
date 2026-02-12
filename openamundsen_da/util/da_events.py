"""Helpers for loading assimilation events from project YAML."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from openamundsen_da.io.paths import find_project_yaml
from openamundsen_da.observer.fraction_obs import resolve_obs_product_tag


@dataclass(frozen=True)
class AssimilationEvent:
    date: date
    variable: str
    product: str


def _read_yaml(path: Path) -> dict:
    """Read a YAML file into a dict (best-effort)."""
    try:
        import ruamel.yaml as _yaml

        y = _yaml.YAML(typ="safe")
        with path.open("r", encoding="utf-8") as f:
            return y.load(f) or {}
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not read YAML from {path}: {exc}") from exc


def _parse_event_date(text: str | None) -> date:
    if not text:
        raise ValueError("Empty assimilation event date")
    t = str(text).strip()
    try:
        dt = datetime.strptime(t, "%Y-%m-%d")
        return dt.date()
    except Exception as exc:
        raise ValueError(f"Invalid assimilation event date (expected YYYY-MM-DD): {text}") from exc


def load_assimilation_events(project_dir: Path) -> list[AssimilationEvent]:
    """Load assimilation events from project YAML (variable/product per date)."""
    project_yaml = find_project_yaml(project_dir)
    cfg = _read_yaml(project_yaml) or {}
    setup_dir = project_dir.parent.parent if project_dir.parent.parent.is_dir() else None

    events: list[AssimilationEvent] = []

    da_cfg = cfg.get("data_assimilation") or {}
    raw_events = da_cfg.get("assimilation_events") or []
    for entry in raw_events:
        if not isinstance(entry, dict):
            continue
        dtxt = entry.get("date")
        if not dtxt:
            continue
        dval = _parse_event_date(str(dtxt))
        var = str(entry.get("variable") or "scf")
        default_prod = resolve_obs_product_tag(var, setup_dir=setup_dir, project_dir=project_dir)
        if "product" in entry and entry["product"] is not None:
            prod = str(entry["product"])
        else:
            prod = default_prod

        prod_upper = prod.upper()
        if prod_upper in {"MOD10A1", "SNOWFLAKE", "SNOWFLAKES"} and var == "scf":
            prod_upper = default_prod
        if prod_upper in {"S1"} and var == "wet_snow":
            prod_upper = default_prod

        events.append(AssimilationEvent(date=dval, variable=var, product=prod_upper))

    if not events:
        raise ValueError(f"No assimilation_events found in {project_yaml}")

    events.sort(key=lambda ev: ev.date)
    return events



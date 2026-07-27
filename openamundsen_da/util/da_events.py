"""Helpers for loading assimilation events from project YAML."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from openamundsen_da.io.paths import find_project_yaml
from openamundsen_da.observer.fraction_obs import resolve_obs_product_tag
from openamundsen_da.util.observation_time import parse_utc_timestamp
from openamundsen_da.util.station_da import is_station_variable
from openamundsen_da.util.yaml_utils import read_yaml_mapping


@dataclass(frozen=True)
class AssimilationEvent:
    date: date
    variable: str
    product: str
    observation_time: datetime | None = None


def _parse_event_date(text: str | None) -> date:
    if not text:
        raise ValueError("Empty assimilation event date")
    t = str(text).strip()
    try:
        dt = datetime.strptime(t, "%Y-%m-%d")
        return dt.date()
    except Exception as exc:
        raise ValueError(f"Invalid assimilation event date (expected YYYY-MM-DD): {text}") from exc


def _parse_event_variable(raw: object, *, idx: int) -> str:
    value = str(raw).strip().lower()
    if not value:
        raise ValueError(f"Missing required configuration key: data_assimilation.assimilation_events[{idx}].variable")
    if value == "wet_snow_fraction":
        value = "wet_snow"
    if value not in {"scf", "wet_snow", "wet_snow_line"} and not is_station_variable(value):
        raise ValueError(
            f"Unsupported assimilation variable at data_assimilation.assimilation_events[{idx}].variable: {raw!r}"
        )
    return value


def load_assimilation_events(project_dir: Path) -> list[AssimilationEvent]:
    """Load assimilation events from project YAML (variable/product per date)."""
    project_yaml = find_project_yaml(project_dir)
    cfg = read_yaml_mapping(project_yaml, error_cls=RuntimeError, context="Project YAML root")
    setup_dir = project_dir.parent.parent if project_dir.parent.parent.is_dir() else None

    events: list[AssimilationEvent] = []

    da_cfg = cfg.get("data_assimilation")
    if not isinstance(da_cfg, dict):
        raise ValueError(f"Missing required configuration key: {project_yaml} -> data_assimilation")
    raw_events = da_cfg.get("assimilation_events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"No assimilation_events found in {project_yaml}")

    for idx, entry in enumerate(raw_events, start=1):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Expected mapping at data_assimilation.assimilation_events[{idx}], got {type(entry).__name__}"
            )
        if "date" not in entry:
            raise ValueError(f"Missing required configuration key: data_assimilation.assimilation_events[{idx}].date")
        dtxt = entry.get("date")
        if dtxt is None or str(dtxt).strip() == "":
            raise ValueError(
                f"Configuration value must not be empty: data_assimilation.assimilation_events[{idx}].date"
            )
        dval = _parse_event_date(str(dtxt))
        observation_time = None
        if entry.get("observation_time") is not None:
            observation_time = parse_utc_timestamp(
                entry["observation_time"],
                field=f"data_assimilation.assimilation_events[{idx}].observation_time",
            )
            if observation_time.date() != dval:
                raise ValueError(
                    f"data_assimilation.assimilation_events[{idx}].observation_time has UTC date "
                    f"{observation_time.date()}, expected {dval}"
                )
        var = _parse_event_variable(entry.get("variable"), idx=idx)
        if "product" in entry and entry["product"] is not None:
            prod = str(entry["product"]).strip()
            if not prod:
                raise ValueError(
                    f"Configuration value must not be empty: data_assimilation.assimilation_events[{idx}].product"
                )
            prod_upper = prod.upper()
        elif is_station_variable(var):
            prod_upper = "STATION"
        else:
            prod_upper = resolve_obs_product_tag(var, setup_dir=setup_dir, project_dir=project_dir)

        events.append(
            AssimilationEvent(
                date=dval,
                variable=var,
                product=prod_upper,
                observation_time=observation_time,
            )
        )

    events.sort(key=lambda ev: ev.date)
    return events

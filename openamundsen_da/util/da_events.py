"""Helpers for loading assimilation events from season.yml."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from openamundsen_da.io.paths import find_season_yaml


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


def load_assimilation_events(season_dir: Path) -> list[AssimilationEvent]:
    """Load assimilation events from season.yml (variable/product per date).

    Supports both the new structured block:

    data_assimilation:
      assimilation_events:
        - date: 2020-03-19
          variable: scf
          product: MOD10A1

    and the legacy flat list:

    assimilation_dates:
      - 2020-03-19
      - 2020-04-06

    In the legacy case, all events use variable='scf' and product='MOD10A1'.
    """
    season_yaml = find_season_yaml(season_dir)
    cfg = _read_yaml(season_yaml) or {}

    events: list[AssimilationEvent] = []

    da_cfg = cfg.get("data_assimilation") or {}
    raw_events = da_cfg.get("assimilation_events") or []
    if isinstance(raw_events, list) and raw_events:
        for entry in raw_events:
            if not isinstance(entry, dict):
                continue
            dtxt = entry.get("date")
            if not dtxt:
                continue
            dval = _parse_event_date(str(dtxt))
            var = str(entry.get("variable") or "scf")
            if "product" in entry and entry["product"] is not None:
                prod = str(entry["product"])
            else:
                prod = "MOD10A1" if var == "scf" else "S1"
            events.append(AssimilationEvent(date=dval, variable=var, product=prod))
    else:
        raw_dates = cfg.get("assimilation_dates") or []
        for text in raw_dates:
            dval = _parse_event_date(str(text))
            events.append(AssimilationEvent(date=dval, variable="scf", product="MOD10A1"))

    if not events:
        raise ValueError(f"No assimilation_events or assimilation_dates found in {season_yaml}")

    events.sort(key=lambda ev: ev.date)
    return events

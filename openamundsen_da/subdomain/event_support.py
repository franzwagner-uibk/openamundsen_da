"""Strict final-event support contracts for subdomain workflows."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from openamundsen_da.subdomain.manifest import SubdomainManifest, SubdomainMeta
from openamundsen_da.util.da_events import AssimilationEvent, load_assimilation_events
from openamundsen_da.util.da_observables import weights_csv_name
from openamundsen_da.util.ts import parse_datetime_opt


class SubdomainEventSupportError(RuntimeError):
    """Raised when leaf event plans cannot support the top-level schedule."""


def _event_key(event: AssimilationEvent) -> tuple[str, str]:
    return event.date.isoformat(), str(event.variable).strip().lower()


def _row_key(row: dict) -> tuple[str, str]:
    raw_date = row.get("date")
    parsed = parse_datetime_opt(str(raw_date)) if raw_date is not None else None
    if parsed is None:
        raise SubdomainEventSupportError(
            f"Dropped-event row has an invalid date: {raw_date!r}"
        )
    variable = str(row.get("variable") or "").strip().lower()
    if not variable:
        raise SubdomainEventSupportError(f"Dropped-event row has no variable: {row!r}")
    return parsed.date().isoformat(), variable


def _leaf_dropped_events(subdomain: SubdomainMeta) -> list[dict]:
    rows = list(subdomain.dropped_events or [])
    if rows:
        return rows
    path = subdomain.setup_dir / "subdomain_dropped_events.csv"
    if not path.is_file():
        return []
    frame = pd.read_csv(
        path, dtype={"subdomain_id": "string", "active_station_ids": "string"}
    )
    return frame.fillna("").to_dict(orient="records")


def _event_time(project_yaml: Path, event: AssimilationEvent) -> str:
    from openamundsen_da.core.env import _read_yaml_file

    cfg = _read_yaml_file(project_yaml) or {}
    raw_start = cfg.get("start_date")
    start = parse_datetime_opt(str(raw_start)) if raw_start is not None else None
    if start is None:
        raise SubdomainEventSupportError(
            f"Project start_date is required to resolve event support timestamps: {project_yaml}"
        )
    return datetime.combine(event.date, start.time()).isoformat(sep=" ")


def _weight_artifact(subdomain: SubdomainMeta, event: AssimilationEvent) -> Path:
    filename = weights_csv_name(
        event.variable,
        datetime.combine(event.date, datetime.min.time()),
    )
    matches = sorted((subdomain.project_dir / "steps").glob(f"step_*/assim/{filename}"))
    if len(matches) != 1:
        raise SubdomainEventSupportError(
            f"Supporting leaf {subdomain.id} must contain exactly one weights artifact for "
            f"{event.date.isoformat()} {event.variable}; found {len(matches)} ({filename})"
        )
    return matches[0]


def resolve_subdomain_event_plan(
    manifest: SubdomainManifest,
    *,
    require_artifacts: bool,
) -> list[dict[str, object]]:
    """Resolve one explicit kept/dropped row per top-level event and leaf."""
    top_events = load_assimilation_events(manifest.project_dir)
    top_by_key = {_event_key(event): event for event in top_events}
    if len(top_by_key) != len(top_events):
        raise SubdomainEventSupportError(
            "Top-level assimilation events are not unique by date and variable"
        )

    rows: list[dict[str, object]] = []
    support_counts = {key: 0 for key in top_by_key}
    for subdomain_id, subdomain in sorted(manifest.subdomains.items()):
        leaf_events = load_assimilation_events(subdomain.project_dir)
        leaf_by_key = {_event_key(event): event for event in leaf_events}
        if len(leaf_by_key) != len(leaf_events):
            raise SubdomainEventSupportError(
                f"Leaf {subdomain_id} assimilation events are not unique by date and variable"
            )
        extra = sorted(set(leaf_by_key) - set(top_by_key))
        if extra:
            raise SubdomainEventSupportError(
                f"Leaf {subdomain_id} contains events absent from the top-level schedule: {extra}"
            )

        dropped_by_key: dict[tuple[str, str], dict] = {}
        for dropped in _leaf_dropped_events(subdomain):
            key = _row_key(dropped)
            if key in dropped_by_key:
                raise SubdomainEventSupportError(
                    f"Leaf {subdomain_id} has duplicate dropped-event rows for {key[0]} {key[1]}"
                )
            dropped_by_key[key] = dict(dropped)
        extra_dropped = sorted(set(dropped_by_key) - set(top_by_key))
        if extra_dropped:
            raise SubdomainEventSupportError(
                f"Leaf {subdomain_id} records dropped events absent from the top-level schedule: {extra_dropped}"
            )

        for key, top_event in top_by_key.items():
            kept_event = leaf_by_key.get(key)
            dropped = dropped_by_key.get(key)
            if kept_event is not None and dropped is not None:
                raise SubdomainEventSupportError(
                    f"Leaf {subdomain_id} marks {key[0]} {key[1]} as both kept and dropped"
                )
            if kept_event is not None:
                if kept_event.product.upper() != top_event.product.upper():
                    raise SubdomainEventSupportError(
                        f"Leaf {subdomain_id} product mismatch for {key[0]} {key[1]}: "
                        f"{kept_event.product!r} != {top_event.product!r}"
                    )
                if require_artifacts:
                    _weight_artifact(subdomain, kept_event)
                support_counts[key] += 1
                row: dict[str, object] = {
                    "subdomain_id": subdomain_id,
                    "date": key[0],
                    "assimilation_time": _event_time(
                        subdomain.project_yaml, kept_event
                    ),
                    "variable": key[1],
                    "product": kept_event.product,
                    "reason": "",
                    "metric": "",
                    "value": "",
                    "threshold": "",
                    "active_station_ids": "",
                    "project_yaml": str(subdomain.project_yaml),
                    "status": "kept",
                }
            else:
                row = {
                    "subdomain_id": subdomain_id,
                    "date": key[0],
                    "assimilation_time": str(
                        (dropped or {}).get("assimilation_time")
                        or _event_time(subdomain.project_yaml, top_event)
                    ),
                    "variable": key[1],
                    "product": str((dropped or {}).get("product") or top_event.product),
                    "reason": str(
                        (dropped or {}).get("reason")
                        or "not_configured_in_leaf_assimilation_events"
                    ),
                    "metric": str((dropped or {}).get("metric") or ""),
                    "value": (dropped or {}).get("value", ""),
                    "threshold": (dropped or {}).get("threshold", ""),
                    "active_station_ids": str((dropped or {}).get("active_station_ids") or ""),
                    "project_yaml": str(subdomain.project_yaml),
                    "status": "dropped",
                }
            rows.append(row)

    unsupported = [
        f"{date} {variable}"
        for (date, variable), count in support_counts.items()
        if count == 0
    ]
    if unsupported:
        raise SubdomainEventSupportError(
            "Top-level assimilation events have no supporting subdomain: "
            + "; ".join(unsupported)
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["subdomain_id"]),
            str(row["date"]),
            str(row["variable"]),
            str(row["status"]),
        ),
    )


__all__ = [
    "SubdomainEventSupportError",
    "resolve_subdomain_event_plan",
]

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openamundsen_da.exceptions import ProjectRenderError
from openamundsen_da.subdomain.event_support import (
    SubdomainEventSupportError,
    resolve_subdomain_event_plan,
)
from openamundsen_da.subdomain import render as render_mod


def _write_project(project_dir: Path, events: list[tuple[str, str]]) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "start_date: '2024-01-01 00:00:00'",
        "end_date: '2024-02-01 00:00:00'",
        "data_assimilation:",
        "  assimilation_events:",
    ]
    for date, variable in events:
        lines.extend(
            [
                f"    - date: '{date}'",
                f"      variable: {variable}",
                *(["      product: SNOWCOVER"] if variable == "scf" else []),
            ]
        )
    (project_dir / f"{project_dir.name}.yml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_weight(project_dir: Path, date: str, variable: str) -> None:
    path = (
        project_dir
        / "steps"
        / "step_00_init"
        / "assim"
        / f"weights_{variable}_{date.replace('-', '')}.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("member_id,weight\nmember_000,1.0\n", encoding="utf-8")


def _mixed_manifest(tmp_path: Path) -> SimpleNamespace:
    project_dir = tmp_path / "setup" / "projects" / "winter"
    _write_project(project_dir, [("2024-01-03", "scf"), ("2024-01-10", "station_hs")])

    subdomains = {}
    for subdomain_id, events in {
        "sd_01": [("2024-01-03", "scf"), ("2024-01-10", "station_hs")],
        "sd_02": [("2024-01-10", "station_hs")],
    }.items():
        setup_dir = project_dir / "subdomains" / subdomain_id
        leaf_project = setup_dir / "projects" / "winter"
        _write_project(leaf_project, events)
        for date, variable in events:
            _write_weight(leaf_project, date, variable)
        dropped = []
        if subdomain_id == "sd_02":
            dropped.append(
                {
                    "subdomain_id": subdomain_id,
                    "date": "2024-01-03",
                    "assimilation_time": "2024-01-03 00:00:00",
                    "variable": "scf",
                    "product": "SNOWCOVER",
                    "reason": "cloud_fraction_above_threshold",
                    "metric": "cloud_reference_fraction",
                    "value": 0.3,
                    "threshold": 0.2,
                    "active_station_ids": "",
                    "project_yaml": str(leaf_project / "winter.yml"),
                }
            )
        subdomains[subdomain_id] = SimpleNamespace(
            id=subdomain_id,
            status="success",
            setup_dir=setup_dir,
            project_dir=leaf_project,
            project_yaml=leaf_project / "winter.yml",
            dropped_events=dropped,
        )
    return SimpleNamespace(project_dir=project_dir, subdomains=subdomains)


def test_final_render_accepts_mixed_supported_and_dropped_leaves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _mixed_manifest(tmp_path)
    project_dir = manifest.project_dir
    merged = project_dir / "results" / "grids" / "da_output_grids.nc"
    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_bytes(b"netcdf")
    calls: list[str] = []

    monkeypatch.setattr(render_mod, "ensure_run_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(render_mod, "save_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        render_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(
        render_mod, "write_subdomain_reports", lambda **kwargs: calls.append("tables")
    )
    monkeypatch.setattr(render_mod, "project_maps_enabled", lambda _project: False)

    def _report(**_kwargs) -> Path:
        calls.append("report")
        path = project_dir / "results" / "reports" / "project_report.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pdf")
        return path

    monkeypatch.setattr(render_mod, "build_project_collection_pdf", _report)
    monkeypatch.setattr(
        render_mod,
        "cleanup_compact_grid_artifacts",
        lambda **_kwargs: calls.append("cleanup") or ([], 0),
    )

    plan = resolve_subdomain_event_plan(manifest, require_artifacts=True)
    render_mod.render_subdomain_outputs(project_dir)

    scf_rows = [row for row in plan if row["date"] == "2024-01-03"]
    assert [(row["subdomain_id"], row["status"]) for row in scf_rows] == [
        ("sd_01", "kept"),
        ("sd_02", "dropped"),
    ]
    assert calls == ["tables", "report", "cleanup"]


def test_leaf_yaml_omission_is_sufficient_to_mark_event_unsupported(tmp_path: Path) -> None:
    manifest = _mixed_manifest(tmp_path)
    leaf = manifest.subdomains["sd_02"]
    leaf.dropped_events = []

    rows = resolve_subdomain_event_plan(manifest, require_artifacts=True)

    omitted = [
        row
        for row in rows
        if row["subdomain_id"] == "sd_02" and row["date"] == "2024-01-03"
    ]
    assert len(omitted) == 1
    assert omitted[0]["status"] == "dropped"
    assert omitted[0]["reason"] == "not_configured_in_leaf_assimilation_events"


def test_final_event_plan_rejects_event_without_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _mixed_manifest(tmp_path)
    subdomain = manifest.subdomains["sd_01"]
    _write_project(subdomain.project_dir, [("2024-01-10", "station_hs")])
    subdomain.dropped_events.append(
        {
            "subdomain_id": "sd_01",
            "date": "2024-01-03",
            "variable": "scf",
            "product": "SNOWCOVER",
            "reason": "cloud_fraction_above_threshold",
        }
    )

    with pytest.raises(SubdomainEventSupportError, match="no supporting subdomain"):
        resolve_subdomain_event_plan(manifest, require_artifacts=True)

    monkeypatch.setattr(render_mod, "ensure_run_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        render_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    with pytest.raises(ProjectRenderError, match="no supporting subdomain"):
        render_mod.render_subdomain_outputs(manifest.project_dir)

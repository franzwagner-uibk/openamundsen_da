from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from openamundsen_da.subdomain import report as report_mod


def test_write_subdomain_reports_writes_overview_and_assimilation_stats(tmp_path: Path, monkeypatch) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    sub_setup_dir = project_dir / "subdomains" / "sd_01"
    sub_project_dir = sub_setup_dir / "projects" / "project_2022_2023"
    step_dir = sub_project_dir / "steps" / "step_01_20221001-20221003"
    assim_dir = step_dir / "assim"
    assim_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "step_01_20221001-20221003.yml").write_text(
        "start_date: 2022-10-01 00:00:00\nend_date: 2022-10-03 21:00:00\n",
        encoding="utf-8",
    )

    weights_csv = assim_dir / "weights_scf_20221003.csv"
    weights_csv.write_text(
        "member_id,weight,sigma\nmember_001,0.7,0.15\nmember_002,0.2,0.15\nmember_003,0.1,0.15\n",
        encoding="utf-8",
    )

    run_manifest_path = sub_setup_dir / "run_manifest.json"
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path.write_text(
        json.dumps({"status": "success", "duration_seconds": 12.5}),
        encoding="utf-8",
    )

    sub = SimpleNamespace(
        id="sd_01",
        label="sd_01",
        setup_dir=sub_setup_dir,
        project_dir=sub_project_dir,
        status="success",
        run_manifest=run_manifest_path,
        station_counts={
            "obs_stations_selected": 3,
            "obs_stations_inside_grid": 2,
            "obs_stations_da_active": 2,
            "obs_stations_benchmark_active": 1,
            "obs_station_series_copied": 3,
        },
    )
    manifest = SimpleNamespace(
        run_mode="subdomain",
        project_dir=project_dir,
        subdomains={"sd_01": sub},
    )

    monkeypatch.setattr(
        report_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(report_mod, "ensure_run_mode", lambda *args, **kwargs: "subdomain")
    monkeypatch.setattr(report_mod, "load_assimilation_events", lambda _project_dir: [object()])

    out_dir = project_dir / "results"
    outputs = report_mod.write_subdomain_reports(
        manifest_path=tmp_path / "subdomain_manifest.json",
        out_dir=out_dir,
    )

    assert outputs["overview"] == out_dir / "subdomain_overview.csv"
    assert outputs["assimilation_stats"] == out_dir / "subdomain_assimilation_stats.csv"
    assert outputs["assimilation_aggregate"] == out_dir / "subdomain_assimilation_aggregate.csv"

    overview_df = pd.read_csv(outputs["overview"])
    assert list(overview_df["subdomain_id"]) == ["sd_01"]
    assert list(overview_df["status"]) == ["success"]
    assert int(overview_df.loc[0, "obs_stations_selected"]) == 3
    assert int(overview_df.loc[0, "obs_stations_benchmark_active"]) == 1

    stats_df = pd.read_csv(outputs["assimilation_stats"])
    assert list(stats_df["variable"]) == ["scf"]
    assert list(stats_df["date"]) == ["2022-10-03"]
    assert int(stats_df.loc[0, "n_members"]) == 3
    assert float(stats_df.loc[0, "sigma"]) == 0.15

    agg_df = pd.read_csv(outputs["assimilation_aggregate"])
    assert list(agg_df["subdomain_id"]) == ["sd_01"]
    assert int(agg_df.loc[0, "events_count"]) == 1

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from openamundsen_da.subdomain.event_filter import filter_project_events_for_subdomain


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_subdomain_event_filter_drops_scf_by_invalid_fraction(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_yaml = tmp_path / "project" / "project.yml"
    _write(
        project_yaml,
        """
        data_assimilation:
          subdomain_event_filter:
            enabled: true
            drop_unavailable: true
            variables:
              scf:
                max_invalid_fraction: 0.10
          assimilation_events:
            - date: '2024-01-03'
              variable: scf
              product: SNOWCOVER
            - date: '2024-01-10'
              variable: scf
              product: SNOWCOVER
        """,
    )
    summary_dir = setup_dir / "obs" / "project_2024"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2024-01-03", "scf": 0.5, "invalid_fraction": 0.05},
            {"date": "2024-01-10", "scf": 0.6, "invalid_fraction": 0.20},
        ]
    ).to_csv(summary_dir / "scf_summary.csv", index=False)

    dropped = filter_project_events_for_subdomain(
        project_yaml=project_yaml,
        setup_dir=setup_dir,
        project_name="project_2024",
        subdomain_id="sd_01",
        dropped_events_csv=tmp_path / "dropped.csv",
    )

    assert len(dropped) == 1
    assert dropped[0]["date"] == "2024-01-10"
    assert dropped[0]["reason"] == "invalid_fraction_above_threshold"
    cfg = yaml.safe_load(project_yaml.read_text(encoding="utf-8"))
    events = cfg["data_assimilation"]["assimilation_events"]
    assert [event["date"] for event in events] == ["2024-01-03"]
    dropped_csv = pd.read_csv(tmp_path / "dropped.csv")
    assert dropped_csv.loc[0, "subdomain_id"] == "sd_01"


def test_subdomain_event_filter_supports_cloud_fraction_override(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_yaml = tmp_path / "project" / "project.yml"
    _write(
        project_yaml,
        """
        data_assimilation:
          subdomain_event_filter:
            enabled: true
            drop_unavailable: true
            variables:
              scf:
                max_cloud_fraction: 0.20
            subdomains:
              sd_01:
                variables:
                  scf:
                    max_cloud_fraction: 0.25
          assimilation_events:
            - date: '2024-01-03'
              variable: scf
              product: SNOWCOVER
            - date: '2024-01-10'
              variable: scf
              product: SNOWCOVER
        """,
    )
    summary_dir = setup_dir / "obs" / "project_2024"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"date": "2024-01-03", "scf": 0.5, "cloud_fraction": 0.23},
            {"date": "2024-01-10", "scf": 0.6, "cloud_fraction": 0.27},
        ]
    ).to_csv(summary_dir / "scf_summary.csv", index=False)

    dropped = filter_project_events_for_subdomain(
        project_yaml=project_yaml,
        setup_dir=setup_dir,
        project_name="project_2024",
        subdomain_id="sd_01",
        dropped_events_csv=tmp_path / "dropped.csv",
    )

    assert len(dropped) == 1
    assert dropped[0]["reason"] == "cloud_fraction_above_threshold"
    cfg = yaml.safe_load(project_yaml.read_text(encoding="utf-8"))
    events = cfg["data_assimilation"]["assimilation_events"]
    assert [event["date"] for event in events] == ["2024-01-03"]


def test_subdomain_event_filter_respects_station_da_role(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_yaml = tmp_path / "project" / "project.yml"
    _write(
        project_yaml,
        """
        data_assimilation:
          subdomain_event_filter:
            enabled: true
            drop_unavailable: true
            variables:
              station_hs:
                min_active_stations: 1
                max_time_delta_hours: 12
          assimilation_events:
            - date: '2024-01-03'
              variable: station_hs
            - date: '2024-01-10'
              variable: station_hs
        """,
    )
    stations_dir = setup_dir / "obs" / "stations"
    stations_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "station_id": "station_a",
                "station_uncertainty_pct": 10,
                "hs_sigma_abs_min": 0.1,
                "use_for_da": False,
            },
            {
                "station_id": "station_b",
                "station_uncertainty_pct": 10,
                "hs_sigma_abs_min": 0.1,
                "use_for_da": True,
            },
        ]
    ).to_csv(stations_dir / "stations_da_metadata.csv", index=False)
    pd.DataFrame(
        [{"time": "2024-01-03 00:00:00", "snow_depth": 1.0}]
    ).to_csv(stations_dir / "station_a.csv", index=False)
    pd.DataFrame(
        [{"time": "2024-01-10 00:00:00", "snow_depth": 1.0}]
    ).to_csv(stations_dir / "station_b.csv", index=False)

    dropped = filter_project_events_for_subdomain(
        project_yaml=project_yaml,
        setup_dir=setup_dir,
        project_name="project_2024",
        subdomain_id="sd_01",
        dropped_events_csv=tmp_path / "dropped.csv",
    )

    assert len(dropped) == 1
    assert dropped[0]["variable"] == "station_hs"
    assert dropped[0]["reason"] == "active_station_count_below_minimum"
    cfg = yaml.safe_load(project_yaml.read_text(encoding="utf-8"))
    events = cfg["data_assimilation"]["assimilation_events"]
    assert [event["date"] for event in events] == ["2024-01-10"]


def test_subdomain_event_filter_prunes_station_benchmark_for_stationless_subdomain(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_yaml = tmp_path / "project" / "project.yml"
    _write(
        project_yaml,
        """
        data_assimilation:
          benchmark:
            enabled: true
            variables: [scf, station_hs]
            independent_variables: [station_hs]
            performance_scores_exclude_variables: [station_hs]
          subdomain_event_filter:
            enabled: true
            drop_unavailable: true
            variables:
              scf:
                max_cloud_fraction: 0.20
              station_hs:
                min_active_stations: 1
          assimilation_events:
            - date: '2024-01-03'
              variable: scf
              product: SNOWCOVER
            - date: '2024-01-10'
              variable: station_hs
              product: STATION
        """,
    )
    summary_dir = setup_dir / "obs" / "project_2024"
    summary_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"date": "2024-01-03", "scf": 0.5, "cloud_fraction": 0.05}]
    ).to_csv(summary_dir / "scf_summary.csv", index=False)

    dropped = filter_project_events_for_subdomain(
        project_yaml=project_yaml,
        setup_dir=setup_dir,
        project_name="project_2024",
        subdomain_id="sd_01",
        dropped_events_csv=tmp_path / "dropped.csv",
    )

    assert len(dropped) == 1
    assert dropped[0]["variable"] == "station_hs"
    cfg = yaml.safe_load(project_yaml.read_text(encoding="utf-8"))
    da_cfg = cfg["data_assimilation"]
    assert da_cfg["assimilation_events"] == [
        {"date": "2024-01-03", "variable": "scf", "product": "SNOWCOVER"}
    ]
    assert da_cfg["benchmark"]["variables"] == ["scf"]
    assert da_cfg["benchmark"]["independent_variables"] == []
    assert da_cfg["benchmark"]["performance_scores_exclude_variables"] == []


def test_subdomain_event_filter_raises_when_disabled_event_is_unavailable(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_yaml = tmp_path / "project" / "project.yml"
    _write(
        project_yaml,
        """
        data_assimilation:
          assimilation_events:
            - date: '2024-01-03'
              variable: scf
              product: SNOWCOVER
        """,
    )

    try:
        filter_project_events_for_subdomain(
            project_yaml=project_yaml,
            setup_dir=setup_dir,
            project_name="project_2024",
            subdomain_id="sd_01",
            dropped_events_csv=tmp_path / "dropped.csv",
        )
    except ValueError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("Expected unavailable event to fail when filtering is disabled")

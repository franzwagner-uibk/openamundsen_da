from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBDOMAIN_ROOT = REPO_ROOT / "examples" / "subdomains"
SUBDOMAIN_PROJECT_YAML = (
    SUBDOMAIN_ROOT
    / "projects"
    / "project_2022_2023"
    / "project_2022_2023.yml"
)


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_shipped_subdomain_example_uses_current_project_tunes() -> None:
    cfg = _read_yaml(SUBDOMAIN_PROJECT_YAML)
    da_cfg = cfg["data_assimilation"]

    assert cfg["run_mode"] == "subdomain"
    assert da_cfg["resampling"]["ess_threshold_ratio"] == 0.7
    assert da_cfg["output"]["retention"] == "full"
    assert da_cfg["landcover_mask"]["classes_to_exclude"] == [2, 3, 13]
    assert da_cfg["prior_forcing"]["humidity_perturbation_method"] == "dew_point"
    assert da_cfg["rejuvenation"]["humidity_perturbation_method"] == "dew_point"

    scf_unc = da_cfg["uncertainty"]["scf"]
    assert scf_unc["ingest"] == {
        "scf_variable": "fsc",
        "time_variable": "time",
        "uncertainty_source": "internal",
    }
    assert scf_unc["assimilation"] == {
        "sigma_mode": "uncertainty_layer",
        "aggregate_metric": "unc_mean",
    }
    assert scf_unc["u_min"] == 5.0
    assert scf_unc["u_max"] == 20.0
    assert scf_unc["penalties"] == [
        {
            "name": "forest",
            "source": "landcover",
            "enabled": True,
            "classes": [8, 9, 10, 11, 12],
            "penalty": 20.0,
        }
    ]

    expected_sigmas = {
        "sigma_t": 0.5,
        "sigma_p": 0.5,
        "sigma_rh": 0.5,
        "sigma_sw": 0.05,
    }
    assert {key: da_cfg["prior_forcing"][key] for key in expected_sigmas} == expected_sigmas
    assert {key: da_cfg["rejuvenation"][key] for key in expected_sigmas} == expected_sigmas


def test_shipped_subdomain_example_uses_retuned_da_event_schedule() -> None:
    cfg = _read_yaml(SUBDOMAIN_PROJECT_YAML)
    events = cfg["data_assimilation"]["assimilation_events"]

    expected_station_dates = [
        "2022-11-12",
        "2022-12-17",
        "2022-12-29",
        "2023-01-31",
        "2023-02-15",
        "2023-02-21",
        "2023-03-03",
        "2023-03-22",
        "2023-04-20",
    ]
    expected_scf_dates = [
        "2022-10-05",
        "2022-10-28",
        "2022-11-27",
        "2023-01-06",
        "2023-03-07",
        "2023-04-06",
        "2023-04-26",
        "2023-05-26",
        "2023-06-02",
    ]
    snowflakes_dates = [
        "2022-10-05",
        "2022-10-28",
        "2022-11-12",
        "2022-11-27",
        "2022-12-17",
        "2022-12-29",
        "2023-01-06",
        "2023-02-15",
        "2023-03-07",
        "2023-03-22",
        "2023-04-03",
        "2023-04-06",
        "2023-04-26",
        "2023-05-26",
        "2023-06-02",
    ]

    station_dates = [event["date"] for event in events if event["variable"] == "station_hs"]
    scf_dates = [event["date"] for event in events if event["variable"] == "scf"]

    assert len(events) == 18
    assert station_dates == expected_station_dates
    assert scf_dates == expected_scf_dates
    assert set(station_dates).isdisjoint(scf_dates)
    assert len(set(station_dates) & set(snowflakes_dates)) == 5
    assert [event["date"] for event in events] == sorted(event["date"] for event in events)

    snowcover_dir = SUBDOMAIN_ROOT / "obs" / "snowcover"
    assert sorted(path.name for path in snowcover_dir.glob("SnowFLAKES_*_subdomain_example.nc")) == [
        f"SnowFLAKES_{date.replace('-', '')}_v3_eurac_subdomain_example.nc"
        for date in snowflakes_dates
    ]

    quality = snowcover_dir / "selected_scf_quality.csv"
    rows = quality.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + len(snowflakes_dates) * 8
    assert {line.split(",", 1)[0] for line in rows[1:]} == set(snowflakes_dates)


def test_shipped_subdomain_example_configs_are_generic_and_minimal() -> None:
    setup_cfg = _read_yaml(SUBDOMAIN_ROOT / "subdomains.yml")
    maps_cfg = _read_yaml(SUBDOMAIN_ROOT / "projects" / "project_2022_2023" / "maps.yml")
    plots_cfg = _read_yaml(SUBDOMAIN_ROOT / "projects" / "project_2022_2023" / "plots.yml")

    assert setup_cfg["domain"] == "subdomain_example"
    assert list(maps_cfg["maps"]) == ["subdomain_example_setup_overview"]
    assert maps_cfg["maps"]["subdomain_example_setup_overview"]["output_name"] == "setup_overview"
    assert [panel["panel"] for panel in plots_cfg["panels"]] == [
        "fSC",
        "roi-sd",
        "ess",
        "scores-crpss",
    ]


def test_shipped_subdomain_example_uses_buffered_generic_grid_stems() -> None:
    grid_dir = SUBDOMAIN_ROOT / "grids"
    dem_100 = grid_dir / "dem_subdomain_example_100.asc"

    assert dem_100.is_file()
    assert not (grid_dir / "dem_north_tyrol_100.asc").exists()

    header = dem_100.read_text(encoding="utf-8").splitlines()[:2]
    assert header == ["ncols        1146", "nrows        721"]

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ROFENTAL_PROJECT_YAML = (
    REPO_ROOT
    / "examples"
    / "rofental"
    / "projects"
    / "project_2022_2023"
    / "project_2022_2023.yml"
)


def _load_rofental_project_config() -> dict:
    with ROFENTAL_PROJECT_YAML.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_shipped_rofental_uses_promoted_golden_project_config() -> None:
    cfg = _load_rofental_project_config()

    assert cfg["obs"]["snowcover"]["summary_csv"] == (
        "obs/summaries/project_2022_2023/scf_summary.csv"
    )
    assert cfg["obs"]["wetsnow"]["summary_csv"] == (
        "obs/summaries/project_2022_2023/wet_snow_summary.csv"
    )

    da_cfg = cfg["data_assimilation"]
    assert da_cfg["prior_forcing"]["random_seed"] == 113
    assert da_cfg["resampling"]["seed"] == 113
    assert da_cfg["rejuvenation"]["seed"] == 113

    expected_sigmas = {
        "sigma_t": 0.5,
        "sigma_p": 0.5,
        "sigma_rh": 0.5,
        "sigma_sw": 0.05,
    }
    assert {
        key: da_cfg["prior_forcing"][key]
        for key in expected_sigmas
    } == expected_sigmas
    assert {
        key: da_cfg["rejuvenation"][key]
        for key in expected_sigmas
    } == expected_sigmas

    assert da_cfg["assimilation_events"] == [
        {"date": "2022-11-17", "variable": "station_hs"},
        {"date": "2022-12-07", "variable": "station_hs"},
        {"date": "2023-01-01", "variable": "station_hs"},
        {"date": "2023-01-31", "variable": "station_hs"},
        {"date": "2023-02-21", "variable": "station_hs"},
        {
            "date": "2023-03-24",
            "variable": "wet_snow_line",
            "product": "WETSNOW",
        },
        {"date": "2023-04-16", "variable": "scf", "product": "SNOWCOVER"},
        {"date": "2023-04-26", "variable": "scf", "product": "SNOWCOVER"},
        {"date": "2023-05-18", "variable": "scf", "product": "SNOWCOVER"},
        {"date": "2023-05-26", "variable": "scf", "product": "SNOWCOVER"},
    ]

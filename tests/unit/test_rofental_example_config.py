import csv
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
    assert da_cfg["prior_forcing"]["random_seed"] == 1415935400
    assert da_cfg["resampling"]["seed"] == 1415935400
    assert da_cfg["rejuvenation"]["seed"] == 1415935400

    expected_sigmas = {
        "sigma_t": 1.0,
        "mu_p": -0.055,
        "sigma_p": 0.6,
        "sigma_rh": 1.2,
        "sigma_sw": 0.15,
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
        {"date": "2023-04-26", "variable": "scf", "product": "SNOWCOVER"},
        {
            "date": "2023-05-03",
            "observation_time": "2023-05-03T05:26:24Z",
            "variable": "wet_snow_line",
            "product": "WETSNOW",
        },
        {"date": "2023-05-26", "variable": "scf", "product": "SNOWCOVER"},
    ]
    assert da_cfg["likelihood"]["wet_snow_line"]["obs_sigma"] == 200.0
    assert da_cfg["likelihood"]["wet_snow_line"]["min_model_finite_fraction"] == 0.95
    assert da_cfg["uncertainty"]["scf"]["assimilation"] == {
        "sigma_mode": "uncertainty_layer",
        "aggregate_metric": "unc_mean",
    }


def test_shipped_rofental_snowcover_files_use_neutral_prefix() -> None:
    rofental_root = REPO_ROOT / "examples" / "rofental"
    snowcover_dir = rofental_root / "obs" / "snowcover"
    renamed_files = tuple(snowcover_dir.glob("s2_fsc_rofental_*"))
    legacy_prefix = "s2_fsc_" + "snowflake_rofental_*"

    assert len(renamed_files) == 153
    assert not tuple(snowcover_dir.glob(legacy_prefix))

    summary_path = (
        rofental_root
        / "obs"
        / "summaries"
        / "project_2022_2023"
        / "scf_summary.csv"
    )
    with summary_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows
    assert all(row["source"].startswith("s2_fsc_rofental_") for row in rows)
    assert all("snowflake" not in row["source"].lower() for row in rows)


def test_shipped_rofental_wet_snow_summary_uses_uppermost_crossing() -> None:
    summary_path = (
        REPO_ROOT
        / "examples"
        / "rofental"
        / "obs"
        / "summaries"
        / "project_2022_2023"
        / "wet_snow_summary.csv"
    )
    with summary_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows
    assert {row["wet_snow_line_method"] for row in rows} == {
        "uppermost_crossing_fraction"
    }
    may_3 = next(row for row in rows if row["date"] == "2023-05-03")
    assert may_3["acquisition_time"] == "2023-05-03T05:26:24Z"
    assert float(may_3["wet_snow_line"]) == 3066.718769538577

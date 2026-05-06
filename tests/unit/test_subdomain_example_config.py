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

    expected_sigmas = {
        "sigma_t": 0.5,
        "sigma_p": 0.5,
        "sigma_rh": 0.5,
        "sigma_sw": 0.05,
    }
    assert {key: da_cfg["prior_forcing"][key] for key in expected_sigmas} == expected_sigmas
    assert {key: da_cfg["rejuvenation"][key] for key in expected_sigmas} == expected_sigmas


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

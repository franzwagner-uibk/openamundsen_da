from __future__ import annotations

from pathlib import Path

import pytest

from openamundsen_da.configuration import load_project_configuration
from openamundsen_da.exceptions import ProjectValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_valid_project(tmp_path: Path) -> Path:
    setup_dir = tmp_path / "alpine"
    project_dir = setup_dir / "projects" / "winter"
    project_dir.mkdir(parents=True)
    (setup_dir / "alpine.yml").write_text(
        """
input_data:
  grids:
    dir: grids
  meteo:
    dir: meteo
output_data:
  grids:
    format: netcdf
""".lstrip(),
        encoding="utf-8",
    )
    (project_dir / "winter.yml").write_text(
        """
run_mode: single
start_date: '2022-10-01'
end_date: '2023-06-30'
obs:
  stations:
    dir: obs/stations
  snowcover:
    dir: obs/snowcover
    format: geotiff
    product_tag: SNOWCOVER
    summary_csv: obs/summaries/winter/scf_summary.csv
    classes:
      valid: [0, 1]
      cloud: [2]
      water: [3]
      nodata: [255]
data_assimilation:
  prior_forcing:
    ensemble_size: 3
    random_seed: 11
    sigma_t: 0.5
    mu_p: 0.0
    sigma_p: 0.2
    sigma_rh: 0.3
    sigma_sw: 0.05
  resampling:
    algorithm: systematic
    ess_threshold_ratio: 0.7
    seed: 12
  rejuvenation:
    seed: 13
  likelihood:
    scf:
      obs_sigma: 0.1
      use_binomial: false
      sigma_floor: 0.05
      sigma_cloud_scale: 0.1
      min_sigma: 0.03
      min_support_coverage_ratio: 0.0
  restart:
    dump_state: true
    state_pattern: model_state.pickle.gz
  uncertainty:
    scf:
      enabled: true
      input_dir: obs/snowcover
      ingest: {scf_variable: fsc}
      assimilation: {sigma_mode: formula}
  output:
    grids:
      format: netcdf
  assimilation_events:
    - date: '2023-04-26'
      variable: scf
      product: SNOWCOVER
""".lstrip(),
        encoding="utf-8",
    )
    return project_dir


def test_load_project_configuration_returns_canonical_absolute_paths(tmp_path: Path) -> None:
    project_dir = _write_valid_project(tmp_path)

    config = load_project_configuration(project_dir)

    assert config.project_dir == project_dir.resolve()
    assert config.setup_dir == project_dir.parent.parent.resolve()
    assert config.model_grid_format == "netcdf"


def test_single_domain_configuration_allows_runner_managed_meteo_dir(tmp_path: Path) -> None:
    project_dir = _write_valid_project(tmp_path)
    setup_yaml = project_dir.parent.parent / "alpine.yml"
    setup_yaml.write_text(
        setup_yaml.read_text(encoding="utf-8").replace("  meteo:\n    dir: meteo\n", "  meteo: {}\n"),
        encoding="utf-8",
    )

    config = load_project_configuration(project_dir)

    assert config.setup_dir == project_dir.parent.parent.resolve()


def test_subdomain_configuration_allows_preparation_owned_summary(tmp_path: Path) -> None:
    project_dir = _write_valid_project(tmp_path)
    project_yaml = project_dir / "winter.yml"
    project_yaml.write_text(
        project_yaml.read_text(encoding="utf-8")
        .replace("run_mode: single", "run_mode: subdomain")
        .replace("    summary_csv: obs/summaries/winter/scf_summary.csv\n", ""),
        encoding="utf-8",
    )

    config = load_project_configuration(project_dir)

    assert config.project["run_mode"] == "subdomain"


def test_shipped_rofental_project_configuration_matches_strict_schema() -> None:
    project_dir = (
        REPO_ROOT
        / "examples"
        / "rofental"
        / "projects"
        / "project_2022_2023"
    )

    config = load_project_configuration(project_dir)

    assert config.project_dir == project_dir.resolve()


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("run_mode: single", "run_mode: single\nlegacy_mode: true", "Unknown configuration key"),
        ("run_mode: single", "run_mode: legacy", "project.run_mode"),
        ("dir: obs/snowcover", "dir: ../../outside", "escapes the setup directory"),
        ("format: geotiff", "format: ascii", "project.obs.snowcover.format"),
        ("format: netcdf", "format: memory", "setup.output_data.grids.format"),
        (
            "state_pattern: model_state.pickle.gz",
            "state_pattern: model_state.pickle.gz\n    cleanup_after_setup: false",
            "cleanup_after_setup",
        ),
        ("variable: scf", "variable: wet_snow_fraction", "removed alias"),
    ],
)
def test_load_project_configuration_rejects_invalid_contract(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    project_dir = _write_valid_project(tmp_path)
    target = project_dir / "winter.yml"
    if old == "format: netcdf" and new == "format: memory":
        target = project_dir.parent.parent / "alpine.yml"
    target.write_text(target.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ProjectValidationError, match=message):
        load_project_configuration(project_dir)


def test_load_project_configuration_rejects_duplicate_event_dates(tmp_path: Path) -> None:
    project_dir = _write_valid_project(tmp_path)
    path = project_dir / "winter.yml"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "    - date: '2023-04-26'\n      variable: station_hs\n      product: STATION\n",
        encoding="utf-8",
    )

    with pytest.raises(ProjectValidationError, match="Duplicate assimilation event date"):
        load_project_configuration(project_dir)


def test_subdomain_event_filter_requires_external_final_selection(tmp_path: Path) -> None:
    project_dir = _write_valid_project(tmp_path)
    path = project_dir / "winter.yml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "data_assimilation:\n",
            "data_assimilation:\n  subdomain_event_filter:\n    enabled: true\n",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProjectValidationError, match="Finalize every project or leaf schedule"):
        load_project_configuration(project_dir)

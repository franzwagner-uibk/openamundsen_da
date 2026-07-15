from __future__ import annotations

from pathlib import Path

import pytest

from openamundsen_da.configuration import load_project_configuration
from openamundsen_da.exceptions import ProjectValidationError


def _write_valid_project(tmp_path: Path) -> Path:
    setup_dir = tmp_path / "alpine"
    project_dir = setup_dir / "projects" / "winter"
    project_dir.mkdir(parents=True)
    (setup_dir / "alpine.yml").write_text(
        """
input_data:
  grids:
    dir: grids
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


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("run_mode: single", "run_mode: single\nlegacy_mode: true", "Unknown configuration key"),
        ("dir: obs/snowcover", "dir: ../../outside", "escapes the setup directory"),
        ("format: geotiff", "format: ascii", "project.obs.snowcover.format"),
        ("format: netcdf", "format: memory", "setup.output_data.grids.format"),
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

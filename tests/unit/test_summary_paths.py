from __future__ import annotations

import textwrap
from pathlib import Path

import ruamel.yaml

from openamundsen_da.observer.summary_paths import record_fraction_summary_path, resolve_fraction_summary_path


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def test_resolve_fraction_summary_path_uses_project_config(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023_ens23"
    _write_yaml(
        project_dir / "project_2022_2023_ens23.yml",
        """
        obs:
          snowcover:
            summary_csv: obs/summaries/project_2022_2023/scf_summary.csv
          wetsnow:
            summary_csv: obs/summaries/project_2022_2023/wet_snow_summary.csv
        data_assimilation:
          assimilation_events: []
        """,
    )

    assert resolve_fraction_summary_path(setup_dir, project_dir, "scf_summary.csv") == (
        setup_dir / "obs" / "summaries" / "project_2022_2023" / "scf_summary.csv"
    )
    assert resolve_fraction_summary_path(setup_dir, project_dir, "wet_snow_summary.csv") == (
        setup_dir / "obs" / "summaries" / "project_2022_2023" / "wet_snow_summary.csv"
    )
    assert resolve_fraction_summary_path(setup_dir, project_dir, "wet_snow_line_diagnostics.csv") == (
        setup_dir / "obs" / "summaries" / "project_2022_2023" / "wet_snow_line_diagnostics.csv"
    )


def test_record_fraction_summary_path_writes_setup_relative_project_config(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023_ens23"
    _write_yaml(
        project_dir / "project_2022_2023_ens23.yml",
        """
        obs:
          wetsnow:
            product_tag: WETSNOW
        data_assimilation:
          assimilation_events: []
        """,
    )
    summary_csv = setup_dir / "obs" / "summaries" / "project_2022_2023" / "wet_snow_summary.csv"

    record_fraction_summary_path(
        setup_dir=setup_dir,
        project_dir=project_dir,
        filename="wet_snow_summary.csv",
        summary_csv=summary_csv,
    )

    yaml = ruamel.yaml.YAML(typ="safe")
    with (project_dir / "project_2022_2023_ens23.yml").open("r", encoding="utf-8") as f:
        data = yaml.load(f)
    assert data["obs"]["wetsnow"]["summary_csv"] == "obs/summaries/project_2022_2023/wet_snow_summary.csv"

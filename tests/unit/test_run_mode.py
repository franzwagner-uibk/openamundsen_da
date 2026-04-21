from __future__ import annotations

from pathlib import Path

import pytest

from openamundsen_da.util.run_mode import ensure_run_mode, read_run_mode


def _write_project_yaml(project_dir: Path, text: str) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / f"{project_dir.name}.yml").write_text(text, encoding="utf-8")


def test_ensure_run_mode_writes_missing_marker(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(
        project_dir,
        "start_date: '2022-10-01'\nend_date: '2022-10-02'\ndata_assimilation: {}\n",
    )

    mode = ensure_run_mode(project_dir, expected="single", write_if_missing=True)

    assert mode == "single"
    assert read_run_mode(project_dir) == "single"
    rendered = (project_dir / f"{project_dir.name}.yml").read_text(encoding="utf-8")
    assert "run_mode: single" in rendered
    assert "data_assimilation:\n  run_mode:" not in rendered


def test_ensure_run_mode_rejects_mismatch(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(
        project_dir,
        "run_mode: subdomain\nstart_date: '2022-10-01'\nend_date: '2022-10-02'\ndata_assimilation: {}\n",
    )

    with pytest.raises(ValueError, match="run_mode='subdomain'"):
        ensure_run_mode(project_dir, expected="single", write_if_missing=False)


def test_read_run_mode_ignores_removed_nested_legacy_marker(tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(
        project_dir,
        "start_date: '2022-10-01'\nend_date: '2022-10-02'\ndata_assimilation:\n  run_mode: subdomain\n",
    )

    assert read_run_mode(project_dir) is None

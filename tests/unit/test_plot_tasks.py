from __future__ import annotations

from pathlib import Path

from openamundsen_da.pipeline import plot_tasks


def _write_project_yaml(project_dir: Path, events: list[tuple[str, str]]) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "start_date: '2023-01-01'",
        "end_date: '2023-01-31'",
        "data_assimilation:",
        "  assimilation_events:",
    ]
    for date, variable in events:
        lines.extend(
            [
                f"    - date: '{date}'",
                f"      variable: {variable}",
                "      product: TEST",
            ]
        )
    (project_dir / f"{project_dir.name}.yml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_aggregate_fraction_envelopes_only_uses_configured_event_variables(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(
        project_dir,
        [
            ("2023-01-10", "scf"),
            ("2023-01-15", "station_hs"),
        ],
    )

    calls: list[tuple[str, str, Path]] = []

    def _fake_aggregate_fraction(*, project_dir: Path, filename: str, value_col: str, output_path: Path):
        calls.append((filename, value_col, output_path))
        return output_path

    monkeypatch.setattr(plot_tasks, "_aggregate_fraction", _fake_aggregate_fraction)

    plot_tasks.aggregate_fraction_envelopes(
        project_dir=project_dir,
        project_fraction_envelope_path=lambda project_dir, variable: project_dir / "results" / "misc" / f"{variable}.csv",
    )

    assert calls == [("point_scf_roi.csv", "scf", project_dir / "results" / "misc" / "scf.csv")]


def test_aggregate_fraction_envelopes_uses_wet_fraction_support_for_wet_snow_line(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir, [("2023-01-10", "wet_snow_line")])

    calls: list[tuple[str, str, Path]] = []

    def _fake_aggregate_fraction(*, project_dir: Path, filename: str, value_col: str, output_path: Path):
        calls.append((filename, value_col, output_path))
        return output_path

    monkeypatch.setattr(plot_tasks, "_aggregate_fraction", _fake_aggregate_fraction)

    plot_tasks.aggregate_fraction_envelopes(
        project_dir=project_dir,
        project_fraction_envelope_path=lambda project_dir, variable: project_dir / "results" / "misc" / f"{variable}.csv",
    )

    assert calls == [
        ("point_wet_snow_roi.csv", "wet_snow_fraction", project_dir / "results" / "misc" / "wet_snow.csv"),
        ("point_wet_snow_line_roi.csv", "wet_snow_line", project_dir / "results" / "misc" / "wet_snow_line.csv"),
    ]

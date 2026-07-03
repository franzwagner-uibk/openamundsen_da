from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openamundsen_da.util.fraction_envelope import aggregate_fraction_envelope


def _write_member_series(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time": ["2023-01-01", "2023-01-02"],
            "scf": values,
        }
    ).to_csv(path, index=False)


def test_aggregate_fraction_envelope_uses_member_quantiles_and_excludes_open_loop(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    step_dir = project_dir / "steps" / "step_00"
    _write_member_series(step_dir / "ensembles" / "prior" / "open_loop" / "results" / "point_scf_roi.csv", [0.0, 0.0])
    _write_member_series(step_dir / "ensembles" / "prior" / "member_001" / "results" / "point_scf_roi.csv", [0.2, 0.4])
    _write_member_series(step_dir / "ensembles" / "prior" / "member_002" / "results" / "point_scf_roi.csv", [0.8, 1.0])

    out_path = aggregate_fraction_envelope(
        setup_dir=project_dir,
        filename="point_scf_roi.csv",
        value_col="scf",
        output_name="results/misc/point_scf_roi_envelope.csv",
    )

    assert out_path is not None
    out = pd.read_csv(out_path)
    assert list(out["n"]) == [2, 2]
    assert list(out["value_mean"]) == pytest.approx([0.5, 0.7])
    assert list(out["value_min"]) == pytest.approx([0.23, 0.43])
    assert list(out["value_max"]) == pytest.approx([0.77, 0.97])

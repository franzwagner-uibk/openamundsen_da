from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from openamundsen_da.util.fraction_envelope import aggregate_fraction_envelope
from openamundsen_da.methods.pf.weights import write_prior_weights


def _write_member_series(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time": ["2023-01-01", "2023-01-02"],
            "scf": values,
        }
    ).to_csv(path, index=False)


def test_aggregate_fraction_envelope_uses_member_minmax_and_excludes_open_loop(tmp_path: Path) -> None:
    project_dir = tmp_path / "project"
    step_dir = project_dir / "steps" / "step_00"
    _write_member_series(step_dir / "ensembles" / "prior" / "open_loop" / "results" / "point_scf_roi.csv", [0.0, 0.0])
    _write_member_series(step_dir / "ensembles" / "prior" / "member_001" / "results" / "point_scf_roi.csv", [0.2, 0.4])
    _write_member_series(step_dir / "ensembles" / "prior" / "member_002" / "results" / "point_scf_roi.csv", [0.8, 1.0])
    write_prior_weights(
        step_dir,
        member_ids=["member_001", "member_002"],
        weights=[0.25, 0.75],
        mode="carried_posterior",
    )

    out_path = aggregate_fraction_envelope(
        setup_dir=project_dir,
        filename="point_scf_roi.csv",
        value_col="scf",
        output_name="results/misc/point_scf_roi_envelope.csv",
    )

    assert out_path is not None
    out = pd.read_csv(out_path)
    assert list(out["n"]) == [2, 2]
    assert list(out["value_mean"]) == pytest.approx([0.65, 0.85])
    assert list(out["value_std"]) == pytest.approx([0.259807621, 0.259807621])
    assert list(out["ess"]) == pytest.approx([1.6, 1.6])
    assert list(out["value_min"]) == pytest.approx([0.2, 0.4])
    assert list(out["value_max"]) == pytest.approx([0.8, 1.0])
    assert set(out["weighting"]) == {"pf_prior_ledger"}
    assert set(out["bounds_semantics"]) == {"materialized_member_range"}

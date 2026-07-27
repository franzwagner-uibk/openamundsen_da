from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openamundsen_da.core.constants import MEMBER_SOURCE_POINTER
from openamundsen_da.methods.pf.resample import resample_from_weights
from openamundsen_da.methods.pf.weights import carry_weights_to_next_step, load_prior_weights


def _source_members(step_dir: Path, count: int) -> None:
    for idx in range(1, count + 1):
        (step_dir / "ensembles" / "prior" / f"member_{idx:03d}" / "results").mkdir(
            parents=True,
            exist_ok=True,
        )


def _run(step_dir: Path, weights: list[tuple[str, float]], *, threshold_ratio: float) -> dict:
    _source_members(step_dir, len(weights))
    weights_csv = step_dir / "assim" / "weights_scf_20230101.csv"
    weights_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(weights, columns=["member_id", "weight"]).to_csv(weights_csv, index=False)
    return resample_from_weights(
        step_dir=step_dir,
        source_ensemble="prior",
        weights_csv=weights_csv,
        target_ensemble="posterior",
        seed=17,
        algorithm="systematic",
        ess_threshold=0.0,
        ess_threshold_ratio=threshold_ratio,
        overwrite=False,
    )


def test_skipped_resampling_preserves_ancestry_and_validates_resume(tmp_path: Path) -> None:
    step = tmp_path / "step_00"
    result = _run(step, [("member_001", 0.5), ("member_002", 0.5)], threshold_ratio=0.7)

    assert result["resampled"] is False
    mapping = pd.read_csv(result["indices_csv"])
    assert list(mapping["source_member_id"]) == ["member_001", "member_002"]
    for row in mapping.itertuples(index=False):
        pointer = json.loads(
            (step / "ensembles" / "posterior" / row.posterior_member_id / MEMBER_SOURCE_POINTER).read_text(
                encoding="utf-8"
            )
        )
        assert Path(pointer["member_dir"]).name == row.source_member_id

    resumed = _run(step, [("member_002", 0.5), ("member_001", 0.5)], threshold_ratio=0.7)
    assert resumed["indices_csv"] == result["indices_csv"]


def test_systematic_resampling_records_ancestry_and_resets_child_weights(tmp_path: Path) -> None:
    step = tmp_path / "step_00"
    result = _run(step, [("member_001", 0.99), ("member_002", 0.01)], threshold_ratio=0.9)

    assert result["resampled"] is True
    mapping = pd.read_csv(result["indices_csv"])
    event_weights = pd.DataFrame(
        {"member_id": ["member_001", "member_002"], "weight": [0.99, 0.01]}
    )
    next_step = tmp_path / "step_01"
    carry_weights_to_next_step(
        current_step_dir=step,
        next_step_dir=next_step,
        event_weights=event_weights,
        mapping=mapping,
        resampled=True,
        source_weights=Path(step / "assim" / "weights_scf_20230101.csv"),
    )
    ledger = load_prior_weights(next_step, mapping["posterior_member_id"].astype(str))
    np.testing.assert_allclose(ledger["weight"], [0.5, 0.5])


def test_systematic_mapping_is_invariant_to_weight_row_order(tmp_path: Path) -> None:
    weights = [("member_001", 0.6), ("member_002", 0.3), ("member_003", 0.1)]
    first = _run(tmp_path / "first" / "step_00", weights, threshold_ratio=0.99)
    second = _run(tmp_path / "second" / "step_00", list(reversed(weights)), threshold_ratio=0.99)

    first_mapping = pd.read_csv(first["indices_csv"])
    second_mapping = pd.read_csv(second["indices_csv"])
    assert list(first_mapping["source_member_id"]) == list(second_mapping["source_member_id"])


def test_resume_rejects_corrupt_materialized_ancestry(tmp_path: Path) -> None:
    step = tmp_path / "step_00"
    _run(step, [("member_001", 0.5), ("member_002", 0.5)], threshold_ratio=0.7)
    pointer = step / "ensembles" / "posterior" / "member_001" / MEMBER_SOURCE_POINTER
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["member_dir"] = str(step / "ensembles" / "prior" / "member_002")
    pointer.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="ancestry mismatch"):
        _run(step, [("member_001", 0.5), ("member_002", 0.5)], threshold_ratio=0.7)

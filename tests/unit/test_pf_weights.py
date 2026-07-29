from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from openamundsen_da.methods.pf.weights import (
    carry_weights_to_next_step,
    combine_event_weights,
    initialize_prior_weights,
    load_event_weights,
    load_prior_weights,
    write_event_weights,
)


MEMBERS = ["member_001", "member_002", "member_003"]


def _event(likelihoods: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "member_id": MEMBERS,
            "log_likelihood": np.log(np.asarray(likelihoods, dtype=float)),
        }
    )


def test_two_skipped_events_multiply_persistent_weights(tmp_path) -> None:
    step_0 = tmp_path / "step_00"
    step_1 = tmp_path / "step_01"
    initialize_prior_weights(step_0, MEMBERS)

    first = combine_event_weights(_event([0.6, 0.3, 0.1]), step_dir=step_0)
    mapping = pd.DataFrame(
        {"posterior_member_id": MEMBERS, "source_member_id": MEMBERS}
    )
    carry_weights_to_next_step(
        current_step_dir=step_0,
        next_step_dir=step_1,
        event_weights=first,
        mapping=mapping,
        resampled=False,
        source_weights=step_0 / "assim" / "weights_scf_20230101.csv",
    )
    second = combine_event_weights(_event([0.2, 0.5, 0.3]), step_dir=step_1)

    expected = np.asarray([0.6, 0.3, 0.1]) * np.asarray([0.2, 0.5, 0.3])
    expected /= expected.sum()
    assert second["weight"].to_numpy() == pytest.approx(expected)
    assert second["prior_weight"].to_numpy() == pytest.approx([0.6, 0.3, 0.1])


def test_actual_resampling_resets_child_weights_to_uniform(tmp_path) -> None:
    step_0 = tmp_path / "step_00"
    step_1 = tmp_path / "step_01"
    initialize_prior_weights(step_0, MEMBERS)
    event = combine_event_weights(_event([0.8, 0.15, 0.05]), step_dir=step_0)
    mapping = pd.DataFrame(
        {
            "posterior_member_id": MEMBERS,
            "source_member_id": ["member_001", "member_001", "member_002"],
        }
    )

    carry_weights_to_next_step(
        current_step_dir=step_0,
        next_step_dir=step_1,
        event_weights=event,
        mapping=mapping,
        resampled=True,
        source_weights=step_0 / "assim" / "weights_scf_20230101.csv",
    )

    ledger = load_prior_weights(step_1, MEMBERS)
    assert ledger["weight"].to_numpy() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_event_requires_existing_ledger_unless_explicitly_initialized(tmp_path) -> None:
    step = tmp_path / "step_00"
    with pytest.raises(FileNotFoundError, match="initialize a new chain explicitly"):
        combine_event_weights(_event([0.5, 0.3, 0.2]), step_dir=step)

    result = combine_event_weights(
        _event([0.5, 0.3, 0.2]),
        step_dir=step,
        initialize=True,
    )
    assert result["weight"].to_numpy() == pytest.approx([0.5, 0.3, 0.2])


def test_neutral_likelihood_preserves_prior_weights(tmp_path) -> None:
    step = tmp_path / "step_00"
    initialize_prior_weights(step, MEMBERS)
    first = combine_event_weights(_event([0.7, 0.2, 0.1]), step_dir=step)
    from openamundsen_da.methods.pf.weights import write_prior_weights

    write_prior_weights(
        step,
        member_ids=MEMBERS,
        weights=first["weight"],
        mode="carried_posterior",
        overwrite=True,
    )
    neutral = pd.DataFrame({"member_id": MEMBERS, "log_likelihood": [0.0, 0.0, 0.0]})
    result = combine_event_weights(neutral, step_dir=step)
    assert result["weight"].to_numpy() == pytest.approx(first["weight"].to_numpy())


@pytest.mark.parametrize("bad", [[np.nan, 0.0, 0.0], [-np.inf, -np.inf, -np.inf]])
def test_event_rejects_invalid_log_likelihoods(tmp_path, bad) -> None:
    step = tmp_path / "step_00"
    initialize_prior_weights(step, MEMBERS)
    frame = pd.DataFrame({"member_id": MEMBERS, "log_likelihood": bad})
    with pytest.raises(ValueError, match="must be finite"):
        combine_event_weights(frame, step_dir=step)


def test_event_weights_resume_manifest_binds_config_inputs_and_ledger(tmp_path: Path) -> None:
    project = tmp_path / "project"
    step = project / "steps" / "step_00"
    project.mkdir()
    (project / "project.yml").write_text("data_assimilation: {}\n", encoding="utf-8")
    step.mkdir(parents=True)
    (step / "step.yml").write_text("start_date: 2023-01-01\n", encoding="utf-8")
    obs = step / "obs" / "obs_scf_TEST_20230101.csv"
    obs.parent.mkdir()
    obs.write_text("scf,n_valid\n0.5,10\n", encoding="utf-8")
    initialize_prior_weights(step, MEMBERS)
    weights = combine_event_weights(_event([0.5, 0.3, 0.2]), step_dir=step)
    weights["resampling_threshold"] = 2.1
    weights["resampled"] = False
    weights_csv = step / "assim" / "weights_scf_20230101.csv"
    write_event_weights(
        weights_csv,
        weights,
        project_dir=project,
        step_dir=step,
    )

    resumed = load_event_weights(weights_csv, project_dir=project, step_dir=step)
    assert list(resumed["member_id"]) == MEMBERS

    obs.write_text("scf,n_valid\n0.6,10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inputs changed"):
        load_event_weights(weights_csv, project_dir=project, step_dir=step)

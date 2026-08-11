from __future__ import annotations

from pathlib import Path

import pytest

from openamundsen_da.core import runner
from openamundsen_da.util.restart_state import validate_restart_state


class _StateCategory:
    _meta = {"swe": object()}

    def __getitem__(self, name: str) -> int:
        assert name == "swe"
        return 42


class _State:
    categories = ("snow",)

    def __getitem__(self, name: str) -> _StateCategory:
        assert name == "snow"
        return _StateCategory()


class _Model:
    state = _State()


def test_dump_state_is_atomic_and_validated(tmp_path: Path) -> None:
    output = tmp_path / "results" / "model_state.pickle.gz"
    runner._dump_init_data(_Model(), output)

    assert validate_restart_state(output) == output
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_step_successor_detection_is_strict(tmp_path: Path) -> None:
    project = tmp_path / "project"
    first = project / "steps" / "step_00"
    final = project / "steps" / "step_01"
    for step in (first, final):
        step.mkdir(parents=True)
        (step / f"{step.name}.yml").write_text("start_date: 2020-01-01\n", encoding="utf-8")

    assert runner._step_has_successor(project, first)
    assert not runner._step_has_successor(project, final)
    with pytest.raises(RuntimeError, match="not part of the prepared project"):
        runner._step_has_successor(project, project / "steps" / "missing")


def test_nonfinal_state_dump_failure_is_fatal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        runner,
        "_dump_init_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(RuntimeError, match="Required successor checkpoint"):
        runner._save_state_dump(
            _Model(),
            tmp_path / "state.gz",
            required=True,
            member_name="member_001",
        )
    assert not runner._save_state_dump(
        _Model(),
        tmp_path / "state.gz",
        required=False,
        member_name="member_001",
    )

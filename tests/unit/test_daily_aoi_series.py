from __future__ import annotations

from datetime import datetime
from pathlib import Path

import openamundsen_da.methods.daily_aoi_series as daily_mod
import pytest


class _FakeFuture:
    def __init__(self, result=None, exc: Exception | None = None):
        self._result = result
        self._exc = exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeExecutor:
    def __init__(self, futures: list[_FakeFuture], **kwargs):
        self._futures = futures
        self._idx = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, worker, *args):
        fut = self._futures[self._idx]
        self._idx += 1
        return fut


def test_compute_step_daily_series_raises_with_member_context_for_worker_failure(
    monkeypatch,
    tmp_path: Path,
) -> None:
    step_dir = tmp_path / "step_00"
    open_loop_results = step_dir / "ensembles" / "prior" / "open_loop" / "results"
    member_results = step_dir / "ensembles" / "prior" / "member_001" / "results"
    open_loop_results.mkdir(parents=True)
    member_results.mkdir(parents=True)

    monkeypatch.setattr(
        daily_mod,
        "list_member_dirs",
        lambda ensembles_root, ensemble: [step_dir / "ensembles" / ensemble / "member_001"],
    )
    monkeypatch.setattr(
        daily_mod,
        "open_loop_dir",
        lambda step: step_dir / "ensembles" / "prior" / "open_loop",
    )

    futures = [
        _FakeFuture(result=True),
        _FakeFuture(exc=RuntimeError("boom")),
    ]
    monkeypatch.setattr(daily_mod.cf, "ProcessPoolExecutor", lambda max_workers: _FakeExecutor(futures))
    monkeypatch.setattr(daily_mod.cf, "as_completed", lambda futures_iterable: list(futures_iterable))

    with pytest.raises(RuntimeError, match="Daily AOI series failed for 1 / 2 member\\(s\\)") as exc_info:
        daily_mod.compute_step_daily_series_for_all_members(
            step_dir=step_dir,
            aoi_path=tmp_path / "roi.gpkg",
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 2),
            csv_name="point_scf_aoi.csv",
            worker=lambda *args: True,
            max_workers=2,
            overwrite=True,
        )
    msg = str(exc_info.value)
    assert "point_scf_aoi.csv" in msg
    assert "step_00" in msg
    assert str(member_results) in msg
    assert "boom" in msg

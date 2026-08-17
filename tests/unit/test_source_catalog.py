from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

from openamundsen_da.util.source_catalog import SourceCatalog
from openamundsen_da.util.storage_budget import estimate_step_forcing_bytes


def _write_meteo(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "stations.csv").write_text(
        "station_id,name\nstation_a,Station A\n",
        encoding="utf-8",
    )
    (directory / "station_a.csv").write_text(
        "date,temp,precip\n"
        "2020-01-01 00:00:00,1,0\n"
        "2020-01-01 01:00:00,2,0\n"
        "2020-01-01 02:00:00,3,1\n"
        "2020-01-01 03:00:00,4,0\n",
        encoding="utf-8",
    )


def test_catalog_forcing_windows_match_legacy_estimator(tmp_path: Path) -> None:
    meteo = tmp_path / "meteo"
    _write_meteo(meteo)
    start = datetime(2020, 1, 1, 1)
    end = datetime(2020, 1, 1, 2)

    legacy = estimate_step_forcing_bytes(
        meteo,
        start=start,
        end=end,
        ensemble_size=3,
    )
    catalog = SourceCatalog(trusted_root=tmp_path)
    indexed = estimate_step_forcing_bytes(
        meteo,
        start=start,
        end=end,
        ensemble_size=3,
        source_catalog=catalog,
    )

    assert indexed == legacy
    assert catalog.summary()["forcing_files_parsed"] == 1
    assert catalog.summary()["forcing_window_queries"] == 1


def test_catalog_reuses_one_forcing_inode_for_many_windows_and_aliases(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_meteo(first)
    second.mkdir()
    os.link(first / "station_a.csv", second / "station_a.csv")
    os.link(first / "stations.csv", second / "stations.csv")
    catalog = SourceCatalog(trusted_root=tmp_path)

    for hour in range(3):
        for meteo in (first, second):
            catalog.estimate_step_forcing_bytes(
                meteo,
                start=datetime(2020, 1, 1, hour),
                end=datetime(2020, 1, 1, hour + 1),
                ensemble_size=2,
            )

    summary = catalog.summary()
    assert summary["forcing_files_parsed"] == 1
    assert summary["forcing_directories"] == 2
    assert summary["forcing_window_queries"] == 6
    assert summary["unique_source_files"] == 2
    assert summary["logical_source_paths"] == 4


def test_catalog_hashes_one_inode_once(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    alias = tmp_path / "alias.bin"
    source.write_bytes(b"catalog payload")
    os.link(source, alias)
    catalog = SourceCatalog(trusted_root=tmp_path)

    assert catalog.sha256_file(source) == catalog.sha256_file(alias)
    summary = catalog.summary()
    assert summary["unique_hashed_files"] == 1
    assert summary["unique_hashed_bytes"] == source.stat().st_size


def test_catalog_rejects_source_outside_trusted_root(tmp_path: Path) -> None:
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("date,temp\n2020-01-01,1\n", encoding="utf-8")
    catalog = SourceCatalog(trusted_root=trusted)

    with pytest.raises(ValueError, match="escapes trusted root"):
        catalog.sha256_file(outside)


def test_catalog_snapshot_detects_mutation_without_rehashing(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"first")
    catalog = SourceCatalog(trusted_root=tmp_path)
    catalog.sha256_file(source)
    snapshot = catalog.snapshot()

    SourceCatalog.verify_snapshot(snapshot, trusted_root=tmp_path)
    source.write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="changed after preflight"):
        SourceCatalog.verify_snapshot(snapshot, trusted_root=tmp_path)

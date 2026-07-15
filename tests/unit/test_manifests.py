from __future__ import annotations

from pathlib import Path

import pytest

from openamundsen_da.manifests import (
    file_inventory,
    inventory_digest,
    load_manifest,
    write_manifest_atomic,
)


def test_atomic_manifest_round_trip_and_replacement(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    write_manifest_atomic(path, {"status": "running"})
    write_manifest_atomic(path, {"status": "success", "outputs": ["a.nc"]})

    assert load_manifest(path) == {
        "schema_version": 1,
        "status": "success",
        "outputs": ["a.nc"],
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_file_inventory_is_content_bound_and_root_relative(tmp_path: Path) -> None:
    first = tmp_path / "b.txt"
    second = tmp_path / "a.txt"
    first.write_text("b", encoding="utf-8")
    second.write_text("a", encoding="utf-8")

    before = file_inventory(root=tmp_path, files=[first, second])
    first.write_text("changed", encoding="utf-8")
    after = file_inventory(root=tmp_path, files=[first, second])

    assert [entry["path"] for entry in before] == ["a.txt", "b.txt"]
    assert inventory_digest(before) != inventory_digest(after)


def test_file_inventory_rejects_outside_file(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    with pytest.raises(ValueError, match="outside"):
        file_inventory(root=tmp_path, files=[outside])

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_module() -> ModuleType:
    script = Path(__file__).parents[2] / "scripts" / "release" / "materialize_manuscript_setup.py"
    spec = importlib.util.spec_from_file_location("materialize_manuscript_setup", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    base = tmp_path / "base"
    snapshot = tmp_path / "snapshot"
    base.mkdir()
    snapshot.mkdir()
    (base / "kept.txt").write_text("kept\n", encoding="utf-8")
    (base / "data.txt").write_text("new\n", encoding="utf-8")
    frozen = snapshot / "data.txt"
    frozen.write_text("manuscript\n", encoding="utf-8")
    manifest = snapshot / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": [
                    {
                        "path": "data.txt",
                        "sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return base, snapshot, manifest


def test_default_snapshot_manifest_checksums_match() -> None:
    module = _load_module()
    manifest = module._read_manifest(module.DEFAULT_MANIFEST)

    for record in manifest["files"]:
        module._snapshot_file(module.DEFAULT_SNAPSHOT, record)


def test_materialize_setup_applies_validated_snapshot(tmp_path: Path) -> None:
    module = _load_module()
    base, snapshot, manifest = _fixture(tmp_path)
    target = tmp_path / "target"

    written = module.materialize_setup(
        base_setup=base,
        snapshot_root=snapshot,
        manifest_path=manifest,
        target=target,
    )

    assert written == ((target / "data.txt").resolve(),)
    assert (target / "data.txt").read_text(encoding="utf-8") == "manuscript\n"
    assert (target / "kept.txt").read_text(encoding="utf-8") == "kept\n"


def test_materialize_setup_rejects_checksum_drift(tmp_path: Path) -> None:
    module = _load_module()
    base, snapshot, manifest = _fixture(tmp_path)
    (snapshot / "data.txt").write_text("changed\n", encoding="utf-8")

    with pytest.raises(module.ManuscriptSetupError, match="checksum differs"):
        module.materialize_setup(
            base_setup=base,
            snapshot_root=snapshot,
            manifest_path=manifest,
            target=tmp_path / "target",
        )


def test_materialize_setup_requires_explicit_overwrite(tmp_path: Path) -> None:
    module = _load_module()
    base, snapshot, manifest = _fixture(tmp_path)
    target = tmp_path / "target"
    target.mkdir()

    with pytest.raises(module.ManuscriptSetupError, match="--overwrite"):
        module.materialize_setup(
            base_setup=base,
            snapshot_root=snapshot,
            manifest_path=manifest,
            target=target,
        )


@pytest.mark.parametrize("target_kind", ["inside", "ancestor"])
def test_materialize_setup_rejects_overlapping_source_tree(
    tmp_path: Path, target_kind: str
) -> None:
    module = _load_module()
    protected_parent = tmp_path / "protected"
    base = protected_parent / "base"
    snapshot = tmp_path / "snapshot"
    base.mkdir(parents=True)
    snapshot.mkdir()
    manifest = snapshot / "manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "files": [{"path": "data.txt", "sha256": ""}]}),
        encoding="utf-8",
    )
    target = base / "nested" if target_kind == "inside" else protected_parent

    with pytest.raises(module.ManuscriptSetupError, match="outside protected source tree"):
        module.materialize_setup(
            base_setup=base,
            snapshot_root=snapshot,
            manifest_path=manifest,
            target=target,
            overwrite=True,
        )

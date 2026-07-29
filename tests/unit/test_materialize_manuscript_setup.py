from __future__ import annotations

import importlib.util
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


def _fixture(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    (base / "kept.txt").write_text("kept\n", encoding="utf-8")
    (base / "data.txt").write_text("shipped\n", encoding="utf-8")
    return base


def test_default_source_is_shipped_rofental_example() -> None:
    module = _load_module()

    assert module.DEFAULT_BASE.is_dir()
    assert (
        module.DEFAULT_BASE
        / "projects"
        / "project_2022_2023"
        / "project_2022_2023.yml"
    ).is_file()


def test_materialize_setup_copies_shipped_setup_without_overlay(tmp_path: Path) -> None:
    module = _load_module()
    base = _fixture(tmp_path)
    target = tmp_path / "target"

    written = module.materialize_setup(
        base_setup=base,
        target=target,
    )

    assert written == target.resolve()
    assert (target / "data.txt").read_text(encoding="utf-8") == "shipped\n"
    assert (target / "kept.txt").read_text(encoding="utf-8") == "kept\n"


def test_materialize_setup_requires_explicit_overwrite(tmp_path: Path) -> None:
    module = _load_module()
    base = _fixture(tmp_path)
    target = tmp_path / "target"
    target.mkdir()

    with pytest.raises(module.ManuscriptSetupError, match="--overwrite"):
        module.materialize_setup(
            base_setup=base,
            target=target,
        )


@pytest.mark.parametrize("target_kind", ["inside", "ancestor"])
def test_materialize_setup_rejects_overlapping_source_tree(
    tmp_path: Path, target_kind: str
) -> None:
    module = _load_module()
    protected_parent = tmp_path / "protected"
    base = protected_parent / "base"
    base.mkdir(parents=True)
    target = base / "nested" if target_kind == "inside" else protected_parent

    with pytest.raises(module.ManuscriptSetupError, match="outside the shipped setup tree"):
        module.materialize_setup(
            base_setup=base,
            target=target,
            overwrite=True,
        )

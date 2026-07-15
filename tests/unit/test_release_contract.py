from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "release" / "validate_release.py"
SPEC = importlib.util.spec_from_file_location("validate_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validate_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate_release)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v0.9.0rc1", ("0.9.0rc1", True)),
        ("v0.9.0rc12", ("0.9.0rc12", True)),
        ("v0.9.0", ("0.9.0", False)),
    ],
)
def test_release_from_tag(tag: str, expected: tuple[str, bool]) -> None:
    assert validate_release.release_from_tag(tag) == expected


@pytest.mark.parametrize(
    "tag",
    ["0.9.0", "v0.9", "v0.9.0rc0", "v0.9.0-beta1", "v1.0.0.dev1"],
)
def test_release_from_tag_rejects_unsupported_forms(tag: str) -> None:
    with pytest.raises(ValueError, match="Unsupported release tag"):
        validate_release.release_from_tag(tag)

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "release" / "validate_release.py"
ROOT = SCRIPT.parents[2]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
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


def test_fiona_trivy_exception_requires_non_vulnerable_gdal_pin() -> None:
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    ignore = (ROOT / ".trivyignore").read_text(encoding="utf-8")

    assert re.search(r"^\s*-\s+fiona=1\.9\.6\s*$", environment, re.MULTILINE)
    assert re.search(r"^\s*-\s+gdal=3\.8\.5\s*$", environment, re.MULTILINE)
    assert "GHSA-q5fm-55c2-v6j9 exp:2027-07-15" in ignore


def test_release_workflow_prepares_metadata_directory_before_sbom() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    prepare_step = "      - name: Prepare release metadata directory"
    sbom_step = "      - name: Generate SPDX package SBOM"

    prepare_index = workflow.index(prepare_step)
    sbom_index = workflow.index(sbom_step)

    assert prepare_index < sbom_index
    assert "run: mkdir -p release-metadata" in workflow[prepare_index:sbom_index]

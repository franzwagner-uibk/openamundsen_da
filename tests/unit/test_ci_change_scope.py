from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "classify_changes.py"
SPEC = importlib.util.spec_from_file_location("classify_changes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
classify_changes = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classify_changes
SPEC.loader.exec_module(classify_changes)


@pytest.mark.parametrize(
    "path",
    [
        "docs/Tutorial/07-results-and-diagnostics.md",
        "docs/assets/images/tutorial/reference.png",
        "tests/baselines/rofental_es30_tutorial_assets.json",
        "CHANGELOG.md",
    ],
)
def test_documentation_only_paths_are_explicitly_allowlisted(path: str) -> None:
    assert classify_changes.is_documentation_only_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "scripts/ci/validate_docs.py",
        ".github/workflows/deploy-docs.yml",
        "examples/rofental/rofental.yml",
        "openamundsen_da/api.py",
        "tests/baselines/rofental_es30_manuscript_assets.json",
        "notes/CHANGELOG.md",
        "../docs/index.md",
        "/docs/index.md",
        " docs/index.md",
        "",
    ],
)
def test_every_non_allowlisted_or_uncertain_path_requires_full_ci(path: str) -> None:
    assert classify_changes.is_documentation_only_path(path) is False


def test_mixed_empty_and_forced_diffs_fail_safe_to_full_ci() -> None:
    assert classify_changes.classify_paths(["docs/index.md"]) is True
    assert classify_changes.classify_paths(["docs/index.md", "CHANGELOG.md"]) is True
    assert classify_changes.classify_paths(["docs/index.md", "README.md"]) is False
    assert classify_changes.classify_paths([]) is False
    assert classify_changes.classify_paths(["docs/index.md"], force_full=True) is False


def test_cli_writes_boolean_github_output(tmp_path: Path) -> None:
    paths_file = tmp_path / "paths.txt"
    output = tmp_path / "output.txt"
    paths_file.write_text(
        "docs/Tutorial/07-results-and-diagnostics.md\n"
        "tests/baselines/rofental_es30_tutorial_assets.json\n",
        encoding="utf-8",
    )

    result = classify_changes.main(
        ["--paths-file", str(paths_file), "--github-output", str(output)]
    )

    assert result == 0
    assert output.read_text(encoding="utf-8") == "docs_only=true\n"


def test_cli_missing_diff_file_fails_safe_to_full_ci(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"

    result = classify_changes.main(
        ["--paths-file", str(tmp_path / "missing"), "--github-output", str(output)]
    )

    assert result == 0
    assert output.read_text(encoding="utf-8") == "docs_only=false\n"

from __future__ import annotations

import tomllib
from pathlib import Path


def test_only_umbrella_console_script_is_published() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert data["project"]["scripts"] == {
        "openamundsen-da": "openamundsen_da.cli:main",
    }

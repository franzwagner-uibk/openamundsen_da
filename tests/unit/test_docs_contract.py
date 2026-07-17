from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).parents[2]


def _load_script(relative_path: str, module_name: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_curated_cli_guide_covers_supported_workflows() -> None:
    guide = (ROOT / "docs/reference/cli.md").read_text(encoding="utf-8")

    assert "## Single-domain data assimilation" in guide
    assert "## Subdomain data assimilation" in guide
    assert "## Plain-model subdomains" in guide
    assert "openamundsen-da observations snow-cover PROJECT_DIR" in guide
    assert "openamundsen-da subdomains render PROJECT_DIR" in guide
    assert "openamundsen-da subdomains model merge SETUP_DIR" in guide
    assert "scripts/docs/render_cli_reference.py" not in guide
    assert "usage: openamundsen-da" not in guide
    assert "oa-da-" not in guide


def test_published_documentation_contract_is_current() -> None:
    module = _load_script("scripts/ci/validate_docs.py", "validate_docs")

    assert module.validate_docs() == ()

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


def test_generated_cli_reference_covers_supported_command_tree() -> None:
    module = _load_script("scripts/docs/render_cli_reference.py", "render_cli_reference")

    rendered = module.render_cli_reference()

    assert "`openamundsen-da observations snow-cover`" in rendered
    assert "`openamundsen-da clean`" in rendered
    assert "`openamundsen-da subdomains render`" in rendered
    assert "`openamundsen-da subdomains model merge`" in rendered
    assert "oa-da-" not in rendered


def test_published_documentation_contract_is_current() -> None:
    module = _load_script("scripts/ci/validate_docs.py", "validate_docs")

    assert module.validate_docs() == ()


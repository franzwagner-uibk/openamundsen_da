from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_module(name: str, relative_path: str):
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


validate_trimmed_project = _load_module(
    "validate_trimmed_project",
    "scripts/ci/validate_trimmed_project.py",
)
validate_trimmed_subdomain = _load_module(
    "validate_trimmed_subdomain",
    "scripts/ci/validate_trimmed_subdomain.py",
)
validate_trimmed_model_subdomain = _load_module(
    "validate_trimmed_model_subdomain",
    "scripts/ci/validate_trimmed_model_subdomain.py",
)


@pytest.mark.parametrize(
    ("module", "warning_text"),
    [
        (
            validate_trimmed_project,
            "2026-04-13 12:13:04.372 | WARNING  | "
            "Skipping analysis benchmark for scf on 2022-11-22: missing observation row",
        ),
        (
            validate_trimmed_subdomain,
            "2026-04-13 12:13:04.372 | WARNING  | "
            "Skipping analysis benchmark for wet_snow on 2023-05-26: missing observation row",
        ),
        (
            validate_trimmed_model_subdomain,
            "2026-04-13 12:13:04.372 | WARNING  | "
            "Skipping analysis benchmark for scf on 2022-11-22: missing observation row",
        ),
    ],
)
def test_check_logs_allows_benchmark_missing_observation_row_warning(tmp_path: Path, module, warning_text: str):
    log_file = tmp_path / "integration.log"
    log_file.write_text(
        f"{warning_text}\n2026-04-13 12:13:14.669 | INFO     | Project processing complete\n",
        encoding="utf-8",
    )

    module._check_logs(log_file)


@pytest.mark.parametrize(
    "module",
    [validate_trimmed_project, validate_trimmed_subdomain, validate_trimmed_model_subdomain],
)
def test_check_logs_allows_wsla_continuous_missing_model_values_warning(tmp_path: Path, module):
    log_file = tmp_path / "integration.log"
    log_file.write_text(
        "2026-04-29 14:32:40.609 | WARNING  | "
        "Skipping wet_snow_line benchmark case at 2023-03-28: missing model values\n",
        encoding="utf-8",
    )

    module._check_logs(log_file)


@pytest.mark.parametrize(
    "module",
    [validate_trimmed_project, validate_trimmed_subdomain, validate_trimmed_model_subdomain],
)
def test_check_logs_still_rejects_other_missing_warnings(tmp_path: Path, module):
    log_file = tmp_path / "integration.log"
    log_file.write_text(
        "2026-04-13 12:13:04.372 | WARNING  | Missing some required downstream artifact\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="severe warning lines"):
        module._check_logs(log_file)

from __future__ import annotations

"""
openamundsen_da.core.env

Purpose
- Apply project-level GDAL/PROJ/numeric-thread settings so every runner sees a consistent environment.
-
Key Behaviors
- Read the `environment` block from `project.yml` and export the constrained list of vars.
- Fall back to active conda `CONDA_PREFIX`/`PREFIX` for GDAL/PROJ when keys are missing.
- Enforce a single-thread limit for numeric libraries and capture snapshots for diagnostics.

Inputs/Outputs
- `apply_env_from_project(path)` exports vars and returns the subset applied.
- `ensure_gdal_proj_from_conda()` and `apply_numeric_thread_defaults()` mutate `os.environ`.
- `snapshot_env(keys)` reports the current values of the selected variables.

Assumptions
- `project.yml` is UTF-8 readable and may omit the `environment` block entirely.
- Conda prefixes live under `CONDA_PREFIX` or `PREFIX` when present.
"""

import os
from pathlib import Path
from typing import Iterable, Dict

import ruamel.yaml

from openamundsen_da.core.constants import ENVIRONMENT, ENV_VARS_EXPORT


_yaml = ruamel.yaml.YAML(typ="safe")


def _read_yaml_file(p: Path) -> dict:
    try:
        with Path(p).open("r", encoding="utf-8") as f:
            return _yaml.load(f) or {}
    except Exception:
        return {}


def apply_env_from_project(project_yaml: Path) -> Dict[str, str]:
    """Read environment section from project.yml and export selected vars.

    Returns a dict of the keys applied (subset of ENV_VARS_EXPORT) with their values.
    """
    cfg = _read_yaml_file(Path(project_yaml))
    env_cfg = (cfg or {}).get(ENVIRONMENT) or {}
    applied: Dict[str, str] = {}
    for k in ENV_VARS_EXPORT:
        v = env_cfg.get(k)
        if v:
            os.environ[k] = str(v)
            applied[k] = os.environ[k]
    return applied


def ensure_gdal_proj_from_conda() -> None:
    """If GDAL/PROJ vars are unset, set them based on the current conda env."""
    conda = os.environ.get("CONDA_PREFIX") or os.environ.get("PREFIX")
    if not conda:
        return
    os.environ.setdefault("GDAL_DATA", str(Path(conda) / "Library" / "share" / "gdal"))
    os.environ.setdefault("PROJ_LIB", str(Path(conda) / "Library" / "share" / "proj"))


def apply_numeric_thread_defaults() -> None:
    """Pin numeric library threads to 1 unless already configured."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def snapshot_env(keys: Iterable[str]) -> Dict[str, str | None]:
    """Return current values for selected environment variables."""
    return {k: os.environ.get(k) for k in keys}


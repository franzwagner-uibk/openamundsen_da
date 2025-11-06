from __future__ import annotations
from pathlib import Path
from typing import Union

from openamundsen_da.core.constants import (
    ENSEMBLE_PRIOR,
    MEMBER_PREFIX,
    OPEN_LOOP,
    VAR_HS,
    VAR_SWE,
)

# ---- YAML discovery helpers -------------------------------------------------

def find_project_yaml(project_dir: str | Path) -> Path:
    project_dir = Path(project_dir)
    for name in ("project.yml", "project.yaml"):
        p = project_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not find project.yml in {project_dir}")

def find_season_yaml(season_dir: str | Path) -> Path:
    season_dir = Path(season_dir)
    for name in ("season.yml", "season.yaml"):
        p = season_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not find season.yml in {season_dir}")

def find_step_yaml(step_dir: str | Path) -> Path:
    step_dir = Path(step_dir)
    # allow flexible step file name (e.g. step_00.yml)
    ymls = sorted(step_dir.glob("*.yml")) + sorted(step_dir.glob("*.yaml"))
    if not ymls:
        raise FileNotFoundError(f"No step YAML found in {step_dir}")
    return ymls[0]

# ---- Ensemble layout helpers -----------------------------------------------

def meteo_dir_for_member(member_dir: str | Path) -> Path:
    """Member meteo directory: <member_dir>/meteo"""
    return Path(member_dir) / "meteo"

def default_results_dir(member_dir: str | Path) -> Path:
    """Default outputs under <member_dir>/results"""
    return Path(member_dir) / "results"

def list_member_dirs(base_dir: str | Path, ensemble: str) -> list[Path]:
    base_dir = Path(base_dir)
    if ensemble not in {"prior", "posterior"}:
        raise ValueError("ensemble must be 'prior' or 'posterior'")

    roots = [base_dir / ensemble, base_dir / "ensembles" / ensemble]
    for root in roots:
        if root.is_dir():
            return [p for p in sorted(root.glob("member_*")) if p.is_dir()]
    return []


# ---- Generic path helpers ---------------------------------------------------

def abspath_relative_to(base: str | Path, p: str | Path) -> str:
    """Return absolute path string, resolving `p` against `base` if relative."""
    base = Path(base)
    pp = Path(p)
    return str(pp if pp.is_absolute() else (base / pp))


# ---- Prior ensemble layout helpers -----------------------------------------

PathLike = Union[str, Path]

def prior_root(step_dir: PathLike) -> Path:
    """<step_dir>/ensembles/prior root directory."""
    return Path(step_dir) / "ensembles" / ENSEMBLE_PRIOR

def open_loop_dir(step_dir: PathLike) -> Path:
    """<step_dir>/ensembles/prior/open_loop directory."""
    return prior_root(step_dir) / OPEN_LOOP

def member_dir_for_index(step_dir: PathLike, index: int, width: int = 3) -> Path:
    """Member directory path using zero-padded index: member_XXX."""
    name = f"{MEMBER_PREFIX}{index:0{width}d}"
    return prior_root(step_dir) / name


# ---- Member results helpers -------------------------------------------------

def member_id_from_results_dir(results_dir: str | Path) -> str:
    """Return member ID (e.g., 'member_001') given a member results dir."""
    return Path(results_dir).parent.name


def find_member_daily_raster(results_dir: str | Path, variable: str, date_str: str) -> Path:
    """Find a daily raster for a given variable and date in a member results dir.

    Parameters
    ----------
    results_dir : Path-like
        Path to the member results directory (contains daily GeoTIFFs).
    variable : str
        One of VAR_HS ('hs') or VAR_SWE ('swe').
    date_str : str
        Date string in 'YYYY-MM-DD' format.

    Returns
    -------
    Path
        Path to the first matching raster.
    """
    results_dir = Path(results_dir)
    if variable == VAR_HS:
        prefix = "snowdepth_daily_"
    elif variable == VAR_SWE:
        prefix = "swe_daily_"
    else:
        raise ValueError(f"Unknown variable '{variable}', expected '{VAR_HS}' or '{VAR_SWE}'")
    patt = f"{prefix}{date_str}T*.tif"
    matches = sorted(results_dir.glob(patt))
    if not matches:
        raise FileNotFoundError(f"No raster found matching {patt} in {results_dir}")
    return matches[0]

from __future__ import annotations
from pathlib import Path

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

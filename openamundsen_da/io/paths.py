from pathlib import Path
from typing import Iterable, List


def find_project_yaml(project_dir: Path | str) -> Path:
    """
    Return path to project.yml inside the project root directory.
    Raises FileNotFoundError if missing.
    """
    p = Path(project_dir) / "project.yml"
    if not p.is_file():
        raise FileNotFoundError(f"project.yml not found at: {p}")
    return p


def find_season_yaml(season_dir: Path | str) -> Path:
    """
    Return path to season.yml inside a season folder (e.g. propagation/season_YYYY-YYYY).
    """
    p = Path(season_dir) / "season.yml"
    if not p.is_file():
        raise FileNotFoundError(f"season.yml not found at: {p}")
    return p


def find_step_yaml(step_dir: Path | str) -> Path:
    """
    Return the single step YAML inside a step folder.
    Convention: exactly one *.yml file at the step root (e.g. step_00.yml).
    """
    step_dir = Path(step_dir)
    ymls = [p for p in step_dir.glob("*.yml") if p.is_file()]
    if len(ymls) != 1:
        raise FileNotFoundError(
            f"Expected exactly one step YAML in {step_dir}, found: {[p.name for p in ymls]}"
        )
    return ymls[0]


def discover_prior_members(step_dir: Path | str) -> List[Path]:
    """
    List member directories under: <step_dir>/ensembles/prior/member_*/
    Returns sorted paths (by name). Empty list if none.
    """
    base = Path(step_dir) / "ensembles" / "prior"
    if not base.is_dir():
        return []
    members = [p for p in base.glob("member_*") if p.is_dir()]
    return sorted(members, key=lambda p: p.name)


def meteo_dir_for_member(member_dir: Path | str) -> Path:
    """
    Conventional meteo input location for a member.
    """
    return Path(member_dir) / "meteo"


def default_results_dir(member_dir: Path | str) -> Path:
    """
    Conventional results output location for a member.
    Note: this function only computes the path; it does NOT create any folders.
    """
    return Path(member_dir) / "results"

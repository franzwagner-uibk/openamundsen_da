from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

from openamundsen.conf import parse_config
from openamundsen.util import read_yaml_file, to_yaml
from openamundsen_da.io.paths import abspath_relative_to
from openamundsen_da.core.constants import (
    INPUT_DATA,
    GRIDS,
    METEO,
    DIR,
    RESULTS_DIR,
    START_DATE,
    END_DATE,
    LOG_LEVEL,
    ENVIRONMENT,
)

def merge_configs(
    project_cfg: Dict[str, Any],
    season_cfg: Dict[str, Any],
    step_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Shallow top-level merge with precedence: step > season > project."""
    merged: Dict[str, Any] = {}
    keys = set()
    for d in (project_cfg or {}, season_cfg or {}, step_cfg or {}):
        keys.update(d.keys())
    for k in keys:
        p, s, t = (project_cfg or {}).get(k), (season_cfg or {}).get(k), (step_cfg or {}).get(k)
        if isinstance(p, dict) or isinstance(s, dict) or isinstance(t, dict):
            merged[k] = {}
            if isinstance(p, dict): merged[k].update(p)
            if isinstance(s, dict): merged[k].update(s)
            if isinstance(t, dict): merged[k].update(t)
        else:
            merged[k] = t if t is not None else (s if s is not None else p)
    return merged


## moved to openamundsen_da.io.paths.abspath_relative_to

def load_merged_config(
    project_yaml: Path | str,
    season_yaml: Path | str,
    step_yaml: Path | str,
    *,
    member_meteo_dir: Path | str,
    results_dir: Optional[Path | str] = None,
    log_level: Optional[str] = None,
) -> Any:
    """
    Build and validate an openAMUNDSEN Configuration by merging YAML layers and
    injecting member-specific meteo/results paths.
    """
    project_yaml = Path(project_yaml)
    season_yaml = Path(season_yaml)
    step_yaml   = Path(step_yaml)

    proj_cfg = read_yaml_file(project_yaml)
    seas_cfg = read_yaml_file(season_yaml)
    step_cfg = read_yaml_file(step_yaml)

    cfg = merge_configs(proj_cfg, seas_cfg, step_cfg)

    cfg.pop(ENVIRONMENT, None)

    # Inject per-member paths
    cfg.setdefault(INPUT_DATA, {}).setdefault(METEO, {})
    cfg[INPUT_DATA][METEO][DIR] = str(Path(member_meteo_dir))

    if results_dir is not None:
        cfg[RESULTS_DIR] = str(results_dir)

    # Ensure log level from CLI is applied to openAMUNDSEN logger configuration
    if log_level is not None:
        cfg[LOG_LEVEL] = str(log_level)

    # Make important paths absolute relative to project root
    project_root = project_yaml.parent

    # grids dir
    if INPUT_DATA in cfg and GRIDS in cfg[INPUT_DATA] and DIR in cfg[INPUT_DATA][GRIDS]:
        cfg[INPUT_DATA][GRIDS][DIR] = abspath_relative_to(project_root, cfg[INPUT_DATA][GRIDS][DIR])

    # meteo dir (ensure absolute)
    if INPUT_DATA in cfg and METEO in cfg[INPUT_DATA] and DIR in cfg[INPUT_DATA][METEO]:
        cfg[INPUT_DATA][METEO][DIR] = abspath_relative_to(project_root, cfg[INPUT_DATA][METEO][DIR])

    # results_dir
    if RESULTS_DIR in cfg:
        cfg[RESULTS_DIR] = abspath_relative_to(project_root, cfg[RESULTS_DIR])
    else:
        cfg[RESULTS_DIR] = str(project_root / "results")

    # Basic time keys
    for k in (START_DATE, END_DATE):
        if k not in cfg or not cfg[k]:
            raise ValueError(f"Missing required key '{k}' in merged config.")


    try:
        member_root = Path(member_meteo_dir).parent  # .../member_xxx
        member_root.mkdir(parents=True, exist_ok=True)
        (member_root / "config.yml").write_text(to_yaml(cfg), encoding="utf-8")
    except Exception:
        # Best-effort; do not fail the run on write issues
        pass

    return parse_config(cfg)

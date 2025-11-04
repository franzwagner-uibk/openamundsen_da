from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional

from openamundsen.conf import parse_config
from openamundsen.util import read_yaml_file, to_yaml

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

    cfg.pop("environment", None)

    # Inject per-member paths
    cfg.setdefault("input_data", {}).setdefault("meteo", {})
    cfg["input_data"]["meteo"]["dir"] = str(Path(member_meteo_dir))

    if results_dir is not None:
        cfg["results_dir"] = str(results_dir)

    # Ensure log level from CLI is applied to openAMUNDSEN logger configuration
    if log_level is not None:
        cfg["log_level"] = str(log_level)

    # Make important paths absolute relative to project root
    project_root = project_yaml.parent

    # grids dir
    if "input_data" in cfg and "grids" in cfg["input_data"] and "dir" in cfg["input_data"]["grids"]:
        gdir = Path(cfg["input_data"]["grids"]["dir"])
        if not gdir.is_absolute():
            gdir = project_root / gdir
        cfg["input_data"]["grids"]["dir"] = str(gdir)

    # meteo dir (ensure absolute)
    if "input_data" in cfg and "meteo" in cfg["input_data"] and "dir" in cfg["input_data"]["meteo"]:
        mdir = Path(cfg["input_data"]["meteo"]["dir"])
        if not mdir.is_absolute():
            mdir = project_root / mdir
        cfg["input_data"]["meteo"]["dir"] = str(mdir)

    # results_dir
    if "results_dir" in cfg:
        rdir = Path(cfg["results_dir"])
        if not rdir.is_absolute():
            rdir = project_root / rdir
        cfg["results_dir"] = str(rdir)
    else:
        cfg["results_dir"] = str(project_root / "results")

    # Basic time keys
    for k in ("start_date", "end_date"):
        if k not in cfg or not cfg[k]:
            raise ValueError(f"Missing required key '{k}' in merged config.")


    # Gemergte Config im Member-Ordner als 'config.py' ablegen (Sibling von 'results')
    try:
        member_root = Path(member_meteo_dir).parent  # .../member_xxx
        member_root.mkdir(parents=True, exist_ok=True)
        (member_root / "config.yml").write_text(to_yaml(cfg), encoding="utf-8")
    except Exception:
        # Best-effort; Fehler hier sollen den Lauf nicht abbrechen
        pass

    # Final validation
    return parse_config(cfg)

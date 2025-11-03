# Minimal config merging using openAMUNDSEN utilities.
# - Reads project.yml, season.yml, step.yml with openamundsen.util.read_yaml_file
# - Merges layers with priority: step > season > project
# - Injects per-member meteo directory (mandatory)
# - Optionally sets results_dir
# - Returns a fully validated Configuration via openamundsen.conf.parse_config

from pathlib import Path
from typing import Any, Dict, Optional

from openamundsen.conf import parse_config
from openamundsen.util import read_yaml_file


def merge_configs(
    project_cfg: Dict[str, Any],
    season_cfg: Dict[str, Any],
    step_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge three configuration layers with fixed precedence:
    step_cfg > season_cfg > project_cfg.
    Nested dictionaries are merged shallowly per top-level key.
    """
    merged: Dict[str, Any] = {}
    all_keys = set()
    for d in (project_cfg or {}, season_cfg or {}, step_cfg or {}):
        all_keys.update(d.keys())

    for k in all_keys:
        p = (project_cfg or {}).get(k)
        s = (season_cfg or {}).get(k)
        t = (step_cfg or {}).get(k)

        if isinstance(p, dict) or isinstance(s, dict) or isinstance(t, dict):
            merged[k] = {}
            if isinstance(p, dict):
                merged[k].update(p)
            if isinstance(s, dict):
                merged[k].update(s)
            if isinstance(t, dict):
                merged[k].update(t)
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
) -> Any:
    """
    Build and validate an openAMUNDSEN Configuration by merging YAML layers.

    Parameters
    ----------
    project_yaml : Path or str
        Path to project.yml (global settings).
    season_yaml : Path or str
        Path to season.yml (season-wide configuration).
    step_yaml : Path or str
        Path to step.yml (per-step overrides, may define results_dir or shorter date range).
    member_meteo_dir : Path or str
        Directory of the ensemble member's meteorological input.
        Injected as config.input_data.meteo.dir.
    results_dir : Optional[Path or str]
        Optional override for config.results_dir.
        If None, uses the layered value or defaults to "results".

    Returns
    -------
    openamundsen.conf.Configuration
        Fully parsed and validated Configuration object.
    """
    project_yaml = Path(project_yaml)
    season_yaml = Path(season_yaml)
    step_yaml = Path(step_yaml)

    # Load YAML configs using openAMUNDSEN’s native reader
    proj_cfg = read_yaml_file(project_yaml)
    seas_cfg = read_yaml_file(season_yaml)
    step_cfg = read_yaml_file(step_yaml)

    # Merge configurations with defined precedence
    cfg = merge_configs(proj_cfg, seas_cfg, step_cfg)

    # Ensure meteo key exists before injection
    cfg.setdefault("input_data", {})
    cfg["input_data"].setdefault("meteo", {})

    # Inject ensemble-specific meteo directory
    cfg["input_data"]["meteo"]["dir"] = str(Path(member_meteo_dir))

    # Resolve results directory logic
    if results_dir is not None:
        cfg["results_dir"] = str(Path(results_dir))
    else:
        if "results_dir" not in cfg or not cfg["results_dir"]:
            cfg["results_dir"] = "results"

    # Basic time range sanity check
    missing = [k for k in ("start_date", "end_date") if k not in cfg or not cfg[k]]
    if missing:
        raise ValueError(
            f"Missing required key(s) in merged configuration: {', '.join(missing)}. "
            "Provide them in season.yml and/or step.yml."
        )

    # Final validation + normalization through openAMUNDSEN core
    return parse_config(cfg)

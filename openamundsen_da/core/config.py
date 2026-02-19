from __future__ import annotations
"""
openamundsen_da.core.config

Purpose
- Build a validated openAMUNDSEN configuration for a single ensemble member by
  layering setup/project/step YAML, injecting per-member paths, normalizing
  important paths to absolute, and returning the parsed OA config object.

Key Behaviors
- Shallow (top-level) merge with precedence: step > project > setup.
- Removes the `environment` block (environment is handled centrally in core.env).
- Injects member-specific `input_data.meteo.dir` and optional `results_dir`.
- Applies CLI-provided `log_level` so OA internal logger uses the same level.
- Writes the merged YAML next to the member for provenance.

Inputs
- Paths to setup YAML, project YAML, and a step YAML (any *.yml in the step).
- Member meteo directory and optional explicit results directory.

Outputs
- Parsed openAMUNDSEN configuration (via openamundsen.conf.parse_config).
"""

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
    DA_BLOCK,
)


def merge_configs(
    setup_cfg: Dict[str, Any],
    project_cfg: Dict[str, Any],
    step_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Shallow top-level merge with precedence: step > project > setup.

    Parameters
    - setup_cfg: Parsed setup YAML as dict
    - project_cfg: Parsed project YAML as dict
    - step_cfg: Parsed step YAML as dict

    Returns
    - merged: dict containing all top-level keys with later layers overriding
      earlier ones. For nested dicts, the merge is shallow (no deep merge).
    """
    merged: Dict[str, Any] = {}
    keys = set()
    for d in (setup_cfg or {}, project_cfg or {}, step_cfg or {}):
        keys.update(d.keys())
    for k in keys:
        s, p, t = (setup_cfg or {}).get(k), (project_cfg or {}).get(k), (step_cfg or {}).get(k)
        if isinstance(s, dict) or isinstance(p, dict) or isinstance(t, dict):
            merged[k] = {}
            if isinstance(s, dict):
                merged[k].update(s)
            if isinstance(p, dict):
                merged[k].update(p)
            if isinstance(t, dict):
                merged[k].update(t)
        else:
            merged[k] = t if t is not None else (p if p is not None else s)
    return merged


## moved to openamundsen_da.io.paths.abspath_relative_to


def load_merged_config(
    setup_yaml: Path | str,
    project_yaml: Path | str,
    step_yaml: Path | str,
    *,
    member_meteo_dir: Path | str,
    results_dir: Optional[Path | str] = None,
    log_level: Optional[str] = None,
) -> Any:
    """
    Build and validate an openAMUNDSEN Configuration by merging YAML layers and
    injecting member-specific paths.

    Steps
    1) Read YAML layers
    2) Merge layers (step > project > setup)
    3) Drop environment block (handled by core.env)
    4) Inject member meteo/results paths and optional log level
    5) Normalize critical paths to absolute
    6) Basic validation of required keys
    7) Persist merged YAML near the member for provenance
    """
    setup_yaml = Path(setup_yaml)
    project_yaml = Path(project_yaml)
    step_yaml = Path(step_yaml)

    # Step 1: Read YAML layers
    setup_cfg = read_yaml_file(setup_yaml)
    proj_cfg = read_yaml_file(project_yaml)
    step_cfg = read_yaml_file(step_yaml)

    # Step 2: Merge and Step 3: remove blocks not understood by OA parser
    cfg = merge_configs(setup_cfg, proj_cfg, step_cfg)
    cfg.pop(ENVIRONMENT, None)
    cfg.pop(DA_BLOCK, None)
    cfg.pop("obs", None)
    cfg.pop("run_mode", None)
    cfg.pop("copernicus_download", None)

    # Step 4: Inject per-member paths (meteo dir always, results optional)
    cfg.setdefault(INPUT_DATA, {}).setdefault(METEO, {})
    cfg[INPUT_DATA][METEO][DIR] = str(Path(member_meteo_dir))

    if results_dir is not None:
        cfg[RESULTS_DIR] = str(results_dir)

    # Apply CLI log level to OA logger if provided
    if log_level is not None:
        cfg[LOG_LEVEL] = str(log_level)

    # Step 5: Normalize important paths to absolute (relative to setup root)
    setup_root = setup_yaml.parent

    # grids dir
    if INPUT_DATA in cfg and GRIDS in cfg[INPUT_DATA] and DIR in cfg[INPUT_DATA][GRIDS]:
        cfg[INPUT_DATA][GRIDS][DIR] = abspath_relative_to(setup_root, cfg[INPUT_DATA][GRIDS][DIR])

    # meteo dir (ensure absolute)
    if INPUT_DATA in cfg and METEO in cfg[INPUT_DATA] and DIR in cfg[INPUT_DATA][METEO]:
        cfg[INPUT_DATA][METEO][DIR] = abspath_relative_to(setup_root, cfg[INPUT_DATA][METEO][DIR])

    # results_dir
    if RESULTS_DIR in cfg:
        cfg[RESULTS_DIR] = abspath_relative_to(setup_root, cfg[RESULTS_DIR])
    else:
        cfg[RESULTS_DIR] = str(setup_root / "results")

    # Step 6: Basic validation of required time keys
    for k in (START_DATE, END_DATE):
        if k not in cfg or not cfg[k]:
            raise ValueError(f"Missing required key '{k}' in merged config.")

    # Step 7: Persist merged config near the member for provenance (best-effort)
    try:
        member_root = Path(member_meteo_dir).parent  # .../member_xxx
        member_root.mkdir(parents=True, exist_ok=True)
        (member_root / "config.yml").write_text(to_yaml(cfg), encoding="utf-8")
    except Exception:
        # Best-effort; do not fail the run on write issues
        pass

    return parse_config(cfg)

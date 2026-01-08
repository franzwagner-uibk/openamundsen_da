"""Cleanup utilities for removing season state pickle files.

Features
- Reads state filename pattern from project.yml (data_assimilation.restart.state_pattern).
- Default patterns: configured pattern + model_state.pickle.gz.
- Cleans a single season or all seasons under project/propagation.
- Intended for manual use (CLI) and automatic use after successful season runs.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from loguru import logger

from openamundsen_da.core.constants import (
    DA_BLOCK,
    LOGURU_FORMAT,
    RESTART_BLOCK,
    RESTART_CLEANUP_AFTER_SEASON,
    RESTART_STATE_PATTERN,
    STATE_DEFAULT_NAME,
)
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml


def _read_restart_config(project_dir: Path) -> dict:
    """Return restart config dict from project.yml (best-effort)."""
    try:
        proj = find_project_yaml(project_dir)
        cfg = _read_yaml_file(proj) or {}
        da_cfg = cfg.get(DA_BLOCK) or {}
        return da_cfg.get(RESTART_BLOCK) or {}
    except Exception:
        return {}


def state_patterns_from_project(project_dir: Path) -> List[str]:
    """Return state filename patterns (configured + default)."""
    restart_cfg = _read_restart_config(project_dir)
    patt = restart_cfg.get(RESTART_STATE_PATTERN) or STATE_DEFAULT_NAME
    patterns = [str(patt), STATE_DEFAULT_NAME]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in patterns:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def is_cleanup_enabled(project_dir: Path) -> bool:
    """Return True if cleanup_after_season is enabled (default: True)."""
    restart_cfg = _read_restart_config(project_dir)
    val = restart_cfg.get(RESTART_CLEANUP_AFTER_SEASON)
    return True if val is None else bool(val)


def _list_season_dirs(project_dir: Path) -> List[Path]:
    """Return season directories under project/propagation/ with a season.yml."""
    prop = Path(project_dir) / "propagation"
    if not prop.is_dir():
        return []
    seasons: List[Path] = []
    for cand in sorted(prop.glob("season_*")):
        if not cand.is_dir():
            continue
        if any((cand / name).is_file() for name in ("season.yml", "season.yaml")):
            seasons.append(cand)
    return seasons


def _iter_state_files(season_dir: Path, patterns: Sequence[str]) -> Iterable[Path]:
    """Yield state files under season_dir matching any pattern, limited to results dirs."""
    for patt in patterns:
        for p in season_dir.rglob(patt):
            if p.is_file() and "results" in p.parts:
                yield p


@dataclass
class CleanupSummary:
    season_dir: Path
    patterns: List[str]
    files_deleted: int
    bytes_freed: int
    attempted: int


def cleanup_season_dir(
    *,
    project_dir: Path,
    season_dir: Path,
    patterns: Sequence[str] | None = None,
) -> CleanupSummary:
    """Delete state pickle files for a given season.

    Parameters
    ----------
    project_dir : Path
        Project root (for reading configuration).
    season_dir : Path
        Season directory containing step_* subdirectories.
    patterns : Sequence[str] | None
        Optional explicit patterns; defaults to project config + default name.
    """
    season_dir = Path(season_dir)
    if not season_dir.is_dir():
        raise FileNotFoundError(f"Season directory not found: {season_dir}")

    pats = list(patterns) if patterns is not None else state_patterns_from_project(project_dir)
    seen = set()
    files = []
    for f in _iter_state_files(season_dir, pats):
        rp = f.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        files.append(f)

    bytes_freed = 0
    deleted = 0
    for f in files:
        try:
            size = f.stat().st_size
        except Exception:
            size = 0
        try:
            f.unlink()
            deleted += 1
            bytes_freed += size
            logger.debug("Deleted state file {}", f)
        except Exception as exc:
            logger.warning("Could not delete {}: {}", f, exc)

    return CleanupSummary(
        season_dir=season_dir,
        patterns=pats,
        files_deleted=deleted,
        bytes_freed=bytes_freed,
        attempted=len(files),
    )


def cli_main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="oa-da-clean-season",
        description="Remove model state pickle files for a season (or all seasons) to free disk space.",
    )
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--season-dir", type=Path, help="Season directory to clean (e.g., /data/propagation/season_2020-2021)")
    p.add_argument("--all-seasons", action="store_true", help="Clean all seasons under project/propagation")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)

    logger.remove()
    logger.add(sys.stdout, level=str(args.log_level or "INFO").upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    if not args.all_seasons and args.season_dir is None:
        p.error("Provide --season-dir or --all-seasons")
    if args.all_seasons and args.season_dir is not None:
        p.error("Use either --season-dir or --all-seasons, not both")

    project_dir = Path(args.project_dir)
    seasons: List[Path]
    if args.all_seasons:
        seasons = _list_season_dirs(project_dir)
        if not seasons:
            logger.error("No season directories found under {}/propagation", project_dir)
            return 1
    else:
        seasons = [Path(args.season_dir)]

    pats = state_patterns_from_project(project_dir)
    total_files = 0
    total_bytes = 0
    for s in seasons:
        try:
            summary = cleanup_season_dir(project_dir=project_dir, season_dir=s, patterns=pats)
        except Exception as exc:
            logger.error("Cleanup failed for {}: {}", s, exc)
            return 1
        total_files += summary.files_deleted
        total_bytes += summary.bytes_freed
        logger.info(
            "Cleaned {} -> removed {} file(s), freed {:.1f} MB (patterns={})",
            s,
            summary.files_deleted,
            summary.bytes_freed / 1_000_000.0,
            ",".join(summary.patterns),
        )

    logger.info(
        "Cleanup complete | seasons={} files={} freed={:.1f} MB",
        len(seasons),
        total_files,
        total_bytes / 1_000_000.0,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

"""Cleanup utilities for removing project state pickle files."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from loguru import logger

from openamundsen_da.core.constants import (
    DA_BLOCK,
    RESTART_BLOCK,
    RESTART_CLEANUP_AFTER_SETUP,
    RESTART_STATE_PATTERN,
    STATE_DEFAULT_NAME,
)
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml
from openamundsen_da.util.loguru_utils import configure_cli_logger


def _read_restart_config(project_dir: Path) -> dict:
    """Return restart config dict from project YAML (best-effort)."""
    try:
        project_yaml = find_project_yaml(project_dir)
        cfg = _read_yaml_file(project_yaml) or {}
        da_cfg = cfg.get(DA_BLOCK) or {}
        return da_cfg.get(RESTART_BLOCK) or {}
    except Exception:
        return {}


def state_patterns_from_setup(project_dir: Path) -> List[str]:
    """Return state filename patterns (configured + default).

    Note: function name kept for internal compatibility; input is a project dir.
    """
    restart_cfg = _read_restart_config(project_dir)
    patt = restart_cfg.get(RESTART_STATE_PATTERN) or STATE_DEFAULT_NAME
    patterns = [str(patt), STATE_DEFAULT_NAME]
    seen = set()
    unique = []
    for p in patterns:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


def is_cleanup_enabled(project_dir: Path) -> bool:
    """Return True if cleanup_after_setup is enabled (default: True)."""
    restart_cfg = _read_restart_config(project_dir)
    val = restart_cfg.get(RESTART_CLEANUP_AFTER_SETUP)
    return True if val is None else bool(val)


def _list_project_dirs(setup_dir: Path) -> List[Path]:
    """Return project directories under setup/projects with a project YAML."""
    projects_root = Path(setup_dir) / "projects"
    if not projects_root.is_dir():
        return []
    projects: List[Path] = []
    for cand in sorted(projects_root.iterdir()):
        if not cand.is_dir():
            continue
        try:
            _ = find_project_yaml(cand)
            projects.append(cand)
        except FileNotFoundError:
            continue
    return projects


def _iter_state_files(project_dir: Path, patterns: Sequence[str]) -> Iterable[Path]:
    """Yield state files under project_dir matching any pattern, limited to results dirs."""
    for patt in patterns:
        for p in project_dir.rglob(patt):
            if p.is_file() and "results" in p.parts:
                yield p


@dataclass
class CleanupSummary:
    project_dir: Path
    patterns: List[str]
    files_deleted: int
    bytes_freed: int
    attempted: int
    failures: int


def cleanup_setup_dir(
    *,
    setup_dir: Path,
    patterns: Sequence[str] | None = None,
) -> CleanupSummary:
    """Delete state pickle files for a given project directory.

    Note: parameter name kept for internal compatibility; pass project dir.
    """
    project_dir = Path(setup_dir)
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")

    pats = list(patterns) if patterns is not None else state_patterns_from_setup(project_dir)
    seen = set()
    files = []
    for f in _iter_state_files(project_dir, pats):
        rp = f.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        files.append(f)

    bytes_freed = 0
    deleted = 0
    failures = 0
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
            failures += 1

    # Sub-domain mode keeps most heavy artifacts under project/subdomains.
    subdomains_dir = project_dir / "subdomains"
    if subdomains_dir.is_dir():
        dir_bytes = 0
        try:
            for p in subdomains_dir.rglob("*"):
                if p.is_file():
                    try:
                        dir_bytes += p.stat().st_size
                    except Exception:
                        pass
            shutil.rmtree(subdomains_dir)
            bytes_freed += dir_bytes
            logger.debug("Deleted sub-domain workspace {}", subdomains_dir)
        except Exception as exc:
            logger.warning("Could not delete sub-domain workspace {}: {}", subdomains_dir, exc)
            failures += 1

    subdomain_log = project_dir / "subdomain_run.log"
    if subdomain_log.is_file():
        try:
            log_size = subdomain_log.stat().st_size
        except Exception:
            log_size = 0
        try:
            subdomain_log.unlink()
            bytes_freed += log_size
            logger.debug("Deleted sub-domain run log {}", subdomain_log)
        except Exception as exc:
            logger.warning("Could not delete {}: {}", subdomain_log, exc)
            failures += 1

    return CleanupSummary(
        project_dir=project_dir,
        patterns=pats,
        files_deleted=deleted,
        bytes_freed=bytes_freed,
        attempted=len(files),
        failures=failures,
    )


def cli_main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="oa-da-clean-project",
        description="Remove model state pickle files for one project (or all projects) to free disk space.",
    )
    p.add_argument("--setup-dir", required=True, type=Path, help="Setup directory containing projects/")
    p.add_argument("--project-dir", type=Path, help="Project directory to clean (e.g., /data/projects/project_2022_2023)")
    p.add_argument("--all-projects", action="store_true", help="Clean all projects under setup/projects")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)

    configure_cli_logger(str(args.log_level or "INFO"))

    if not args.all_projects and args.project_dir is None:
        p.error("Provide --project-dir or --all-projects")
    if args.all_projects and args.project_dir is not None:
        p.error("Use either --project-dir or --all-projects, not both")

    setup_dir = Path(args.setup_dir)
    projects: List[Path]
    if args.all_projects:
        projects = _list_project_dirs(setup_dir)
        if not projects:
            logger.error("No project directories found under {}/projects", setup_dir)
            return 1
    else:
        projects = [Path(args.project_dir)]

    total_files = 0
    total_bytes = 0
    for project in projects:
        try:
            summary = cleanup_setup_dir(setup_dir=project, patterns=None)
        except Exception as exc:
            logger.error("Cleanup failed for {}: {}", project, exc)
            return 1
        total_files += summary.files_deleted
        total_bytes += summary.bytes_freed
        patterns_str = ",".join(summary.patterns)
        if summary.attempted == 0:
            logger.info("Cleaned {} -> no matching state files found (patterns={})", project, patterns_str)
        elif summary.failures:
            logger.warning(
                "Cleaned {} -> deleted {}/{} file(s), {} failure(s), freed {:.1f} MB (patterns={})",
                project,
                summary.files_deleted,
                summary.attempted,
                summary.failures,
                summary.bytes_freed / 1_000_000.0,
                patterns_str,
            )
        else:
            logger.info(
                "Cleaned {} -> deleted {}/{} file(s), freed {:.1f} MB (patterns={})",
                project,
                summary.files_deleted,
                summary.attempted,
                summary.bytes_freed / 1_000_000.0,
                patterns_str,
            )

    logger.info(
        "Cleanup complete | projects={} files={} freed={:.1f} MB",
        len(projects),
        total_files,
        total_bytes / 1_000_000.0,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

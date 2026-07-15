"""Preview-first cleanup of package-owned single-domain restart artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from openamundsen_da.core.constants import (
    DA_BLOCK,
    RESTART_BLOCK,
    RESTART_STATE_PATTERN,
    STATE_DEFAULT_NAME,
)
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml
from openamundsen_da.results import CleanupFailure, CleanupResult, WorkflowStatus


def _read_restart_config(project_dir: Path) -> dict:
    """Return restart config dict from project YAML (best-effort)."""
    try:
        project_yaml = find_project_yaml(project_dir)
        cfg = _read_yaml_file(project_yaml) or {}
        da_cfg = cfg.get(DA_BLOCK) or {}
        return da_cfg.get(RESTART_BLOCK) or {}
    except Exception:
        return {}


def state_patterns_from_setup(project_dir: Path) -> list[str]:
    """Return configured and default state filename patterns."""
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


def _single_domain_cleanup_candidates(project_dir: Path, patterns: Sequence[str]) -> list[Path]:
    """Return owned restart artifacts below top-level project steps only."""
    project_dir = Path(project_dir).resolve()
    steps_dir = project_dir / "steps"
    if not steps_dir.is_dir():
        return []
    candidates: dict[str, Path] = {}
    for pattern in patterns:
        for path in steps_dir.glob(f"step_*/ensembles/*/*/results/{pattern}"):
            if path.is_file() and not path.is_symlink():
                resolved = path.resolve()
                resolved.relative_to(project_dir)
                candidates[resolved.as_posix()] = resolved
    state_candidates = set(candidates.values())
    pointer_patterns = (
        "step_*/ensembles/*/*/state_pointer.json",
        "step_*/ensembles/*/*/results/state_pointer.json",
    )
    for pointer in (path for pattern in pointer_patterns for path in steps_dir.glob(pattern)):
        if not pointer.is_file() or pointer.is_symlink():
            continue
        try:
            import json

            data = json.loads(pointer.read_text(encoding="utf-8"))
            target_raw = data.get("path") if isinstance(data, dict) else None
            target = (pointer.parent / str(target_raw)).resolve() if target_raw else None
        except Exception:
            target = None
        if target is None or not target.is_file() or target in state_candidates:
            resolved = pointer.resolve()
            resolved.relative_to(project_dir)
            candidates[resolved.as_posix()] = resolved
    return [candidates[key] for key in sorted(candidates)]


def clean_project_artifacts(project_dir: Path, *, apply: bool) -> CleanupResult:
    """Preview or delete safe single-domain restart artifacts."""
    project_dir = Path(project_dir).resolve()
    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    candidates = _single_domain_cleanup_candidates(project_dir, state_patterns_from_setup(project_dir))
    sizes: dict[Path, int] = {}
    for path in candidates:
        try:
            sizes[path] = path.stat().st_size
        except OSError:
            sizes[path] = 0
    if not apply:
        return CleanupResult(
            project_dir=project_dir,
            status=WorkflowStatus.PREVIEW,
            applied=False,
            eligible_paths=tuple(candidates),
            deleted_paths=(),
            failures=(),
            eligible_bytes=sum(sizes.values()),
            freed_bytes=0,
        )

    deleted: list[Path] = []
    failures: list[CleanupFailure] = []
    freed = 0
    for path in candidates:
        try:
            path.unlink()
            deleted.append(path)
            freed += sizes[path]
        except OSError as exc:
            failures.append(CleanupFailure(path=path, error=str(exc)))
    return CleanupResult(
        project_dir=project_dir,
        status=WorkflowStatus.APPLIED,
        applied=True,
        eligible_paths=tuple(candidates),
        deleted_paths=tuple(deleted),
        failures=tuple(failures),
        eligible_bytes=sum(sizes.values()),
        freed_bytes=freed,
    )

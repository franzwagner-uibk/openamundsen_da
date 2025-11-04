from __future__ import annotations

import argparse
import concurrent.futures as cf
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    # For type checking only; avoids importing heavy dependencies at runtime
    from openamundsen_da.core.runner import RunResult
import sys

from loguru import logger
import ruamel.yaml
from openamundsen_da.core.constants import ENV_VARS_EXPORT, ENVIRONMENT

from openamundsen_da.io.paths import (
    find_project_yaml,
    find_season_yaml,
    find_step_yaml,
    list_member_dirs,
)

# Do NOT import anything that pulls GDAL here. runner is imported later inside the worker.


def _clamp_workers(n: int) -> int:
    # keep some headroom on Windows; OA uses NumPy/numexpr underneath
    return max(1, min(n, os.cpu_count() or 1))


_yaml = ruamel.yaml.YAML(typ="safe")


def _read_yaml_file(p: Path) -> dict:
    try:
        with Path(p).open("r", encoding="utf-8") as f:
            return _yaml.load(f) or {}
    except Exception:
        return {}


def _apply_env_from_project(project_yaml: Path) -> None:
    """
    Read project.yml with openAMUNDSEN's reader and export 'environment' keys.
    Expected structure, e.g.:
      environment:
        GDAL_DATA: "C:/.../Library/share/gdal"
        PROJ_LIB:  "C:/.../Library/share/proj"
        OMP_NUM_THREADS: "1"
    """
    try:
        cfg = _read_yaml_file(project_yaml)
    except Exception as e:
        logger.warning(f"Could not read project YAML to set environment ({project_yaml}): {e}")
        cfg = {}

    env_cfg = (cfg or {}).get(ENVIRONMENT) or {}

    # Only set if specified in YAML
    for k in ENV_VARS_EXPORT:
        v = env_cfg.get(k)
        if v:
            os.environ[k] = str(v)

    # Reasonable fallbacks if still not set (common for conda on Windows)
    # <env>\Library\share\{gdal,proj}
    conda = os.environ.get("CONDA_PREFIX") or os.environ.get("PREFIX")
    if conda:
        gdal_default = str(Path(conda) / "Library" / "share" / "gdal")
        proj_default = str(Path(conda) / "Library" / "share" / "proj")
        os.environ.setdefault("GDAL_DATA", gdal_default)
        os.environ.setdefault("PROJ_LIB", proj_default)

    # Numeric libs -> one thread per process (can be overridden in YAML)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    logger.debug(
        "Env set -> GDAL_DATA='{}', PROJ_LIB='{}', OMP_NUM_THREADS='{}'",
        os.environ.get("GDAL_DATA"),
        os.environ.get("PROJ_LIB"),
        os.environ.get("OMP_NUM_THREADS"),
    )


def _discover_members(
    project_dir: Path, season_dir: Path, step_dir: Path, ensemble: str
) -> Tuple[Path, Path, Path, List[Path]]:
    """
    Resolve YAMLs and member directories using our path helpers.
    """
    proj_yaml = find_project_yaml(project_dir)
    seas_yaml = find_season_yaml(season_dir)
    step_yaml = find_step_yaml(step_dir)

    member_root = step_dir / "ensembles"
    members = list_member_dirs(member_root, ensemble=ensemble)
    if not members:
        raise RuntimeError(
            f"No members found for ensemble='{ensemble}' under {member_root / ensemble}"
        )

    logger.debug(
        "Using: project={}, season={}, step={}",
        str(proj_yaml),
        str(seas_yaml),
        str(step_yaml),
    )
    logger.info(f"Discovered {len(members)} {ensemble} member(s)")
    return proj_yaml, seas_yaml, step_yaml, members


def _run_one(args: Tuple[Path, Path, Path, Path, bool, Path | None, str | None]) -> RunResult:
    """
    Small wrapper so ProcessPoolExecutor can pickle the callable easily.
    Import of runner happens inside the child worker.
    """
    proj_yaml, seas_yaml, step_yaml, member_dir, overwrite, results_root, log_level = args

    # Local import inside the worker to avoid importing GDAL users in the parent
    from openamundsen_da.core.runner import run_member

    # If a global results_root was provided, derive a per-member results_dir from it;
    # otherwise let runner compute its default (member_dir/results/)
    results_dir = None
    if results_root is not None:
        # default helper for per-member results layout
        results_dir = Path(results_root) / Path(member_dir).name

    # The runner will:
    #  - build the merged config (using openAMUNDSEN parse_config)
    #  - inject per-member meteo dir & results dir
    #  - run OA initialize+run
    return run_member(
        project_dir=proj_yaml.parent,
        season_dir=seas_yaml.parent,
        step_dir=step_yaml.parent,
        member_dir=member_dir,
        results_dir=results_dir,
        overwrite=overwrite,
        log_level=log_level,
    )


def launch_members(
    project_dir: Path,
    season_dir: Path,
    step_dir: Path,
    ensemble: str,
    max_workers: int,
    overwrite: bool,
    results_root: Path | None,
    *,
    log_level: str | None,
) -> dict:
    proj_yaml, seas_yaml, step_yaml, members = _discover_members(project_dir, season_dir, step_dir, ensemble)

    # Make sure GDAL/PROJ & threading env are exported before workers spawn
    _apply_env_from_project(proj_yaml)

    # Use spawn on Windows explicitly to be safe
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # already set in this interpreter, ignore
        pass

    # Fan out
    tasks = [
        (proj_yaml, seas_yaml, step_yaml, m, overwrite, results_root, log_level)
        for m in members
    ]

    workers = _clamp_workers(max_workers)
    logger.info(
        "Launching {n} member(s) with max_workers={mw}",
        n=len(tasks),
        mw=workers,
    )
    results: List[RunResult] = []
    failed = 0
    skipped = 0
    ok = 0

    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        # Log a start line as we submit each member to the pool
        fut_to_member = {}
        for t in tasks:
            m = t[3]
            logger.info(f"[{m.name}] starting")
            fut_to_member[ex.submit(_run_one, t)] = m
        for fut in cf.as_completed(fut_to_member):
            m = fut_to_member[fut]
            try:
                res = fut.result()
                results.append(res)
                if res.status == "success":
                    ok += 1
                    logger.info(f"[{res.member_name}] finished: success ({res.duration_seconds:.1f}s)")
                elif res.status == "skipped":
                    skipped += 1
                    logger.info(f"[{res.member_name}] skipped")
                else:
                    failed += 1
                    logger.error(f"[{res.member_name}] finished: failed ({res.error})")
            except Exception as e:
                failed += 1
                logger.error(f"[{m.name}] failed: {e!r}")

    summary = {"total": len(members), "ok": ok, "skipped": skipped, "failed": failed}
    logger.info("Summary: total={total}  ok={ok}  skipped={skipped}  failed={failed}", **summary)
    return {"summary": summary, "results": results}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch openAMUNDSEN ensemble members")
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--season-dir", required=True, type=Path)
    p.add_argument("--step-dir", required=True, type=Path)
    p.add_argument("--ensemble", required=True, choices=("prior", "posterior"))
    p.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Optional global results root; per-member results dirs will be placed under this",
    )
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logger.remove()
    logger.add(
        sys.stderr,
        level=args.log_level,
        enqueue=True,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | {message}",
    )

    try:
        launch_members(
            project_dir=args.project_dir,
            season_dir=args.season_dir,
            step_dir=args.step_dir,
            ensemble=args.ensemble,
            max_workers=args.max_workers,
            overwrite=args.overwrite,
            results_root=args.results_root,
            log_level=args.log_level,
        )
        return 0
    except Exception as e:
        logger.exception(e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

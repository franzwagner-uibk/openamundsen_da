"""Generic helpers for daily AOI scalar series per ensemble member.

These utilities factor out the common pattern used by SCF and wet snow:

- For a given step, read its start/end dates from the YAML.
- Discover prior/posterior members (and optionally open_loop).
- For each member, compute a daily AOI-aggregated series for a scalar
  variable and write a `point_*_aoi.csv` under the member results.
- Use a process pool for per-member parallelism and skip work when all
  outputs already exist (unless overwrite=True).

Observable-specific modules provide the actual per-member worker
function and call :func:`compute_step_daily_series_for_all_members`.
"""

from __future__ import annotations

import concurrent.futures as cf
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from loguru import logger

from openamundsen_da.core.constants import ENSEMBLE_PRIOR
from openamundsen_da.io.paths import list_member_dirs, open_loop_dir, read_step_config


DailySeriesWorker = Callable[
    [Path, Path, datetime, datetime, Path, bool, Dict[str, Any]],
    bool,
]


def step_start_end(step_dir: Path) -> Tuple[datetime, datetime]:
    """Return (start_date, end_date) for a step from its YAML config."""

    cfg = read_step_config(step_dir) or {}
    try:
        s_val = cfg.get("start_date")
        e_val = cfg.get("end_date")
        start = datetime.fromisoformat(str(s_val))
        end = datetime.fromisoformat(str(e_val))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Missing or invalid start_date/end_date in step config for {step_dir}") from exc
    return start, end


def _list_members_with_optional_open_loop(
    step_dir: Path,
    ensemble: str,
    include_open_loop: bool,
) -> List[Path]:
    """Return ordered member directories, optionally including open_loop first."""

    members = list_member_dirs(step_dir / "ensembles", ensemble)
    if not include_open_loop or ensemble != ENSEMBLE_PRIOR:
        return members

    try:
        ol_dir = open_loop_dir(step_dir)
        if ol_dir.is_dir():
            return [ol_dir] + members
    except Exception:
        return members
    return members


def compute_step_daily_series_for_all_members(
    *,
    step_dir: Path,
    aoi_path: Path,
    start: datetime,
    end: datetime,
    csv_name: str,
    worker: DailySeriesWorker,
    ensemble: str = ENSEMBLE_PRIOR,
    include_open_loop: bool = True,
    max_workers: int = 4,
    overwrite: bool = False,
    worker_kwargs: Dict[str, Any] | None = None,
) -> None:
    """Compute daily AOI series for all members in a step.

    Parameters
    ----------
    step_dir : Path
        Step directory (contains ensembles/).
    aoi_path : Path
        Single-feature AOI vector used by the observable-specific worker.
    start, end : datetime
        Date range for the daily series (inclusive, calendar days).
    csv_name : str
        Output filename under each member's results directory
        (e.g., ``point_scf_aoi.csv``).
    worker : callable
        Per-member worker function with signature
        ``(results_dir, aoi_path, start, end, out_csv, overwrite, extra)``.
    ensemble : {"prior","posterior"}, optional
        Ensemble name used below ``step_dir/ensembles``.
    include_open_loop : bool, optional
        When True and ``ensemble='prior'``, include the ``open_loop`` member
        if present, preceding the other members.
    max_workers : int, optional
        Max number of process workers used for per-member parallelism.
    overwrite : bool, optional
        When False, members whose ``csv_name`` already exists are skipped and
        if all members are up-to-date the function returns without work.
    worker_kwargs : dict, optional
        Extra keyword arguments passed to each worker invocation (must be
        picklable for use with :class:`concurrent.futures.ProcessPoolExecutor`).
    """

    step_dir = Path(step_dir)
    aoi_path = Path(aoi_path)
    extra = dict(worker_kwargs or {})

    members = _list_members_with_optional_open_loop(step_dir, ensemble, include_open_loop)
    if not members:
        logger.warning("No members found under {}/ensembles/{}; skipping daily AOI series.", step_dir, ensemble)
        return

    jobs: List[Tuple[Path, Path, datetime, datetime, Path, bool, Dict[str, Any]]] = []
    all_exist = True
    for mdir in members:
        res_dir = Path(mdir) / "results"
        if not res_dir.is_dir():
            all_exist = False
            continue
        out_csv = res_dir / csv_name
        if not out_csv.is_file():
            all_exist = False
        jobs.append((res_dir, aoi_path, start, end, out_csv, bool(overwrite), extra))

    if not jobs:
        logger.warning("No member results directories found for {}; skipping daily AOI series.", step_dir)
        return

    if all_exist and not overwrite:
        logger.info(
            "{} already present for all members in {}; overwrite=False -> skipping daily AOI series.",
            csv_name,
            step_dir.name,
        )
        return

    n_workers = max(1, min(int(max_workers or 1), len(jobs)))
    logger.info(
        "Computing daily AOI series ({}) for {} member(s) in {} using {} worker(s) ...",
        csv_name,
        len(jobs),
        step_dir.name,
        n_workers,
    )

    created = 0
    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(worker, *args) for args in jobs]
        for fut in cf.as_completed(futures):
            try:
                did_create = bool(fut.result())
                if did_create:
                    created += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Daily AOI series worker failed for a member in {}: {}", step_dir, exc)

    logger.info(
        "Daily AOI series ({}) written for {} / {} member(s) in {}",
        csv_name,
        created,
        len(jobs),
        step_dir.name,
    )


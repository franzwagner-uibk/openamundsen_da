"""Utilities for consistent parallel settings (worker limits, seeds).

This module keeps worker clamping and base seed resolution in one place so
stages can share the same defaults:
- MAX_WORKERS from the environment acts as a shared override.
- Worker counts are capped by CPU count and optional job limits.
- OA_BASE_SEED can override per-command seeds for reproducibility.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import secrets
from typing import Optional


def _parse_positive_int(val: object) -> Optional[int]:
    """Return a positive int if ``val`` can be parsed, otherwise None."""
    try:
        n = int(val)  # type: ignore[arg-type]
    except Exception:
        return None
    return n if n > 0 else None


def env_max_workers() -> Optional[int]:
    """Read MAX_WORKERS from the environment (positive int only)."""
    return _parse_positive_int(os.environ.get("MAX_WORKERS"))


def pick_max_workers(
    requested: int | None = None,
    fallback: int | None = None,
    *,
    limit: int | None = None,
) -> int:
    """Return an effective worker count using env/CLI/default caps.

    Precedence:
    1) requested (CLI or explicit call)
    2) MAX_WORKERS env
    3) fallback (code default) or CPU count

    The result is clamped by CPU count and an optional ``limit`` (e.g., number
    of jobs). Always returns at least 1.
    """
    cpu = _parse_positive_int(os.cpu_count()) or 1
    env_val = env_max_workers()

    base = _parse_positive_int(requested)
    if base is None:
        base = env_val
    if base is None:
        base = _parse_positive_int(fallback) or cpu

    caps = [cpu]
    limit_val = _parse_positive_int(limit)
    if limit_val is not None:
        caps.append(limit_val)
    workers = min([base, *caps])
    return max(1, workers)


def resolve_base_seed(seed: int | None = None, env_var: str = "OA_BASE_SEED") -> int:
    """Return a base seed, preferring env ``env_var`` when set.

    Falls back to the provided ``seed``; if neither is available, generates a
    random 32-bit seed.
    """
    env_seed = _parse_positive_int(os.environ.get(env_var))
    if env_seed is not None:
        return env_seed
    cfg_seed = _parse_positive_int(seed)
    if cfg_seed is not None:
        return cfg_seed
    return secrets.randbits(32)


def run_tasks_with_pool(
    func,
    tasks: list[tuple],
    *,
    max_workers: int | None = None,
    fallback_workers: int | None = None,
    label: str = "tasks",
    unpack: bool = True,
):
    """Run tasks in a process pool, falling back to serial when workers=1.

    Each item in ``tasks`` is passed as positional args to ``func``.
    Returns a list of results in completion order.
    """
    workers = pick_max_workers(max_workers, fallback=fallback_workers or len(tasks), limit=len(tasks))
    if workers <= 1:
        return [func(*t) if unpack else func(t) for t in tasks]

    results = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        future_to_task = {ex.submit(func, *t) if unpack else ex.submit(func, t): t for t in tasks}
        for fut in cf.as_completed(future_to_task):
            results.append(fut.result())
    return results

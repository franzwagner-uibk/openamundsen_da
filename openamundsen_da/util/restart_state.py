"""Validation helpers for restart checkpoints owned by openAMUNDSEN-DA."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path


def validate_restart_state(path: str | Path) -> Path:
    """Require a readable gzip-pickle state mapping and return its path."""
    path = Path(path)
    if not path.is_file():
        raise RuntimeError(f"Required restart state is missing: {path}")
    try:
        with gzip.open(path, "rb") as stream:
            state = pickle.load(stream)
    except Exception as exc:
        raise RuntimeError(f"Restart state is unreadable: {path}: {exc}") from exc
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Restart state must contain a non-empty mapping: {path}")
    if not all(isinstance(values, dict) for values in state.values()):
        raise RuntimeError(f"Restart state categories must be mappings: {path}")
    return path


__all__ = ["validate_restart_state"]

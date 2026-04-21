"""Shared helpers for ensemble member metadata in visualization modules."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from openamundsen_da.io.paths import list_member_dirs
from openamundsen_da.methods.viz.station_meta import (
    load_ensemble_station_table,
    load_ensemble_station_table_from_steps,
)


def load_stations_table(step_dir: Path, ensemble: str) -> Optional[pd.DataFrame]:
    """Load stations.csv from open_loop or first member meteo dir if available."""
    return load_ensemble_station_table(step_dir, ensemble)


def load_stations_table_from_steps(step_dirs: Sequence[Path], ensemble: str = "prior") -> Optional[pd.DataFrame]:
    """Load stations.csv from the first step that provides it."""
    return load_ensemble_station_table_from_steps(step_dirs, ensemble)


def read_member_perturbations(step_dir: Path, ensemble: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return mapping member_name -> (delta_T, f_p) parsed from INFO.txt if present."""
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    members = list_member_dirs(Path(step_dir) / "ensembles", ensemble)
    for member in members:
        info = member / "INFO.txt"
        delta_t: Optional[float] = None
        f_p: Optional[float] = None
        if info.is_file():
            try:
                text = info.read_text(encoding="utf-8", errors="ignore")
                m_dt = re.search(r"delta_T\s*\(additive\)\s*:\s*([+-]?\d+\.?\d*)", text, re.IGNORECASE)
                m_fp = re.search(r"precip factor\s*f_p\s*:\s*([+-]?\d+\.?\d*)", text, re.IGNORECASE)
                if m_dt:
                    delta_t = float(m_dt.group(1))
                if m_fp:
                    f_p = float(m_fp.group(1))
            except Exception:
                pass
        out[member.name] = (delta_t, f_p)
    return out


def format_member_label(member_name: str, pert: Tuple[Optional[float], Optional[float]]) -> str:
    """Format member label with optional perturbation info."""
    delta_t, f_p = pert
    if delta_t is None and f_p is None:
        return member_name
    parts = []
    if delta_t is not None:
        parts.append(f"dT={delta_t:+.2f}")
    if f_p is not None:
        parts.append(f"f_p={f_p:.2f}")
    return f"{member_name} ({', '.join(parts)})"

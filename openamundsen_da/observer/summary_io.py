"""Shared summary CSV readers for observation preprocessing and plotting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_fraction_summary(csv_path: Path, *, value_col: str) -> pd.DataFrame:
    """Load and normalize a summary CSV with ``date`` and one value column."""
    df = pd.read_csv(csv_path)
    if "date" not in df or value_col not in df:
        raise ValueError(f"CSV must contain columns 'date' and '{value_col}'")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")


def load_scf_summary(csv_path: Path) -> pd.DataFrame:
    return load_fraction_summary(csv_path, value_col="scf")


__all__ = ["load_fraction_summary", "load_scf_summary"]

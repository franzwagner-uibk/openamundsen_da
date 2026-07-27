"""
aggregate_fractions.py
Author: openamundsen_da
Date: 2025-12-02
Description:
    Aggregate member-level fraction time series into a setup-wide envelope.

    Looks for member CSVs (e.g., point_scf_roi.csv or point_wet_snow_roi.csv)
    under <setup>/steps/step_*/ensembles/prior/*/results and writes an envelope CSV
    with date, value_mean, value_min, value_max, n. The envelope bounds are
    the finite-member minimum and maximum.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.io.paths import list_step_dirs
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.methods.pf.weights import load_prior_weights
from openamundsen_da.util.stats import (
    effective_sample_size,
    normalize_weights,
    weighted_mean,
    weighted_quantile,
    weighted_std,
)


def _find_series_files(setup_dir: Path, filename: str) -> list[Path]:
    """Return all matching member result CSVs for a setup."""
    files: list[Path] = []
    for step_dir in list_step_dirs(setup_dir):
        pattern = "ensembles/prior/*/results/" + filename
        files.extend(p for p in step_dir.glob(pattern) if p.is_file() and p.parent.parent.name != "open_loop")
    return sorted(files)


def _load_series(path: Path, value_col: str) -> pd.DataFrame | None:
    """Read one member CSV and return normalized date/value columns."""
    df = pd.read_csv(path)
    if df.empty or value_col not in df.columns:
        return None
    for col in ("date", "time", "datetime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.rename(columns={col: "date"})
            break
    else:
        return None
    out = df[["date", value_col]].dropna()
    if out.empty:
        return None
    out["date"] = out["date"].dt.normalize()
    return out


def aggregate_fraction_envelope(
    *,
    setup_dir: Path,
    filename: str,
    value_col: str,
    output_name: str,
) -> Path | None:
    """
    Aggregate per-member fraction CSVs into a setup-level envelope CSV.

    Parameters
    ----------
    setup_dir : Path
        Setup directory containing steps/step_* folders.
    filename : str
        Member CSV filename to collect (e.g., point_scf_roi.csv).
    value_col : str
        Value column to aggregate (e.g., scf or wet_snow_fraction).
    output_name : str
        Output CSV name written into the setup directory.

    Returns
    -------
    Path or None
        Path to the written envelope CSV, or None if no inputs found.
    """
    files = _find_series_files(setup_dir, filename)
    if not files:
        logger.warning("No member series found for {} under {}", filename, setup_dir)
        return None

    frames: list[pd.DataFrame] = []
    ledgers: dict[Path, pd.DataFrame] = {}
    for f in files:
        try:
            df = _load_series(f, value_col)
        except Exception as exc:
            logger.warning("Skipping {}: {}", f, exc)
            continue
        if df is not None and not df.empty:
            step_dir = f.parents[4]
            member_id = f.parents[1].name
            if step_dir not in ledgers:
                step_member_ids = sorted(
                    path.name
                    for path in (step_dir / "ensembles" / "prior").glob("member_*")
                    if path.is_dir()
                )
                ledgers[step_dir] = load_prior_weights(step_dir, step_member_ids).set_index("member_id")
            ledger = ledgers[step_dir]
            if member_id not in ledger.index:
                raise ValueError(f"Missing PF prior weight for {member_id} in {step_dir}")
            df["member_id"] = member_id
            df["pf_weight"] = float(ledger.loc[member_id, "weight"])
            frames.append(df)

    if not frames:
        logger.warning("No usable rows found for {} under {}", filename, setup_dir)
        return None

    all_df = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for date, group in all_df.groupby("date", sort=True):
        values = pd.to_numeric(group[value_col], errors="coerce").to_numpy(dtype=float)
        raw_weights = pd.to_numeric(group["pf_weight"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(raw_weights) & (raw_weights >= 0.0)
        if not np.any(valid):
            continue
        values = values[valid]
        weights = normalize_weights(raw_weights[valid])
        rows.append(
            {
                "date": date,
                "value_mean": weighted_mean(values, weights),
                "value_std": weighted_std(values, weights),
                "value_q05": weighted_quantile(values, 0.05, weights),
                "value_q95": weighted_quantile(values, 0.95, weights),
                "value_min": float(np.min(values)),
                "value_max": float(np.max(values)),
                "n": int(values.size),
                "ess": effective_sample_size(weights),
                "weighting": "pf_prior_ledger",
                "bounds_semantics": "materialized_member_range",
            }
        )
    out = pd.DataFrame(rows).sort_values("date")

    out_path = Path(setup_dir) / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Wrote envelope CSV ({} rows) -> {}", len(out), out_path)
    return out_path


def cli_main(argv: list[str] | None = None) -> int:
    """CLI to aggregate SCF or wet-snow envelopes."""
    import argparse

    p = argparse.ArgumentParser(
        prog="oa-da-aggregate-fractions",
        description="Aggregate member fraction time series into a setup-level envelope CSV.",
    )
    p.add_argument("--setup-dir", required=True, type=Path, help="Setup directory (projects/setup_YYYY-YYYY)")
    p.add_argument("--filename", required=True, help="Member CSV filename to collect (e.g., point_scf_roi.csv)")
    p.add_argument("--value-col", required=True, help="Value column to aggregate (e.g., scf)")
    p.add_argument("--output-name", required=True, help="Output CSV name (e.g., point_scf_roi_envelope.csv)")
    p.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = p.parse_args(argv)

    configure_cli_logger(args.log_level)

    try:
        aggregate_fraction_envelope(
            setup_dir=Path(args.setup_dir),
            filename=str(args.filename),
            value_col=str(args.value_col),
            output_name=str(args.output_name),
        )
    except Exception as exc:
        logger.error("Envelope aggregation failed: {}", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

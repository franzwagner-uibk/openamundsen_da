"""openamundsen_da.methods.pf.plot_weights

Plot per-date SCF assimilation weights and residuals.

Inputs
- weights CSV produced by oa-da-assimilate-scf with columns:
  member_id, scf_model, scf_obs, residual, sigma, log_weight, weight

Outputs
- A PNG saved next to the CSV (or --output) with two panels:
  A) sorted normalized weights with ESS annotation
  B) residual histogram with sigma annotated

Logging uses LOGURU_FORMAT from core.constants.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from loguru import logger
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_season_yaml
from openamundsen_da.util.stats import effective_sample_size


def _load_weights(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"weight", "residual"}
    if not needed.issubset(df.columns):
        missing = ", ".join(sorted(needed - set(df.columns)))
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def _plot(df: pd.DataFrame, title: str, subtitle: str | None, *, backend: str = "Agg"):
    import matplotlib
    matplotlib.use(backend or "Agg")
    import matplotlib.pyplot as plt

    w = np.asarray(df["weight"], dtype=float)
    resid = np.asarray(df["residual"], dtype=float)
    sigma = float(df.get("sigma", pd.Series([np.nan])).iloc[0])
    n = w.size
    ess = effective_sample_size(w)

    fig = plt.figure(figsize=(10, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 2])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    # Panel A: sorted weights
    ws = np.sort(w)[::-1]
    ax0.plot(np.arange(1, n + 1), ws, color="#1f77b4", lw=1.8)
    ax0.set_xlabel("member (sorted)")
    ax0.set_ylabel("weight")
    ax0.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax0.set_title(f"weights (ESS={ess:.1f} / N={n})", fontsize=10)
    ax0.set_ylim(0.0, 1.0)

    # Panel B: residual distribution (line histogram to avoid Windows patch crashes)
    resid_finite = resid[np.isfinite(resid)]
    if resid_finite.size > 0:
        bins = min(20, max(3, resid_finite.size // 2))
        counts, edges = np.histogram(resid_finite, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax1.plot(centers, counts, color="#ff7f0e", lw=1.8, marker="o", alpha=0.85)
    ax1.set_xlabel("residual = scf_obs - scf_model")
    ax1.set_ylabel("count")
    if np.isfinite(sigma):
        ax1.axvline(0.0, color="k", lw=1.0)
        ax1.axvline(-sigma, color="#999999", lw=1.0, ls="--", label=f"sigma={sigma:.3f}")
        ax1.axvline(sigma, color="#999999", lw=1.0, ls="--")
        ax1.legend(loc="best", frameon=False, fontsize=8)
    ax1.grid(True, ls=":", lw=0.6, alpha=0.7)

    top_rect = 0.90 if (title or subtitle) else 0.94
    fig.tight_layout(rect=[0.02, 0.04, 0.98, top_rect])
    if title:
        fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=12)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=10, color="#555555")
    return fig


def _default_output_path(csv_path: Path) -> Path:
    """Return default output PNG path for a weights CSV.

    If the CSV lives under <season>/step_XX_*/assim/, write to
    <season>/plots/assim/weights/step_XX_weights.png. Otherwise, fall back
    to csv_path.with_suffix('.png').
    """
    csv_path = csv_path.resolve()
    # Expect .../season_YYYY-YYYY/step_XX_*/assim/weights_scf_YYYYMMDD.csv
    if csv_path.parent.name == "assim":
        step_dir = csv_path.parent.parent
        season_dir = step_dir.parent
        if step_dir.name.startswith("step_") and (season_dir / "season.yml").is_file():
            parts = step_dir.name.split("_")
            step_token = "_".join(parts[:2]) if len(parts) >= 2 else step_dir.name
            out_dir = season_dir / "plots" / "assim" / "weights"
            out_dir.mkdir(parents=True, exist_ok=True)
            return out_dir / f"{step_token}_weights.png"
    # Fallback: same dir as CSV
    return csv_path.with_suffix(".png")


def _step_date_label_from_path(csv_path: Path) -> str | None:
    """Return "DA# - YYYY-MM-DD" (or fallback step label) inferred from the CSV path."""
    try:
        csv_path = csv_path.resolve()
    except Exception:
        return None

    stem = csv_path.stem
    prefix = "weights_scf_"
    if not stem.startswith(prefix):
        return None
    ds = stem[len(prefix) :]
    if len(ds) != 8 or not ds.isdigit():
        return None
    date_str = f"{ds[0:4]}-{ds[4:6]}-{ds[6:8]}"
    try:
        date_val = pd.to_datetime(date_str).date()
    except Exception:
        return None

    if csv_path.parent.name == "assim":
        step_dir = csv_path.parent.parent
        season_dir = step_dir.parent
        try:
            seas_yaml = find_season_yaml(season_dir)
            cfg = _read_yaml_file(seas_yaml) or {}
            assim_dates = cfg.get("assimilation_dates") or []
            for idx, entry in enumerate(assim_dates, start=1):
                try:
                    cand_date = pd.to_datetime(entry).date()
                except Exception:
                    continue
                if cand_date == date_val:
                    return f"DA{idx} - {date_str}"
        except Exception:
            pass

        name = step_dir.name
        if name.startswith("step_"):
            tail = name[len("step_") :]
            token = tail.split("_", 1)[0] if tail else ""
            if token:
                label = f"Step {token}"
            else:
                label = name
            return f"{label} - {date_str}"
    return None
def plot_weights_for_csv(
    csv_path: Path,
    *,
    title: str = "SCF Assimilation Weights",
    subtitle: str | None = None,
    backend: str = "Agg",
) -> Path:
    """Library API: plot weights for a single CSV and return PNG path."""
    df = _load_weights(csv_path)
    # If caller uses the default title and no subtitle, derive a compact
    # label from the path: "Step <number> - <YYYY-MM-DD>".
    if title == "SCF Assimilation Weights" and subtitle is None:
        label = _step_date_label_from_path(csv_path)
        if label:
            subtitle = label
    fig = _plot(df, title=title, subtitle=subtitle, backend=backend)
    out = _default_output_path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.1)
    return out


def cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oa-da-plot-weights", description="Plot SCF assimilation weights and residuals")
    p.add_argument("csv", type=Path, help="Path to weights_scf_YYYYMMDD.csv")
    p.add_argument("--output", type=Path, help="Output PNG path (default: same dir as CSV)")
    p.add_argument("--title", default="SCF Assimilation Weights", help="Plot title")
    p.add_argument("--subtitle", default="", help="Plot subtitle")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--backend", default="Agg", help="Matplotlib backend (Agg, SVG, module://mplcairo.Agg)")
    args = p.parse_args(argv)

    from openamundsen_da.core.constants import LOGURU_FORMAT
    logger.remove()
    # Avoid enqueue for short-lived CLIs so messages flush before exit
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=False, format=LOGURU_FORMAT)

    csv_path = Path(args.csv)
    logger.info("Reading weights CSV: {}", csv_path)
    try:
        df = _load_weights(csv_path)
    except Exception as e:
        logger.error(f"Failed reading weights CSV: {e}")
        return 1

    # Basic stats
    try:
        n = len(df)
        w = np.asarray(df["weight"], dtype=float)
        ess = effective_sample_size(w)
        sigma = df.get("sigma", pd.Series([np.nan])).iloc[0]
        logger.info("Rows={}  ESS={:.1f}  N={}  sigma={}", n, ess, w.size, (f"{sigma:.3f}" if pd.notna(sigma) else "NA"))
    except Exception:
        pass

    try:
        # Automatically derive a compact "Step <number> - <YYYY-MM-DD>" label
        # when the caller uses the default title and no explicit subtitle.
        subtitle = (args.subtitle or None)
        if args.title == "SCF Assimilation Weights" and not subtitle:
            label = _step_date_label_from_path(csv_path)
            if label:
                subtitle = label
        fig = _plot(df, title=args.title, subtitle=subtitle, backend=args.backend)
    except ModuleNotFoundError:
        logger.error("matplotlib is required to plot. Install it in your environment.")
        return 2
    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        return 3

    out = Path(args.output) if args.output else _default_output_path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving plot to: {}", out)
    try:
        fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.1)
    except Exception as e:
        logger.error(f"Saving PNG failed: {e}")
        return 4
    logger.info("Wrote plot: {}", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

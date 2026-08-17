"""Simple diagnostic plots for station assimilation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from openamundsen_da.util.figure_lifecycle import close_created_figures
from openamundsen_da.methods.viz.common import (
    force_figure_text_black,
    save_figure_png,
    set_matplotlib_text_black,
)
from openamundsen_da.methods.viz.plots.theme import COLOR_DA_OBS, LS_STATION_OBS


def _load_diagnostics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"station_id", "member_id", "obs_value", "model_value", "sigma"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(missing)}")
    return df


def _default_output_path(csv_path: Path) -> Path:
    return Path(csv_path).with_suffix(".png")


@close_created_figures
def plot_station_diagnostics_for_csv(
    csv_path: Path,
    *,
    title: str = "Station Assimilation Diagnostics",
    backend: str = "Agg",
) -> Path:
    """Plot per-station member values against observed value and sigma band."""
    import matplotlib

    matplotlib.use(backend or "Agg")
    set_matplotlib_text_black(matplotlib)
    import matplotlib.pyplot as plt

    df = _load_diagnostics(csv_path)
    stations = sorted(df["station_id"].astype(str).unique())
    if not stations:
        raise ValueError(f"No station rows found in {csv_path}")

    fig_height = max(3.2, 2.6 * len(stations))
    fig, axes = plt.subplots(len(stations), 1, figsize=(10.5, fig_height), sharex=False, squeeze=False)
    axes = axes.ravel()

    var_name = str(df["variable"].iloc[0]) if "variable" in df.columns and not df.empty else ""
    date_text = str(df["date"].iloc[0]) if "date" in df.columns and not df.empty else ""

    for ax, station_id in zip(axes, stations):
        sub = df[df["station_id"].astype(str) == station_id].copy()
        if "final_weight" in sub.columns:
            sub = sub.sort_values(["final_weight", "member_id"], ascending=[False, True])
            weights = pd.to_numeric(sub["final_weight"], errors="coerce").to_numpy(dtype=float)
        else:
            sub = sub.sort_values("member_id")
            weights = np.full(len(sub), np.nan, dtype=float)

        x = np.arange(len(sub), dtype=float)
        obs_value = float(pd.to_numeric(sub["obs_value"], errors="coerce").iloc[0])
        sigma = float(pd.to_numeric(sub["sigma"], errors="coerce").iloc[0])
        model_values = pd.to_numeric(sub["model_value"], errors="coerce").to_numpy(dtype=float)

        ax.axhspan(obs_value - sigma, obs_value + sigma, color=COLOR_DA_OBS, alpha=0.35, label="obs +/- sigma")
        ax.axhline(obs_value, color=COLOR_DA_OBS, lw=1.6, ls=LS_STATION_OBS, label="observation")

        if np.isfinite(weights).any():
            w_norm = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            if np.nanmax(w_norm) > 0.0:
                w_norm = w_norm / np.nanmax(w_norm)
            colors = plt.cm.viridis(0.25 + 0.75 * w_norm)
        else:
            colors = "#482475"
        ax.scatter(x, model_values, s=32, c=colors, edgecolors="black", linewidths=0.3, zorder=3)

        ax.set_ylabel(station_id)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_xlim(-0.5, max(len(sub) - 0.5, 0.5))
        ax.set_xticks(x)
        ax.set_xticklabels(sub["member_id"].astype(str), rotation=45, ha="right", fontsize=8)

    axes[-1].set_xlabel("member (sorted by final weight when available)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="best", frameon=False, fontsize=8)

    top_rect = 0.90 if title else 0.94
    fig.tight_layout(rect=[0.02, 0.04, 0.98, top_rect])
    if title:
        suffix = f" | {var_name} | {date_text}" if var_name or date_text else ""
        fig.text(0.5, 0.97, f"{title}{suffix}", ha="center", va="top", fontsize=12)

    out = _default_output_path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    force_figure_text_black(fig, axes)
    save_figure_png(fig, out, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out

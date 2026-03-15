"""Plot SCF and wet-snow fraction time series for one project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from openamundsen_da.methods.viz.fraction_series import (
    default_fraction_obs_path,
    default_fraction_plot_output,
    load_fraction_series,
    load_open_loop_fraction_series,
)
from openamundsen_da.observer.plot_scf_summary import _load_summary as _load_scf_obs
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.methods.viz._utils import (
    draw_assimilation_markers,
    draw_assimilation_vlines,
    dedupe_legend,
    apply_fraction_grid,
    draw_assim_labels,
)
from openamundsen_da.methods.viz._style import (
    COLOR_DA_OBS,
    SIZE_DA_OBS,
    LW_DA_OBS,
    COLOR_DA_STATION_HS,
    COLOR_DA_STATION_SWE,
    LS_DA_STATION_HS,
    LS_DA_STATION_SWE,
    LW_DA_STATION,
)
from openamundsen_da.util.loguru_utils import configure_cli_logger


def _load_scf_obs_series(path: Path) -> pd.DataFrame | None:
    """Load SCF summary data, falling back to a generic fraction-series reader."""
    try:
        return _load_scf_obs(path)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.debug("Falling back to generic SCF summary reader for {}: {}", path, exc)
        return load_fraction_series(path, "scf")


def plot_fraction_timeseries(
    *,
    scf_obs: Optional[pd.DataFrame],
    scf_model: Optional[pd.DataFrame],
    wet_obs: Optional[pd.DataFrame],
    wet_model: Optional[pd.DataFrame],
    scf_env: Optional[pd.DataFrame],
    wet_env: Optional[pd.DataFrame],
    output: Path,
    title: str | None = None,
    assim_scf: Optional[list[pd.Timestamp]] = None,
    assim_wet: Optional[list[pd.Timestamp]] = None,
    assim_station_hs: Optional[list[pd.Timestamp]] = None,
    assim_station_swe: Optional[list[pd.Timestamp]] = None,
    assim_labels: Optional[dict[pd.Timestamp, str]] = None,
    mode: str = "band",
) -> None:
    """Render SCF and wet-snow series into one PNG (obs + ensemble bands)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_scf = scf_obs is not None or scf_model is not None
    has_wet = wet_obs is not None or wet_model is not None
    if scf_env is not None:
        has_scf = True
    if wet_env is not None:
        has_wet = True
    n_axes = int(has_scf) + int(has_wet)
    if n_axes == 0:
        raise ValueError("No data available to plot.")

    fig, axes = plt.subplots(n_axes, 1, figsize=(10, 4 * n_axes), sharex=True)
    if n_axes == 1:
        axes = [axes]

    fig_title = title if title else "openAMUNDSEN ensemble vs observations"
    fig.suptitle(fig_title, fontsize=14, y=0.98)

    idx = 0
    # Build DA date -> label mapping so labels match the global DA sequence
    label_map = {pd.to_datetime(k): v for k, v in (assim_labels or {}).items()}

    def _label_tuples(dates: Optional[list[pd.Timestamp]]) -> list[tuple[pd.Timestamp, str]]:
        if not dates:
            return []
        out = []
        for idx, d in enumerate(sorted(set(pd.to_datetime(dates))), start=1):
            lbl = label_map.get(d)
            if lbl is None and label_map:
                lbl = str(idx)
            out.append((d, lbl if lbl is not None else str(idx)))
        return out

    scf_labels = _label_tuples(assim_scf)
    wet_labels = _label_tuples(assim_wet)
    station_hs_dates = sorted(set(pd.to_datetime(assim_station_hs or [])))
    station_swe_dates = sorted(set(pd.to_datetime(assim_station_swe or [])))
    mode = (mode or "band").lower()
    if mode not in {"band", "members"}:
        mode = "band"

    def _date_bounds(*frames: Optional[pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        mins: list[pd.Timestamp] = []
        maxs: list[pd.Timestamp] = []
        for frame in frames:
            if frame is None or frame.empty or "date" not in frame.columns:
                continue
            dates = pd.to_datetime(frame["date"]).dropna()
            if dates.empty:
                continue
            mins.append(dates.min())
            maxs.append(dates.max())
        if not mins:
            return None
        return min(mins), max(maxs)

    def _draw_station_da(ax) -> None:
        if station_hs_dates:
            draw_assimilation_vlines(
                ax,
                station_hs_dates,
                color=COLOR_DA_STATION_HS,
                ls=LS_DA_STATION_HS,
                lw=LW_DA_STATION,
                alpha=0.95,
                label="_nolegend_",
            )
        if station_swe_dates:
            draw_assimilation_vlines(
                ax,
                station_swe_dates,
                color=COLOR_DA_STATION_SWE,
                ls=LS_DA_STATION_SWE,
                lw=LW_DA_STATION,
                alpha=0.95,
                label="_nolegend_",
            )

    if has_scf:
        ax = axes[idx]
        if mode == "band" and scf_env is not None and not scf_env.empty:
            ax.fill_between(
                scf_env["date"],
                scf_env["value_min"],
                scf_env["value_max"],
                color="#6ba9ff",  # soft snow-blue band
                alpha=0.6,
                label="SCF ensemble band",
            )
            ax.plot(scf_env["date"], scf_env["value_mean"], "-", color="#1f5faa", alpha=0.8, label="SCF ensemble mean")
        if scf_model is not None and not scf_model.empty:
            ax.plot(
                scf_model["date"],
                scf_model["scf"],
                "-",
                color="black",
                label="SCF open loop" if mode == "band" else "_nolegend_",
            )
        if scf_obs is not None and not scf_obs.empty:
            ax.plot(scf_obs["date"], scf_obs["scf"], "o", ms=5, color="tab:orange", label="SCF obs")
        _draw_station_da(ax)
        if assim_scf:
            draw_assimilation_markers(
                ax,
                dates=assim_scf,
                obs=scf_obs,
                value_col="scf",
                color=COLOR_DA_OBS,
                label="SCF DA obs",
                size=SIZE_DA_OBS,
                linewidth=LW_DA_OBS,
            )
        # Draw SCF DA labels on the SCF panel.
        draw_assim_labels(
            ax,
            [d for d, _ in scf_labels],
            labels=[lbl for _, lbl in scf_labels] if scf_labels else None,
            max_labels=12,
            y_offset_pts=3.0,
            fontsize=8.0,
            color="black",
        )
        draw_assim_labels(
            ax,
            station_hs_dates,
            labels=["HS"] * len(station_hs_dates) if station_hs_dates else None,
            max_labels=12,
            y_offset_pts=12.0,
            fontsize=8.0,
            color=COLOR_DA_STATION_HS,
        )
        draw_assim_labels(
            ax,
            station_swe_dates,
            labels=["SWE"] * len(station_swe_dates) if station_swe_dates else None,
            max_labels=12,
            y_offset_pts=12.0,
            fontsize=8.0,
            color=COLOR_DA_STATION_SWE,
        )
        ax.set_ylabel("Snow cover fraction")
        ax.set_ylim(0, 1)
        h, l = ax.get_legend_handles_labels()
        h, l = dedupe_legend(h, l)
        ax.legend(
            h,
            l,
            loc="upper left",
            fontsize=8.5,
            labelspacing=0.3,
            borderpad=0.3,
            handlelength=1.2,
            handletextpad=0.4,
        )
        apply_fraction_grid(ax, y_step=0.1)
        idx += 1

    if has_wet:
        ax = axes[idx]
        if mode == "band" and wet_env is not None and not wet_env.empty:
            ax.fill_between(
                wet_env["date"],
                wet_env["value_min"],
                wet_env["value_max"],
                color="#58c59c",  # wet-snow teal band
                alpha=0.6,
                label="Wet-snow ensemble band",
            )
            ax.plot(wet_env["date"], wet_env["value_mean"], "-", color="#1f7a5d", alpha=0.8, label="Wet-snow ensemble mean")
        if wet_model is not None and not wet_model.empty:
            ax.plot(
                wet_model["date"],
                wet_model["wet_snow_fraction"],
                "-",
                color="black",
                label="Wet-snow open loop" if mode == "band" else "_nolegend_",
            )
        if wet_obs is not None and not wet_obs.empty:
            ax.plot(wet_obs["date"], wet_obs["wet_snow_fraction"], "o", ms=5, color="tab:red", label="Wet-snow obs")
        _draw_station_da(ax)
        if assim_wet:
            draw_assimilation_markers(
                ax,
                dates=assim_wet,
                obs=wet_obs,
                value_col="wet_snow_fraction",
                color=COLOR_DA_OBS,
                label="Wet-snow DA obs",
                size=SIZE_DA_OBS,
                linewidth=LW_DA_OBS,
            )
        # Always draw wet-snow DA labels on the wet-snow panel.
        draw_assim_labels(
            ax,
            [d for d, _ in wet_labels],
            labels=[lbl for _, lbl in wet_labels] if wet_labels else None,
            max_labels=12,
            y_offset_pts=3.0,
            fontsize=8.0,
            color="black",
        )
        draw_assim_labels(
            ax,
            station_hs_dates,
            labels=["HS"] * len(station_hs_dates) if station_hs_dates else None,
            max_labels=12,
            y_offset_pts=12.0,
            fontsize=8.0,
            color=COLOR_DA_STATION_HS,
        )
        draw_assim_labels(
            ax,
            station_swe_dates,
            labels=["SWE"] * len(station_swe_dates) if station_swe_dates else None,
            max_labels=12,
            y_offset_pts=12.0,
            fontsize=8.0,
            color=COLOR_DA_STATION_SWE,
        )
        ax.set_ylabel("Wet snow fraction")
        ax.set_ylim(0, 1)
        h, l = ax.get_legend_handles_labels()
        h, l = dedupe_legend(h, l)
        ax.legend(
            h,
            l,
            loc="upper left",
            fontsize=8.5,
            labelspacing=0.3,
            borderpad=0.3,
            handlelength=1.2,
            handletextpad=0.4,
        )
        apply_fraction_grid(ax, y_step=0.1)

    x_bounds = _date_bounds(scf_model, wet_model, scf_env, wet_env)
    if x_bounds is None:
        x_bounds = _date_bounds(scf_obs, wet_obs)
    if x_bounds is not None:
        for ax in axes:
            ax.set_xlim(*x_bounds)

    axes[-1].set_xlabel("")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cli_main(argv: list[str] | None = None, *, configure_logger: bool = True) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="oa-da-plot-fractions",
        description="Plot SCF and wet-snow fractions (obs and model) in one figure.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory (setup/projects/project_YYYY_YYYY)")
    parser.add_argument("--setup-dir", type=Path, help="Setup directory (default: project_dir/../..)")
    parser.add_argument("--scf-obs-csv", type=Path, help="Path to scf_summary.csv (obs)")
    parser.add_argument("--wet-obs-csv", type=Path, help="Path to wet_snow_summary.csv (obs)")
    parser.add_argument("--scf-model-csv", type=Path, help="Model SCF CSV (date/time + scf)")
    parser.add_argument("--wet-model-csv", type=Path, help="Model wet-snow CSV (date/time + wet_snow_fraction)")
    parser.add_argument("--scf-env-csv", type=Path, help="SCF envelope CSV (value_min/value_max/value_mean)")
    parser.add_argument("--wet-env-csv", type=Path, help="Wet-snow envelope CSV (value_min/value_max/value_mean)")
    parser.add_argument("--output", type=Path, help="Output PNG path (default: <project>/plots/results/fraction_timeseries.png)")
    parser.add_argument("--title", type=str, help="Figure title")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    parser.add_argument("--mode", choices=["band", "members"], default="band", help="Plot mode: band (default) or members")
    args = parser.parse_args(argv)

    if configure_logger:
        configure_cli_logger(args.log_level)

    project_dir = Path(args.project_dir)
    setup_dir = Path(args.setup_dir) if args.setup_dir else project_dir.parent.parent
    project_name = project_dir.name

    scf_obs_path = Path(args.scf_obs_csv) if args.scf_obs_csv else default_fraction_obs_path(setup_dir, project_name, "scf_summary.csv")
    wet_obs_path = Path(args.wet_obs_csv) if args.wet_obs_csv else default_fraction_obs_path(setup_dir, project_name, "wet_snow_summary.csv")
    scf_env_path = Path(args.scf_env_csv) if args.scf_env_csv else (project_dir / "point_scf_roi_envelope.csv")
    wet_env_path = Path(args.wet_env_csv) if args.wet_env_csv else (project_dir / "point_wet_snow_roi_envelope.csv")

    scf_obs = _load_scf_obs_series(scf_obs_path)
    wet_obs = load_fraction_series(wet_obs_path, "wet_snow_fraction")
    scf_model = load_fraction_series(Path(args.scf_model_csv), "scf") if args.scf_model_csv else None
    wet_model = load_fraction_series(Path(args.wet_model_csv), "wet_snow_fraction") if args.wet_model_csv else None
    if scf_model is None:
        scf_model = load_open_loop_fraction_series(project_dir, "point_scf_roi.csv", "scf")
    if wet_model is None:
        wet_model = load_open_loop_fraction_series(project_dir, "point_wet_snow_roi.csv", "wet_snow_fraction")
    scf_env = load_fraction_series(scf_env_path, "value_mean")
    if scf_env is not None and not scf_env.empty and {"value_min", "value_max"}.issubset(scf_env.columns) is False:
        scf_env = None
    wet_env = load_fraction_series(wet_env_path, "value_mean")
    if wet_env is not None and not wet_env.empty and {"value_min", "value_max"}.issubset(wet_env.columns) is False:
        wet_env = None

    if scf_obs is None or scf_obs.empty:
        logger.warning("SCF obs not found at {} - plotting without obs points", scf_obs_path)
    if wet_obs is None or wet_obs.empty:
        logger.warning("Wet-snow obs not found at {} - plotting without obs points", wet_obs_path)

    try:
        assim_events = load_assimilation_events(project_dir)
    except FileNotFoundError:
        assim_events = []
    assim_scf = [pd.to_datetime(ev.date) for ev in assim_events if ev.variable == "scf"]
    assim_wet = [pd.to_datetime(ev.date) for ev in assim_events if ev.variable == "wet_snow"]
    assim_station_hs = [pd.to_datetime(ev.date) for ev in assim_events if ev.variable == "station_hs"]
    assim_station_swe = [pd.to_datetime(ev.date) for ev in assim_events if ev.variable == "station_swe"]

    if all(x is None or x.empty for x in (scf_obs, wet_obs, scf_model, wet_model, scf_env, wet_env)):
        logger.error("No data available to plot. Provide at least one obs/model series.")
        return 1

    out_path = default_fraction_plot_output(project_dir, args.output)
    fig_title = str(args.title) if args.title else None
    # Build global DA labels (shared numbering across variables) for consistent
    # annotation in both SCF and wet-snow panels.
    assim_labels = {pd.to_datetime(ev.date): str(i) for i, ev in enumerate(assim_events, start=1)}

    try:
        plot_fraction_timeseries(
            scf_obs=scf_obs,
            scf_model=scf_model,
            wet_obs=wet_obs,
            wet_model=wet_model,
            scf_env=scf_env,
            wet_env=wet_env,
            output=out_path,
            title=fig_title or "openAMUNDSEN ensemble vs observations",
            assim_scf=assim_scf,
            assim_wet=assim_wet,
            assim_station_hs=assim_station_hs,
            assim_station_swe=assim_station_swe,
            assim_labels=assim_labels,
            mode=str(args.mode or "band"),
        )
    except ModuleNotFoundError as exc:
        logger.error("matplotlib is required to plot: {}", exc)
        return 1
    except Exception as exc:
        logger.error("Plotting failed: {}", exc)
        return 1

    logger.info("Wrote plot: {}", out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

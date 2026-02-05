"""
prepare_season_obs.py
Author: Franz Wagner
Date: 2026-02-05
Description:
    Screen season-wide SCF and wet-snow observation summaries, propose
    assimilation dates based on simple thresholds, and emit helper
    artifacts (season.yml + obs-only plots) for manual review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.observer.fraction_obs import resolve_obs_product_tag
from openamundsen_da.observer.plot_fractions import (
    _default_obs_path,
    _load_fraction,
    plot_fractions,
)
from openamundsen_da.observer.plot_scf_summary import _load_summary as _load_scf_summary

DEFAULT_PRIORITY = ["wet_snow", "scf"]


@dataclass(frozen=True)
class Candidate:
    """Container for filtered observation candidates."""

    date: pd.Timestamp
    variable: str
    value: float


def _parse_date(text: str | None) -> pd.Timestamp:
    if not text:
        raise ValueError("Date is required (YYYY-MM-DD)")
    return pd.to_datetime(str(text).strip()).normalize()


def _date_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def _load_config(config_path: Path | None) -> dict:
    if not config_path:
        return {}
    if not config_path.is_file():
        logger.warning("Config not found at {} – using CLI/defaults", config_path)
        return {}
    try:
        return _read_yaml_file(config_path) or {}
    except Exception:
        logger.warning("Failed to read config {} – using CLI/defaults", config_path)
        return {}


def _var_filter_cfg(variable: str, cfg: dict, args: argparse.Namespace) -> dict:
    filters = (cfg.get("filters") or {}).get(variable, {})
    defaults = {
        "min_fraction": args.min_fraction,
        "max_fraction": args.max_fraction,
        "max_deviation": args.max_deviation,
        "max_delta": args.max_delta,
        "smoothing_window": args.smoothing_window,
    }
    out = {}
    for key, default in defaults.items():
        val = filters.get(key, default)
        out[key] = float(val) if key != "smoothing_window" else int(val)
    return out


def _load_obs(
    *,
    season_dir: Path,
    project_dir: Path,
    scf_summary: Path | None,
    wet_summary: Path | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    season_name = season_dir.name
    scf_path = Path(scf_summary) if scf_summary else _default_obs_path(project_dir, season_name, "scf_summary.csv")
    wet_path = Path(wet_summary) if wet_summary else _default_obs_path(project_dir, season_name, "wet_snow_summary.csv")

    scf_df: pd.DataFrame | None = None
    wet_df: pd.DataFrame | None = None

    if scf_path.is_file():
        scf_df = _load_scf_summary(scf_path)
        scf_df = _date_range(scf_df, start, end)
    else:
        logger.warning("SCF summary not found at {}", scf_path)

    if wet_path.is_file():
        wet_df = _load_fraction(wet_path, "wet_snow_fraction")
        if wet_df is not None:
            wet_df = _date_range(wet_df, start, end)
    else:
        logger.warning("Wet-snow summary not found at {}", wet_path)

    return scf_df, wet_df


def _filter_candidates(
    df: pd.DataFrame,
    *,
    variable: str,
    value_col: str,
    min_fraction: float,
    max_fraction: float,
    max_deviation: float,
    max_delta: float,
    smoothing_window: int,
) -> List[Candidate]:
    df = df.copy()
    df["rolling_med"] = df[value_col].rolling(smoothing_window, center=True, min_periods=1).median()
    df["delta"] = df[value_col].diff().abs()

    mask = df[value_col].between(min_fraction, max_fraction)
    mask &= (df[value_col] - df["rolling_med"]).abs() <= max_deviation
    mask &= df["delta"].fillna(0) <= max_delta

    kept = []
    for _, row in df[mask].iterrows():
        kept.append(Candidate(date=row["date"], variable=variable, value=float(row[value_col])))
    return kept


def _select_with_spacing(
    candidates: Sequence[Candidate],
    *,
    spacing_days: int,
    priority: Sequence[str] | None,
    secondary_every_n: int | None,
) -> list[Candidate]:
    spacing = max(1, int(spacing_days))
    items = sorted(candidates, key=lambda c: c.date)
    selected: list[Candidate] = []
    idx = 0
    priority_list = list(priority) if priority else []
    primary_var = priority_list[0] if priority_list else None
    secondary_var = priority_list[1] if len(priority_list) > 1 else None
    alt_counter = 0
    sec_n = secondary_every_n if secondary_every_n and secondary_every_n > 0 else None

    def _var_rank(var: str) -> int:
        if var in priority_list:
            return priority_list.index(var)
        return len(priority_list) + 1

    while idx < len(items):
        window_start = items[idx].date
        window_end = window_start + pd.Timedelta(days=spacing - 1)
        window: list[Candidate] = []
        j = idx
        while j < len(items) and items[j].date <= window_end:
            window.append(items[j])
            j += 1
        window_sorted = sorted(window, key=lambda c: (_var_rank(c.variable), c.date))

        choose = None
        if sec_n and primary_var and secondary_var and secondary_var in {c.variable for c in window} and primary_var in {c.variable for c in window}:
            alt_counter += 1
            if alt_counter % sec_n == 0:
                secondary_candidates = [c for c in window_sorted if c.variable == secondary_var]
                if secondary_candidates:
                    choose = secondary_candidates[0]

        if choose is None:
            choose = window_sorted[0]

        selected.append(choose)
        idx = j
    return selected


def _resolve_products(variables: Iterable[str], *, project_dir: Path, season_dir: Path) -> dict[str, str]:
    products: dict[str, str] = {}
    for var in variables:
        products[var] = resolve_obs_product_tag(var, project_dir=project_dir, season_dir=season_dir)
    return products


def _write_season_yaml(
    *,
    season_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    events: list[Candidate],
    products: dict[str, str],
    overwrite: bool,
) -> Path:
    season_yaml = season_dir / "season.yml"
    if season_yaml.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing {season_yaml} (use --overwrite)")

    data = _read_yaml_file(season_yaml) if season_yaml.exists() else {}
    data["start_date"] = start.strftime("%Y-%m-%d")
    data["end_date"] = end.strftime("%Y-%m-%d")
    da_cfg = data.get("data_assimilation") or {}

    assim_events = []
    for cand in sorted(events, key=lambda c: c.date):
        entry = {"date": cand.date.strftime("%Y-%m-%d"), "variable": cand.variable}
        product = products.get(cand.variable)
        if product:
            entry["product"] = product
        assim_events.append(entry)

    da_cfg["assimilation_events"] = assim_events
    data["data_assimilation"] = da_cfg

    try:
        import ruamel.yaml as _yaml

        y = _yaml.YAML()
        y.default_flow_style = False
        season_dir.mkdir(parents=True, exist_ok=True)
        with season_yaml.open("w", encoding="utf-8") as f:
            y.dump(data, f)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to write {season_yaml}: {exc}") from exc

    return season_yaml


def _plot_obs_only(
    *,
    season_dir: Path,
    scf_obs: pd.DataFrame | None,
    wet_obs: pd.DataFrame | None,
    selected: list[Candidate],
    title: str | None,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    assim_labels = {cand.date: str(idx) for idx, cand in enumerate(sorted(selected, key=lambda c: c.date), start=1)}
    assim_scf = [cand.date for cand in selected if cand.variable == "scf"]
    assim_wet = [cand.date for cand in selected if cand.variable == "wet_snow"]

    plot_fractions(
        scf_obs=scf_obs,
        scf_model=None,
        wet_obs=wet_obs,
        wet_model=None,
        scf_env=None,
        wet_env=None,
        output=output,
        title=title or f"Observation screening for {season_dir.name}",
        assim_scf=assim_scf,
        assim_wet=assim_wet,
        assim_labels=assim_labels,
        mode="band",
    )


def _infer_season_window(
    *, season_dir: Path, start: str | None, end: str | None, scf: pd.DataFrame | None, wet: pd.DataFrame | None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if start:
        start_dt = _parse_date(start)
    else:
        cfg = _read_yaml_file(season_dir / "season.yml")
        start_txt = cfg.get("start_date") if cfg else None
        start_dt = _parse_date(start_txt) if start_txt else None
    if end:
        end_dt = _parse_date(end)
    else:
        cfg = _read_yaml_file(season_dir / "season.yml")
        end_txt = cfg.get("end_date") if cfg else None
        end_dt = _parse_date(end_txt) if end_txt else None

    if start_dt is None:
        min_candidates = []
        for df in (scf, wet):
            if df is not None and not df.empty:
                min_candidates.append(df["date"].min())
        if not min_candidates:
            raise ValueError("Could not infer start date (provide --start-date)")
        start_dt = min(min_candidates)

    if end_dt is None:
        max_candidates = []
        for df in (scf, wet):
            if df is not None and not df.empty:
                max_candidates.append(df["date"].max())
        if not max_candidates:
            raise ValueError("Could not infer end date (provide --end-date)")
        end_dt = max(max_candidates)

    return start_dt.normalize(), end_dt.normalize()


def _configure_logger(level: str) -> None:
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level.upper(),
        colorize=True,
        format=LOGURU_FORMAT,
        enqueue=True,
    )


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point for observation screening."""
    parser = argparse.ArgumentParser(
        prog="oa-da-prepare-season-obs",
        description="Screen observation summaries and propose assimilation dates.",
    )
    parser.add_argument("--season-dir", required=True, type=Path, help="Season directory (e.g., propagation/season_2020_2021)")
    parser.add_argument("--project-dir", type=Path, help="Project directory (default: season_dir/../..)")
    parser.add_argument("--config", type=Path, help="Config YAML for per-variable filters/priority (default: <project>/obs_selection.config.yml if present)")
    parser.add_argument("--scf-summary", type=Path, help="Path to scf_summary.csv")
    parser.add_argument("--wet-summary", type=Path, help="Path to wet_snow_summary.csv")
    parser.add_argument("--start-date", type=str, help="Season start date (YYYY-MM-DD). Defaults to season.yml or data min.")
    parser.add_argument("--end-date", type=str, help="Season end date (YYYY-MM-DD). Defaults to season.yml or data max.")
    parser.add_argument("--min-fraction", type=float, default=0.20, help="Minimum fraction to keep (default: 0.20)")
    parser.add_argument("--max-fraction", type=float, default=0.80, help="Maximum fraction to keep (default: 0.80)")
    parser.add_argument("--max-deviation", type=float, default=0.25, help="Max abs deviation from rolling median (default: 0.25)")
    parser.add_argument("--max-delta", type=float, default=0.35, help="Max day-to-day jump to keep (default: 0.35)")
    parser.add_argument("--smoothing-window", type=int, default=3, help="Rolling window for median smoothing (default: 3)")
    parser.add_argument("--spacing-days", type=int, default=7, help="Minimum spacing between selected dates (default: 7)")
    parser.add_argument(
        "--secondary-every-n",
        type=int,
        default=0,
        help="When both primary and secondary vars are present in a window, pick the secondary every Nth time (0 disables)",
    )
    parser.add_argument("--output-plot", type=Path, help="Output PNG for obs-only plot (default: <season>/plots/results/obs_selection.png)")
    parser.add_argument("--output-season-yaml", type=Path, help="Output season.yml path (default: <season>/season.yml)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing season.yml")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-write-season", action="store_true", help="Skip writing season.yml")
    args = parser.parse_args(argv)

    _configure_logger(args.log_level)

    season_dir = Path(args.season_dir)
    project_dir = Path(args.project_dir) if args.project_dir else season_dir.parent.parent

    default_cfg_path = project_dir / "obs_selection.config.yml"
    config_path = Path(args.config) if args.config else (default_cfg_path if default_cfg_path.is_file() else None)
    cfg = _load_config(config_path)

    scf_df, wet_df = _load_obs(
        season_dir=season_dir,
        project_dir=project_dir,
        scf_summary=args.scf_summary,
        wet_summary=args.wet_summary,
        start=_parse_date(args.start_date) if args.start_date else pd.Timestamp("1900-01-01"),
        end=_parse_date(args.end_date) if args.end_date else pd.Timestamp("2100-01-01"),
    )

    start_dt, end_dt = _infer_season_window(
        season_dir=season_dir,
        start=args.start_date,
        end=args.end_date,
        scf=scf_df,
        wet=wet_df,
    )

    if scf_df is not None:
        scf_df = _date_range(scf_df, start_dt, end_dt)
    if wet_df is not None:
        wet_df = _date_range(wet_df, start_dt, end_dt)

    selection_cfg = cfg.get("selection") or {}
    priority = selection_cfg.get("prefer_variables") or DEFAULT_PRIORITY
    spacing_days = int(selection_cfg.get("spacing_days", args.spacing_days))
    secondary_every_n = int(selection_cfg.get("secondary_every_n", args.secondary_every_n))

    candidates: list[Candidate] = []
    if scf_df is not None and not scf_df.empty:
        scf_cfg = _var_filter_cfg("scf", cfg, args)
        scf_cands = _filter_candidates(
            scf_df,
            variable="scf",
            value_col="scf",
            min_fraction=scf_cfg["min_fraction"],
            max_fraction=scf_cfg["max_fraction"],
            max_deviation=scf_cfg["max_deviation"],
            max_delta=scf_cfg["max_delta"],
            smoothing_window=scf_cfg["smoothing_window"],
        )
        logger.info("SCF candidates kept: {} of {}", len(scf_cands), len(scf_df))
        candidates.extend(scf_cands)
    else:
        logger.warning("No SCF data available in the selected window")

    if wet_df is not None and not wet_df.empty:
        wet_cfg = _var_filter_cfg("wet_snow", cfg, args)
        wet_cands = _filter_candidates(
            wet_df,
            variable="wet_snow",
            value_col="wet_snow_fraction",
            min_fraction=wet_cfg["min_fraction"],
            max_fraction=wet_cfg["max_fraction"],
            max_deviation=wet_cfg["max_deviation"],
            max_delta=wet_cfg["max_delta"],
            smoothing_window=wet_cfg["smoothing_window"],
        )
        logger.info("Wet-snow candidates kept: {} of {}", len(wet_cands), len(wet_df))
        candidates.extend(wet_cands)
    else:
        logger.warning("No wet-snow data available in the selected window")

    selected = _select_with_spacing(
        candidates,
        spacing_days=spacing_days,
        priority=priority,
        secondary_every_n=secondary_every_n if secondary_every_n > 0 else None,
    )
    if not selected:
        logger.error("No observations passed the filters. Adjust thresholds and retry.")
        return 1

    products = _resolve_products({c.variable for c in selected}, project_dir=project_dir, season_dir=season_dir)

    for idx, cand in enumerate(sorted(selected, key=lambda c: c.date), start=1):
        logger.info(
            "[{:02d}] {} | {} | value={:.3f} | product={}",
            idx,
            cand.date.date(),
            cand.variable,
            cand.value,
            products.get(cand.variable, "-"),
        )

    if not args.no_write_season:
        season_yaml = args.output_season_yaml if args.output_season_yaml else (season_dir / "season.yml")
        season_path = _write_season_yaml(
            season_dir=season_yaml.parent,
            start=start_dt,
            end=end_dt,
            events=selected,
            products=products,
            overwrite=args.overwrite,
        )
        logger.info("Wrote season file: {}", season_path)

    if not args.no_plot:
        plot_path = args.output_plot if args.output_plot else (season_dir / "plots" / "results" / "obs_selection.png")
        try:
            _plot_obs_only(
                season_dir=season_dir,
                scf_obs=scf_df,
                wet_obs=wet_df,
                selected=selected,
                title=f"Obs selection for {season_dir.name}",
                output=plot_path,
            )
            logger.info("Wrote obs-only plot: {}", plot_path)
        except ModuleNotFoundError as exc:
            logger.error("Plotting requires matplotlib: {}", exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Plotting failed: {}", exc)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

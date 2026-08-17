"""Project-level summary reports for independent sub-domain runs."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.io.paths import list_steps_sorted
from openamundsen_da.subdomain.manifest import SubdomainManifest
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.run_mode import ensure_run_mode
from openamundsen_da.util.stats import effective_sample_size

_WEIGHTS_RE = re.compile(r"weights_(?P<variable>.+)_(?P<date>\d{8})\.csv$", re.IGNORECASE)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read JSON {}: {}", path, exc)
        return {}


def _normalize_weights(raw: np.ndarray) -> np.ndarray:
    w = np.asarray(raw, dtype=np.float64)
    w = w[np.isfinite(w)]
    if w.size == 0:
        return w
    s = float(np.sum(w))
    if s <= 0:
        return np.array([], dtype=np.float64)
    return w / s


def _entropy(weights: np.ndarray) -> float:
    if weights.size == 0:
        return float("nan")
    safe = np.where(weights > 0.0, weights, 1.0)
    return float(-np.sum(np.where(weights > 0.0, weights * np.log(safe), 0.0)))


def _first_numeric(series: pd.Series | None) -> float:
    if series is None:
        return float("nan")
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return float("nan")
    return float(vals.iloc[0])


def _subdomain_assimilation_stats(subdomain_id: str, project_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step_dir in list_steps_sorted(project_dir):
        assim_dir = step_dir / "assim"
        if not assim_dir.is_dir():
            continue
        for weights_csv in sorted(assim_dir.glob("weights_*_*.csv")):
            m = _WEIGHTS_RE.match(weights_csv.name)
            if not m:
                continue
            try:
                df = pd.read_csv(weights_csv)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read weights CSV {}: {}", weights_csv, exc)
                continue
            if "weight" not in df.columns:
                logger.warning("Skipping weights CSV without 'weight' column: {}", weights_csv)
                continue
            w = _normalize_weights(pd.to_numeric(df["weight"], errors="coerce").to_numpy(dtype=np.float64))
            if w.size == 0:
                logger.warning("Skipping weights CSV with no valid positive weights: {}", weights_csv)
                continue
            ess = float(effective_sample_size(w))
            n_members = int(w.size)
            date_txt = m.group("date")
            date_iso = datetime.strptime(date_txt, "%Y%m%d").date().isoformat()
            rows.append(
                {
                    "subdomain_id": subdomain_id,
                    "step": step_dir.name,
                    "date": date_iso,
                    "variable": str(m.group("variable")),
                    "n_members": n_members,
                    "ess": ess,
                    "ess_norm": float(ess / n_members),
                    "sigma": _first_numeric(df["sigma"] if "sigma" in df.columns else None),
                    "weight_entropy": _entropy(w),
                    "max_weight": float(np.max(w)),
                    "weights_csv": str(weights_csv),
                }
            )
    return rows


def _p10(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(np.nanquantile(values.to_numpy(dtype=np.float64), 0.10))


def write_subdomain_reports(
    *,
    manifest_path: Path,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Write project-level CSV summaries for independent sub-domain runs."""
    manifest = SubdomainManifest.load(manifest_path)
    if str(getattr(manifest, "run_mode", "")).lower() != "subdomain":
        raise ValueError(f"Manifest at {manifest_path} is not marked as run_mode='subdomain'.")
    ensure_run_mode(manifest.project_dir, expected="subdomain", write_if_missing=False)

    out_base = Path(out_dir) if out_dir is not None else (manifest.project_dir / "results")
    out_base.mkdir(parents=True, exist_ok=True)

    overview_rows: list[dict[str, Any]] = []
    stat_rows: list[dict[str, Any]] = []
    for sid, sub in sorted(manifest.subdomains.items()):
        run_manifest = _read_json(sub.run_manifest if sub.run_manifest is not None else (sub.setup_dir / "run_manifest.json"))
        try:
            n_steps = len(list_steps_sorted(sub.project_dir))
        except Exception:
            n_steps = 0
        try:
            n_events = len(load_assimilation_events(sub.project_dir))
        except Exception:
            n_events = 0
        overview_rows.append(
            {
                "subdomain_id": sid,
                "label": sub.label,
                "status": str(run_manifest.get("status", sub.status)),
                "duration_seconds": float(run_manifest.get("duration_seconds", float("nan"))),
                **{
                    f"{phase}_duration_seconds": float(
                        (run_manifest.get("phases") or {})
                        .get(phase, {})
                        .get("duration_seconds", float("nan"))
                    )
                    for phase in (
                        "propagation",
                        "compact_export",
                        "render",
                        "cleanup",
                    )
                },
                "n_steps": int(n_steps),
                "n_assimilation_events": int(n_events),
                **{
                    key: int(value)
                    for key, value in (getattr(sub, "station_counts", None) or {}).items()
                },
                "setup_dir": str(sub.setup_dir),
                "project_dir": str(sub.project_dir),
                "run_manifest": str(sub.run_manifest or (sub.setup_dir / "run_manifest.json")),
            }
        )
        stat_rows.extend(_subdomain_assimilation_stats(sid, sub.project_dir))

    outputs: dict[str, Path] = {}
    overview_path = out_base / "subdomain_overview.csv"
    pd.DataFrame(overview_rows).to_csv(overview_path, index=False)
    logger.info("Wrote {}", overview_path)
    outputs["overview"] = overview_path

    stats_path = out_base / "subdomain_assimilation_stats.csv"
    if stat_rows:
        stats_df = pd.DataFrame(stat_rows).sort_values(["subdomain_id", "date", "variable", "step"])
        stats_df.to_csv(stats_path, index=False)
        logger.info("Wrote {}", stats_path)
        outputs["assimilation_stats"] = stats_path

        agg_df = (
            stats_df.groupby("subdomain_id", as_index=False)
            .agg(
                events_count=("ess_norm", "count"),
                ess_norm_mean=("ess_norm", "mean"),
                ess_norm_min=("ess_norm", "min"),
                ess_norm_p10=("ess_norm", _p10),
                max_weight_mean=("max_weight", "mean"),
                max_weight_max=("max_weight", "max"),
            )
            .sort_values("subdomain_id")
        )
        agg_path = out_base / "subdomain_assimilation_aggregate.csv"
        agg_df.to_csv(agg_path, index=False)
        logger.info("Wrote {}", agg_path)
        outputs["assimilation_aggregate"] = agg_path
    else:
        logger.warning("No assimilation weight CSV files found for sub-domain report generation.")

    return outputs

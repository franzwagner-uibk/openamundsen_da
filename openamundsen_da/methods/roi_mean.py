"""Daily ROI mean time-series helpers for scalar grid variables.

This module computes full-ROI daily means for scalar grid outputs such as SWE
and snow depth. It intentionally reuses:

- ``find_member_daily_grid_slice`` for GeoTIFF/NetCDF discovery,
- ``compute_step_daily_series_for_all_members`` for step/member orchestration,
- ``read_grid_slice_roi_masked_array`` for ROI clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import VAR_HS, VAR_SWE
from openamundsen_da.core.env import ensure_gdal_proj_from_conda
from openamundsen_da.io.paths import find_member_daily_grid_slice
from openamundsen_da.methods.daily_aoi_series import (
    compute_step_daily_series_for_all_members,
    step_start_end,
)
from openamundsen_da.util.grid_roi import read_grid_slice_roi_masked_array


@dataclass(frozen=True)
class RoiMeanSpec:
    variable: str
    value_col: str
    csv_name: str
    label: str


ROI_MEAN_SPECS: dict[str, RoiMeanSpec] = {
    VAR_SWE: RoiMeanSpec(
        variable=VAR_SWE,
        value_col="swe",
        csv_name="point_swe_roi.csv",
        label="SWE",
    ),
    VAR_HS: RoiMeanSpec(
        variable=VAR_HS,
        value_col="snow_depth",
        csv_name="point_snow_depth_roi.csv",
        label="snow depth",
    ),
}


def roi_mean_spec(variable: str) -> RoiMeanSpec:
    """Return ROI mean output metadata for a supported scalar grid variable."""
    try:
        return ROI_MEAN_SPECS[variable]
    except KeyError as exc:
        supported = ", ".join(sorted(ROI_MEAN_SPECS))
        raise ValueError(f"Unsupported ROI mean variable '{variable}'. Expected one of: {supported}") from exc


def compute_member_roi_mean_daily(
    *,
    results_dir: Path,
    aoi_path: Path,
    start: datetime,
    end: datetime,
    variable: str,
    model_grid_format: str,
) -> pd.DataFrame:
    """Return the full-ROI daily mean for one member over a date range."""
    ensure_gdal_proj_from_conda()
    spec = roi_mean_spec(variable)

    start_day = datetime(start.year, start.month, start.day)
    end_day = datetime(end.year, end.month, end.day)
    if end_day < start_day:
        return pd.DataFrame(columns=["time", spec.value_col])

    dates = pd.date_range(start_day, end_day, freq="D").to_pydatetime()
    rows: list[dict[str, object]] = []
    for dt in dates:
        try:
            slice_ = find_member_daily_grid_slice(
                Path(results_dir),
                spec.variable,
                dt.strftime("%Y-%m-%d"),
                preferred_format=model_grid_format,
            )
            arr = read_grid_slice_roi_masked_array(
                slice_,
                Path(aoi_path),
                landcover_cfg=None,
            )
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ROI mean {} computation failed for {} at {}: {}",
                spec.label,
                results_dir,
                dt.date(),
                exc,
            )
            continue
        rows.append({"time": dt, spec.value_col: float(np.ma.mean(arr))})

    if not rows:
        return pd.DataFrame(columns=["time", spec.value_col])
    return pd.DataFrame(rows).sort_values("time")


def _compute_member_roi_mean_daily_worker(
    results_dir: Path,
    aoi_path: Path,
    start: datetime,
    end: datetime,
    out_csv: Path,
    overwrite: bool,
    extra: Dict[str, Any],
) -> bool:
    """Worker: compute one ROI daily-mean series for a single member."""
    variable = str(extra["variable"])
    model_grid_format = str(extra["model_grid_format"])
    df = compute_member_roi_mean_daily(
        results_dir=results_dir,
        aoi_path=aoi_path,
        start=start,
        end=end,
        variable=variable,
        model_grid_format=model_grid_format,
    )
    if df.empty:
        return False
    if out_csv.exists() and not overwrite:
        return False
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return True


def compute_step_roi_mean_daily_for_all_members(
    *,
    step_dir: Path,
    aoi_path: Path,
    variable: str,
    model_grid_format: str,
    max_workers: int = 4,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    """Compute full-ROI daily mean series for all prior members in one step."""
    spec = roi_mean_spec(variable)
    start, end = step_start_end(step_dir)
    return compute_step_daily_series_for_all_members(
        step_dir=Path(step_dir),
        aoi_path=Path(aoi_path),
        start=start,
        end=end,
        csv_name=spec.csv_name,
        worker=_compute_member_roi_mean_daily_worker,
        ensemble="prior",
        include_open_loop=True,
        max_workers=max_workers,
        overwrite=overwrite,
        worker_kwargs={
            "variable": spec.variable,
            "model_grid_format": model_grid_format,
        },
    )

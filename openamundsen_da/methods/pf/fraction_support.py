from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS
from rasterio.warp import Resampling, reproject

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import abspath_relative_to, find_project_yaml
from openamundsen_da.observer.class_config import load_observation_classes, load_wetsnow_classes
from openamundsen_da.util.config_validators import require_mapping
from openamundsen_da.util.landcover_mask import LandcoverMaskConfig, apply_landcover_mask
from openamundsen_da.util.roi_grid import load_setup_roi_mask


@dataclass(frozen=True)
class ObservationSupportMask:
    mask: np.ndarray
    eligible_mask: np.ndarray
    n_valid: int
    n_eligible: int
    coverage_ratio: float


def _observation_dir(project_dir: Path, *, observable: str, setup_dir: Path) -> Path:
    cfg = require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    obs_cfg = require_mapping(cfg.get("obs"), path="project.obs")
    key = "snowcover" if observable == "scf" else "wetsnow"
    section = require_mapping(obs_cfg.get(key), path=f"project.obs.{key}")
    raw_dir = section.get("dir")
    if raw_dir is None:
        raise ValueError(f"Missing 'obs.{key}.dir' in {find_project_yaml(project_dir)}")
    return Path(abspath_relative_to(setup_dir, Path(str(raw_dir))))


def _split_source_tokens(value: object) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    tokens = [token.strip() for token in re.split(r"[;,]", text) if token.strip()]
    return tuple(sorted(set(tokens)))


def _source_path_from_token(obs_dir: Path, token: str) -> Path:
    source_token = str(token).strip()
    source_name = source_token.split("@", 1)[0]
    source_path = Path(source_name)
    if not source_path.is_absolute():
        source_path = obs_dir / source_path
    if not source_path.is_file():
        raise FileNotFoundError(f"Observation source raster not found: {source_path}")
    if source_path.suffix.lower() not in {".tif", ".tiff"}:
        raise NotImplementedError(
            f"Observation support masking currently supports GeoTIFF sources only, got: {source_path.name}"
        )
    return source_path


def _scf_valid_pixels(data: np.ndarray, *, nodata: float | int | None, project_dir: Path) -> np.ndarray:
    classes = load_observation_classes(project_dir, obs_key="snowcover")
    valid = np.isfinite(data)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        valid &= data != nodata
    valid_classes = set(classes.get("valid", []))
    if valid_classes:
        valid &= np.isin(data, list(valid_classes))
    invalid_classes = set(classes.get("cloud", [])) | set(classes.get("water", [])) | set(classes.get("nodata", []))
    if invalid_classes:
        valid &= ~np.isin(data, list(invalid_classes))
    return valid


def _wet_snow_valid_pixels(data: np.ndarray, *, nodata: float | int | None, project_dir: Path) -> np.ndarray:
    _wet, valid_values, exclude_values = load_wetsnow_classes(project_dir)
    valid = np.isfinite(data)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        valid &= data != nodata
    if valid_values:
        valid &= np.isin(data, list(valid_values))
    if exclude_values:
        valid &= ~np.isin(data, list(exclude_values))
    return valid


def _observation_valid_pixels(
    *,
    data: np.ndarray,
    nodata: float | int | None,
    observable: str,
    project_dir: Path,
) -> np.ndarray:
    if observable == "scf":
        return _scf_valid_pixels(data, nodata=nodata, project_dir=project_dir)
    if observable == "wet_snow":
        return _wet_snow_valid_pixels(data, nodata=nodata, project_dir=project_dir)
    raise ValueError(f"Unsupported observable for observation support: {observable}")


def _eligible_model_mask(
    *,
    setup_dir: Path,
    landcover_cfg: LandcoverMaskConfig | None,
) -> tuple[np.ndarray, object]:
    roi_mask, spec, _ = load_setup_roi_mask(setup_dir, ensure_grid=True)
    arr = np.ma.array(np.ones(roi_mask.shape, dtype=float), mask=~roi_mask, copy=False)
    if landcover_cfg is not None and landcover_cfg.enabled:
        target_crs = CRS.from_user_input(spec.crs) if spec.crs is not None else None
        if target_crs is None:
            raise ValueError(f"Setup grid CRS is missing for {setup_dir}")
        arr, _ = apply_landcover_mask(
            arr,
            transform=spec.transform,
            target_crs=target_crs,
            roi_mask=roi_mask,
            lc_cfg=landcover_cfg,
        )
    eligible = (~np.ma.getmaskarray(arr)) & np.isfinite(np.ma.getdata(arr))
    return eligible, spec


def load_observation_support_mask(
    *,
    setup_dir: Path,
    project_dir: Path,
    obs_csv: Path,
    observable: str,
    landcover_cfg: LandcoverMaskConfig | None,
) -> ObservationSupportMask:
    df = pd.read_csv(obs_csv)
    if df.empty:
        raise ValueError(f"Observation CSV has no rows: {obs_csv}")
    row = df.iloc[0]
    tokens = _split_source_tokens(row.get("source"))
    if not tokens:
        raise ValueError(f"Observation CSV missing source token(s): {obs_csv}")

    eligible_mask, spec = _eligible_model_mask(setup_dir=setup_dir, landcover_cfg=landcover_cfg)
    support_mask = np.zeros(eligible_mask.shape, dtype=bool)
    obs_dir = _observation_dir(project_dir, observable=observable, setup_dir=setup_dir)

    for token in tokens:
        source_path = _source_path_from_token(obs_dir, token)
        with rasterio.open(source_path) as src:
            data = src.read(1).astype(float)
            valid_src = _observation_valid_pixels(
                data=data,
                nodata=src.nodata,
                observable=observable,
                project_dir=project_dir,
            ).astype(np.uint8)
            dst = np.zeros(eligible_mask.shape, dtype=np.uint8)
            reproject(
                source=valid_src,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=spec.transform,
                dst_crs=spec.crs,
                resampling=Resampling.nearest,
                src_nodata=0,
                dst_nodata=0,
            )
            support_mask |= dst.astype(bool)

    support_mask &= eligible_mask
    n_valid = int(np.count_nonzero(support_mask))
    n_eligible = int(np.count_nonzero(eligible_mask))
    coverage_ratio = float(n_valid) / float(max(1, n_eligible))
    return ObservationSupportMask(
        mask=support_mask,
        eligible_mask=eligible_mask,
        n_valid=n_valid,
        n_eligible=n_eligible,
        coverage_ratio=coverage_ratio,
    )


__all__ = ["ObservationSupportMask", "load_observation_support_mask"]

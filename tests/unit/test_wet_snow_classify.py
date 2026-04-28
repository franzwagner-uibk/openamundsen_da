from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from ruamel.yaml import YAML

from openamundsen_da.methods.wet_snow.classify import (
    CLASSIFICATION_METHOD_AMOUNT,
    CLASSIFICATION_METHOD_FRACTION,
    DEFAULT_LIQUID_WATER_AMOUNT_THRESHOLD_MM,
    DepthEntry,
    _compute_fraction,
    load_wet_snow_classification_config,
)
from openamundsen_da.methods.viz.maps import panel_renderers as map_panel_renderers


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = YAML()
    with path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


def _project_dir(tmp_path: Path, wet_snow_cfg: dict) -> Path:
    project_dir = tmp_path / "setup" / "projects" / "project_2024_2025"
    _write_yaml(project_dir / "project_2024_2025.yml", {"data_assimilation": {"wet_snow": wet_snow_cfg}})
    return project_dir


def _depth_entry(depth: np.ndarray) -> DepthEntry:
    return DepthEntry(
        stamp="2024-05-01T0000",
        data=depth.astype(np.float32),
        profile={
            "driver": "GTiff",
            "height": depth.shape[0],
            "width": depth.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:32632",
            "transform": from_origin(0.0, float(depth.shape[0]), 1.0, 1.0),
            "nodata": -9999.0,
        },
    )


def test_wet_snow_classification_config_defaults_to_fraction_method(tmp_path: Path) -> None:
    project_dir = _project_dir(tmp_path, {"classification_threshold_percent": 0.4})

    cfg = load_wet_snow_classification_config(project_dir)

    assert cfg.method == CLASSIFICATION_METHOD_FRACTION
    assert cfg.threshold_percent == pytest.approx(0.4)
    assert cfg.liquid_water_amount_threshold_mm == pytest.approx(DEFAULT_LIQUID_WATER_AMOUNT_THRESHOLD_MM)


def test_wet_snow_classification_config_loads_amount_method_default_threshold(tmp_path: Path) -> None:
    project_dir = _project_dir(tmp_path, {"classification_method": CLASSIFICATION_METHOD_AMOUNT})

    cfg = load_wet_snow_classification_config(project_dir)

    assert cfg.method == CLASSIFICATION_METHOD_AMOUNT
    assert np.isnan(cfg.threshold_percent)
    assert cfg.liquid_water_amount_threshold_mm == pytest.approx(5.0)


def test_wet_snow_map_probability_threshold_supports_amount_method(tmp_path: Path) -> None:
    project_dir = _project_dir(tmp_path, {"classification_method": CLASSIFICATION_METHOD_AMOUNT})

    assert map_panel_renderers._wet_snow_threshold_fraction(project_dir) == pytest.approx(0.5)


def test_wet_snow_classification_config_rejects_invalid_method(tmp_path: Path) -> None:
    project_dir = _project_dir(tmp_path, {"classification_method": "bogus"})

    with pytest.raises(ValueError, match="project.data_assimilation.wet_snow.classification_method"):
        load_wet_snow_classification_config(project_dir)


def test_wet_snow_classification_config_rejects_negative_amount_threshold(tmp_path: Path) -> None:
    project_dir = _project_dir(
        tmp_path,
        {
            "classification_method": CLASSIFICATION_METHOD_AMOUNT,
            "liquid_water_amount_threshold_mm": -1.0,
        },
    )

    with pytest.raises(ValueError, match="project.data_assimilation.wet_snow.liquid_water_amount_threshold_mm"):
        load_wet_snow_classification_config(project_dir)


def test_absolute_liquid_water_method_classifies_without_depth_normalization(tmp_path: Path) -> None:
    out_dir = tmp_path / "wet_snow"
    depth = np.array([[10.0, 1.0], [0.001, -9999.0]], dtype=np.float32)
    lw_total_mm = np.array([[6.0, 4.0], [10.0, 10.0]], dtype=np.float32)

    _compute_fraction(
        depth_entry=_depth_entry(depth),
        lw_arrays=[lw_total_mm],
        threshold_frac=0.005,
        classification_method=CLASSIFICATION_METHOD_AMOUNT,
        liquid_water_amount_threshold_mm=5.0,
        out_dir=out_dir,
        mask_prefix="wet_snow_mask",
        fraction_prefix="lwc_fraction",
        write_fraction=False,
        overwrite=True,
        rho_water=1000.0,
        min_depth_m=0.005,
    )

    with rasterio.open(out_dir / "wet_snow_mask_2024-05-01T0000.tif") as src:
        mask = src.read(1)

    assert mask.tolist() == [
        [1, 0],
        [0, 255],
    ]

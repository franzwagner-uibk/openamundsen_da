from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from ruamel.yaml import YAML

from openamundsen_da.methods.pf.assimilate_fraction import assimilate_wet_snow_line_for_date
from openamundsen_da.methods.pf.fraction_support import ObservationSupportMask
from openamundsen_da.methods.wet_snow.area import compute_member_wet_snow_line_daily, compute_model_wet_snow_line
from openamundsen_da.methods.wet_snow.wsl import compute_wet_snow_line_from_masks


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = YAML()
    with path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


def test_compute_wet_snow_line_from_masks_uses_downward_crossing_fraction() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        setup_dir = Path(tmp) / "setup"
        project_dir = setup_dir / "projects" / "project_2024_2025"
        grids_dir = setup_dir / "grids"
        grids_dir.mkdir(parents=True, exist_ok=True)
        transform = from_origin(0.0, 400.0, 100.0, 100.0)

        _write_yaml(
            setup_dir / "setup.yml",
            {
                "domain": "demo",
                "resolution": 100,
                "crs": "EPSG:25832",
                "input_data": {"grids": {"dir": "grids"}},
            },
        )
        _write_yaml(
            project_dir / "project_2024_2025.yml",
            {
                "data_assimilation": {
                    "wet_snow_line": {
                        "elevation_band_size_m": 100.0,
                        "smoothing_window_bands": 1,
                        "crossing_fraction": 0.5,
                        "wet_elevation_percentile": 95.0,
                        "aspect_diagnostics": "off",
                        "sector_relative_threshold": 0.8,
                    }
                }
            },
        )
        dem = np.array(
            [
                [120.0, 140.0, 220.0, 240.0],
                [120.0, 140.0, 220.0, 240.0],
                [320.0, 340.0, 320.0, 340.0],
                [320.0, 340.0, 320.0, 340.0],
            ],
            dtype=np.float32,
        )
        roi = np.ones_like(dem, dtype=np.uint8)
        with rasterio.open(
            grids_dir / "dem_demo_100.tif",
            "w",
            driver="GTiff",
            width=dem.shape[1],
            height=dem.shape[0],
            count=1,
            dtype="float32",
            crs="EPSG:25832",
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(dem, 1)
        with rasterio.open(
            grids_dir / "roi_demo_100.asc",
            "w",
            driver="AAIGrid",
            width=roi.shape[1],
            height=roi.shape[0],
            count=1,
            dtype="uint8",
            crs="EPSG:25832",
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(roi, 1)

        valid_mask = np.ones_like(dem, dtype=bool)
        wet_mask = np.zeros_like(dem, dtype=bool)
        wet_mask[:2, :] = True

        result = compute_wet_snow_line_from_masks(
            setup_dir=setup_dir,
            project_dir=project_dir,
            valid_mask=valid_mask,
            wet_mask=wet_mask,
        )

        assert result.wet_snow_line == 300.0
        assert result.wet_bands == 2
        assert result.gate_reason is None


def test_compute_wet_snow_line_from_masks_exposes_sector_relative_diagnostics() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        setup_dir = Path(tmp) / "setup"
        project_dir = setup_dir / "projects" / "project_2024_2025"
        grids_dir = setup_dir / "grids"
        grids_dir.mkdir(parents=True, exist_ok=True)
        transform = from_origin(0.0, 500.0, 100.0, 100.0)

        _write_yaml(
            setup_dir / "setup.yml",
            {
                "domain": "demo",
                "resolution": 100,
                "crs": "EPSG:25832",
                "input_data": {"grids": {"dir": "grids"}},
            },
        )
        _write_yaml(
            project_dir / "project_2024_2025.yml",
            {
                "data_assimilation": {
                    "wet_snow_line": {
                        "elevation_band_size_m": 100.0,
                        "smoothing_window_bands": 1,
                        "crossing_fraction": 0.5,
                        "wet_elevation_percentile": 95.0,
                        "aspect_diagnostics": "four_sectors",
                        "sector_relative_threshold": 0.8,
                    }
                }
            },
        )
        dem = np.array(
            [
                [120.0, 120.0],
                [220.0, 220.0],
                [320.0, 320.0],
                [420.0, 420.0],
                [520.0, 520.0],
            ],
            dtype=np.float32,
        )
        roi = np.ones_like(dem, dtype=np.uint8)
        with rasterio.open(
            grids_dir / "dem_demo_100.tif",
            "w",
            driver="GTiff",
            width=dem.shape[1],
            height=dem.shape[0],
            count=1,
            dtype="float32",
            crs="EPSG:25832",
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(dem, 1)
        with rasterio.open(
            grids_dir / "roi_demo_100.asc",
            "w",
            driver="AAIGrid",
            width=roi.shape[1],
            height=roi.shape[0],
            count=1,
            dtype="uint8",
            crs="EPSG:25832",
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(roi, 1)

        valid_mask = np.ones_like(dem, dtype=bool)
        wet_mask = np.array(
            [
                [False, False],
                [True, True],
                [True, True],
                [False, True],
                [False, False],
            ],
            dtype=bool,
        )
        aspect = np.zeros_like(dem, dtype=float)
        slope = np.ones_like(dem, dtype=float)

        with patch(
            "openamundsen_da.methods.wet_snow.wsl._load_dem_and_aspect",
            return_value=(dem, aspect, slope),
        ):
            result = compute_wet_snow_line_from_masks(
                setup_dir=setup_dir,
                project_dir=project_dir,
                valid_mask=valid_mask,
                wet_mask=wet_mask,
            )

        assert result.wet_snow_line == 450.0
        assert round(float(result.sector_relative_lines["N"]), 1) == 390.0
        assert result.sector_relative_lines["E"] is None
        assert result.sector_relative_profiles["N"].shape[0] == 5


def test_compute_wet_snow_line_from_masks_returns_no_crossing_for_fully_wet_profile() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        setup_dir = Path(tmp) / "setup"
        project_dir = setup_dir / "projects" / "project_2024_2025"
        grids_dir = setup_dir / "grids"
        grids_dir.mkdir(parents=True, exist_ok=True)
        transform = from_origin(0.0, 300.0, 100.0, 100.0)

        _write_yaml(
            setup_dir / "setup.yml",
            {
                "domain": "demo",
                "resolution": 100,
                "crs": "EPSG:25832",
                "input_data": {"grids": {"dir": "grids"}},
            },
        )
        _write_yaml(
            project_dir / "project_2024_2025.yml",
            {
                "data_assimilation": {
                    "wet_snow_line": {
                        "elevation_band_size_m": 100.0,
                        "smoothing_window_bands": 1,
                        "crossing_fraction": 0.5,
                        "wet_elevation_percentile": 95.0,
                        "aspect_diagnostics": "off",
                        "sector_relative_threshold": 0.8,
                    }
                }
            },
        )
        dem = np.array(
            [
                [120.0, 140.0, 220.0],
                [240.0, 320.0, 340.0],
            ],
            dtype=np.float32,
        )
        roi = np.ones_like(dem, dtype=np.uint8)
        with rasterio.open(
            grids_dir / "dem_demo_100.tif",
            "w",
            driver="GTiff",
            width=dem.shape[1],
            height=dem.shape[0],
            count=1,
            dtype="float32",
            crs="EPSG:25832",
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(dem, 1)
        with rasterio.open(
            grids_dir / "roi_demo_100.asc",
            "w",
            driver="AAIGrid",
            width=roi.shape[1],
            height=roi.shape[0],
            count=1,
            dtype="uint8",
            crs="EPSG:25832",
            transform=transform,
            nodata=0,
        ) as dst:
            dst.write(roi, 1)

        result = compute_wet_snow_line_from_masks(
            setup_dir=setup_dir,
            project_dir=project_dir,
            valid_mask=np.ones_like(dem, dtype=bool),
            wet_mask=np.ones_like(dem, dtype=bool),
        )

        assert result.wet_snow_line is None
        assert result.gate_reason == "no_crossing_fraction"


def test_assimilate_wet_snow_line_writes_uniform_weights_when_obs_gate_triggers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        setup_dir = Path(tmp) / "setup"
        project_dir = setup_dir / "projects" / "project_2024_2025"
        step_dir = project_dir / "steps" / "step_00_init"
        obs_dir = step_dir / "obs"
        member_dirs = [
            step_dir / "ensembles" / "prior" / "member_001",
            step_dir / "ensembles" / "prior" / "member_002",
        ]
        for member_dir in member_dirs:
            (member_dir / "results").mkdir(parents=True, exist_ok=True)
        obs_dir.mkdir(parents=True, exist_ok=True)

        _write_yaml(
            project_dir / "project_2024_2025.yml",
            {
                "obs": {
                    "wetsnow": {"product_tag": "SWS"},
                },
                "data_assimilation": {
                        "likelihood": {
                            "wet_snow_line": {
                                "obs_sigma": 150.0,
                                "use_binomial": False,
                                "sigma_floor": 25.0,
                            "min_sigma": 25.0,
                            "min_support_coverage_ratio": 0.10,
                            "min_wet_pixels_total": 50,
                            "min_wet_bands": 1,
                        }
                    }
                },
            },
        )

        obs_csv = obs_dir / "obs_wet_snow_line_SWS_20240511.csv"
        pd.DataFrame(
            [
                {
                    "date": "2024-05-11",
                    "wet_snow_line": np.nan,
                    "wet_snow_line_n_wet": 10,
                    "wet_snow_line_wet_bands": 0,
                    "wet_snow_line_gate_reason": "no_crossing_fraction",
                }
            ]
        ).to_csv(obs_csv, index=False)

        with (
            patch(
                "openamundsen_da.methods.pf.assimilate_fraction.load_observation_support_mask",
                return_value=ObservationSupportMask(
                    mask=np.ones((2, 2), dtype=bool),
                    eligible_mask=np.ones((2, 2), dtype=bool),
                    n_valid=4,
                    n_eligible=4,
                    coverage_ratio=1.0,
                ),
            ),
            patch(
                "openamundsen_da.methods.pf.assimilate_fraction.compute_model_wet_snow_line",
                side_effect=[
                    {
                        "wet_snow_line": 2400.0,
                        "wet_snow_line_full_roi": 2400.0,
                        "n_valid": 4,
                        "n_valid_full_roi": 4,
                        "wet_bands": 0,
                        "wet_snow_line_gate_reason": "no_qualifying_band",
                    },
                    {
                        "wet_snow_line": 2500.0,
                        "wet_snow_line_full_roi": 2500.0,
                        "n_valid": 4,
                        "n_valid_full_roi": 4,
                        "wet_bands": 0,
                        "wet_snow_line_gate_reason": "no_qualifying_band",
                    },
                ],
            ),
            patch(
                "openamundsen_da.methods.pf.assimilate_fraction.resolve_landcover_mask",
                return_value=None,
            ),
        ):
            df = assimilate_wet_snow_line_for_date(
                setup_dir=setup_dir,
                step_dir=step_dir,
                ensemble="prior",
                date=datetime(2024, 5, 11),
                aoi=Path(tmp) / "roi.gpkg",
                landcover_cfg=None,
                obs_csv=None,
                product="SWS",
            )

        assert list(df["weight"]) == [0.5, 0.5]
        assert df["sigma"].isna().all()
        assert df["wet_information_gate_triggered"].all()
        assert df["wet_information_gate_reason"].iloc[0] == "no_crossing_fraction"


def _run_wet_snow_line_assimilation_with_model_values(
    *,
    model_values: list[float | None],
    min_model_finite_fraction: float | None = None,
    ess_threshold_ratio: float | None = None,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        setup_dir = Path(tmp) / "setup"
        project_dir = setup_dir / "projects" / "project_2024_2025"
        step_dir = project_dir / "steps" / "step_00_init"
        obs_dir = step_dir / "obs"
        obs_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(1, len(model_values) + 1):
            (step_dir / "ensembles" / "prior" / f"member_{idx:03d}" / "results").mkdir(
                parents=True,
                exist_ok=True,
            )

        wet_snow_line_likelihood: dict[str, object] = {
            "obs_sigma": 150.0,
            "use_binomial": False,
            "sigma_floor": 25.0,
            "min_sigma": 25.0,
            "min_support_coverage_ratio": 0.10,
            "min_wet_pixels_total": 50,
            "min_wet_bands": 1,
        }
        if min_model_finite_fraction is not None:
            wet_snow_line_likelihood["min_model_finite_fraction"] = min_model_finite_fraction
        da_cfg: dict[str, object] = {
            "likelihood": {
                "wet_snow_line": wet_snow_line_likelihood,
            }
        }
        if ess_threshold_ratio is not None:
            da_cfg["resampling"] = {"ess_threshold_ratio": ess_threshold_ratio}
        _write_yaml(
            project_dir / "project_2024_2025.yml",
            {
                "obs": {"wetsnow": {"product_tag": "SWS"}},
                "data_assimilation": da_cfg,
            },
        )

        pd.DataFrame(
            [
                {
                    "date": "2024-05-11",
                    "wet_snow_line": 2400.0,
                    "wet_snow_line_n_wet": 100,
                    "wet_snow_line_wet_bands": 3,
                    "wet_snow_line_gate_reason": "",
                }
            ]
        ).to_csv(obs_dir / "obs_wet_snow_line_SWS_20240511.csv", index=False)

        side_effect = [
            {
                "wet_snow_line": value,
                "wet_snow_line_full_roi": value,
                "n_valid": 4,
                "n_valid_full_roi": 4,
                "wet_bands": 3 if value is not None else 0,
                "wet_snow_line_gate_reason": "" if value is not None else "no_crossing_fraction",
            }
            for value in model_values
        ]
        with (
            patch(
                "openamundsen_da.methods.pf.assimilate_fraction.load_observation_support_mask",
                return_value=ObservationSupportMask(
                    mask=np.ones((2, 2), dtype=bool),
                    eligible_mask=np.ones((2, 2), dtype=bool),
                    n_valid=4,
                    n_eligible=4,
                    coverage_ratio=1.0,
                ),
            ),
            patch(
                "openamundsen_da.methods.pf.assimilate_fraction.compute_model_wet_snow_line",
                side_effect=side_effect,
            ),
            patch(
                "openamundsen_da.methods.pf.assimilate_fraction.resolve_landcover_mask",
                return_value=None,
            ),
        ):
            return assimilate_wet_snow_line_for_date(
                setup_dir=setup_dir,
                step_dir=step_dir,
                ensemble="prior",
                date=datetime(2024, 5, 11),
                aoi=Path(tmp) / "roi.gpkg",
                landcover_cfg=None,
                obs_csv=None,
                product="SWS",
            )


def test_assimilate_wet_snow_line_default_requires_all_members_finite() -> None:
    df = _run_wet_snow_line_assimilation_with_model_values(model_values=[2400.0, None])

    assert list(df["weight"]) == [0.5, 0.5]
    assert df["sigma"].isna().all()
    assert df["model_gate_triggered"].all()
    assert df["model_gate_reason"].iloc[0].startswith("model_finite_fraction<1.0000")
    assert float(df["model_finite_fraction"].iloc[0]) == 0.5
    assert float(df["model_finite_fraction_threshold"].iloc[0]) == 1.0


def test_assimilate_wet_snow_line_allows_small_missing_member_fraction_when_configured() -> None:
    model_values: list[float | None] = [2400.0] * 22 + [None]

    df = _run_wet_snow_line_assimilation_with_model_values(
        model_values=model_values,
        min_model_finite_fraction=0.90,
        ess_threshold_ratio=0.99,
    )

    assert not df["model_gate_triggered"].any()
    assert not df["sigma"].isna().any()
    assert float(df["model_finite_fraction"].iloc[0]) == 22 / 23
    assert float(df["model_finite_fraction_threshold"].iloc[0]) == 0.90
    missing = df[df["value_model"].isna()].iloc[0]
    assert float(missing["log_weight"]) == -1.0e12
    assert float(missing["weight"]) == 0.0
    assert np.isclose(float(df["weight"].sum()), 1.0)
    assert np.isclose(float(df["ess"].iloc[0]), 22.0)
    assert bool(df["ess_below_threshold"].iloc[0])


def test_assimilate_wet_snow_line_noops_when_model_finite_fraction_below_threshold() -> None:
    df = _run_wet_snow_line_assimilation_with_model_values(
        model_values=[2400.0, 2400.0, None, None],
        min_model_finite_fraction=0.90,
    )

    assert list(df["weight"]) == [0.25, 0.25, 0.25, 0.25]
    assert df["sigma"].isna().all()
    assert df["model_gate_triggered"].all()
    assert df["model_gate_reason"].iloc[0].startswith("model_finite_fraction<0.9000")
    assert float(df["model_finite_member_count"].iloc[0]) == 2
    assert float(df["model_member_count"].iloc[0]) == 4


def test_assimilate_wet_snow_line_noops_when_no_model_members_have_finite_wsla() -> None:
    df = _run_wet_snow_line_assimilation_with_model_values(
        model_values=[None, None],
        min_model_finite_fraction=0.0,
    )

    assert list(df["weight"]) == [0.5, 0.5]
    assert df["sigma"].isna().all()
    assert df["model_gate_triggered"].all()
    assert df["model_gate_reason"].iloc[0].startswith("model_no_finite_wet_snow_line")


def test_compute_member_wet_snow_line_daily_uses_full_roi_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        setup_dir = Path(tmp) / "setup"
        project_dir = setup_dir / "projects" / "project_2024_2025"
        results_dir = project_dir / "steps" / "step_00" / "ensembles" / "prior" / "member_001" / "results"
        rows = [
            {
                "wet_snow_line_full_roi": 2400.0,
                "wet_snow_line_p95_full_roi": 2450.0,
                "n_valid_full_roi": 80,
                "n_wet_full_roi": 32,
                "wet_bands_full_roi": 2,
                "wet_snow_line_gate_reason_full_roi": "",
            },
            {
                "wet_snow_line_full_roi": 2500.0,
                "wet_snow_line_p95_full_roi": 2550.0,
                "n_valid_full_roi": 82,
                "n_wet_full_roi": 34,
                "wet_bands_full_roi": 3,
                "wet_snow_line_gate_reason_full_roi": "",
            },
        ]

        with patch(
            "openamundsen_da.methods.wet_snow.area.compute_model_wet_snow_line",
            side_effect=rows,
        ):
            df = compute_member_wet_snow_line_daily(
                setup_dir=setup_dir,
                project_dir=project_dir,
                results_dir=results_dir,
                aoi_path=Path(tmp) / "roi.gpkg",
                landcover_cfg=object(),
                start=datetime(2024, 5, 1),
                end=datetime(2024, 5, 2),
            )

        assert list(df["wet_snow_line"]) == [2400.0, 2500.0]
        assert list(df["wet_snow_line_p95"]) == [2450.0, 2550.0]
        assert list(df["wet_bands"]) == [2, 3]


def test_compute_model_wet_snow_line_uses_unmasked_full_roi_branch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        setup_dir = Path(tmp) / "setup"
        project_dir = setup_dir / "projects" / "project_2024_2025"
        results_dir = project_dir / "steps" / "step_00" / "ensembles" / "prior" / "member_001" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        arr_full = np.ma.array(
            np.array(
                [
                    [1, 1],
                    [0, 0],
                ],
                dtype=np.uint8,
            ),
            mask=np.zeros((2, 2), dtype=bool),
        )
        arr_support = np.ma.array(
            np.array(
                [
                    [1, 1],
                    [0, 0],
                ],
                dtype=np.uint8,
            ),
            mask=np.array(
                [
                    [False, False],
                    [True, True],
                ],
                dtype=bool,
            ),
        )

        with (
            patch(
                "openamundsen_da.methods.wet_snow.area._find_mask_raster",
                return_value=results_dir / "wet_snow" / "wet_snow_mask_2024-05-01T0000.tif",
            ),
            patch(
                "openamundsen_da.methods.wet_snow.area._read_mask_full_grid",
                side_effect=[
                    (arr_full, np.ones((2, 2), dtype=bool), None, None, "", None, None),
                    (arr_support, np.ones((2, 2), dtype=bool), None, None, "", None, None),
                ],
            ),
            patch(
                "openamundsen_da.methods.wet_snow.area.compute_wet_snow_line_from_masks",
                side_effect=[
                    SimpleNamespace(
                        wet_snow_line=150.0,
                        wet_elevation_percentile=175.0,
                        n_valid=4,
                        n_wet=2,
                        wet_bands=2,
                        gate_reason=None,
                        sector_relative_lines={},
                        sector_relative_profiles={},
                        profile=pd.DataFrame(),
                    ),
                    SimpleNamespace(
                        wet_snow_line=None,
                        wet_elevation_percentile=125.0,
                        n_valid=2,
                        n_wet=2,
                        wet_bands=1,
                        gate_reason="no_crossing_fraction",
                        sector_relative_lines={},
                        sector_relative_profiles={},
                        profile=pd.DataFrame(),
                    ),
                ],
            ),
        ):
            out = compute_model_wet_snow_line(
                setup_dir=setup_dir,
                project_dir=project_dir,
                results_dir=results_dir,
                aoi_path=Path(tmp) / "roi.gpkg",
                landcover_cfg=object(),
                date=datetime(2024, 5, 1),
            )

        assert out["wet_snow_line_full_roi"] == 150.0
        assert out["wet_snow_line"] is None
        assert out["wet_snow_line_gate_reason"] == "no_crossing_fraction"

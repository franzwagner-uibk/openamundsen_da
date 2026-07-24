from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from pyproj import CRS
from ruamel.yaml import YAML
from shapely.geometry import box

from openamundsen_da.observer.snowcover import (
    SnowcoverClasses,
    _build_stats_row,
    _load_uncertainty_ingest_config,
    _normalize_netcdf_times,
    summarize_snowcover_directory,
)
from openamundsen_da.util.landcover_mask import LandcoverMaskConfig


def _write_project_yaml(project_dir: Path, payload: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    yml = project_dir / f"{project_dir.name}.yml"
    y = YAML()
    with yml.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


def _write_eurac_style_scf_netcdf(path: Path) -> None:
    ds = xr.Dataset(
        {
            "fsc": (
                ("band", "y", "x"),
                np.array([[[100.0, 205.0], [255.0, 0.0]]], dtype=np.float32),
            ),
            "uncertainty": (
                ("band", "y", "x"),
                np.array(
                    [[[10.12345, 99.0], [99.0, 40.98765]]],
                    dtype=np.float32,
                ),
            ),
        },
        coords={
            "band": [1],
            "x": np.array([0.5, 1.5], dtype=np.float32),
            "y": np.array([1.5, 0.5], dtype=np.float32),
            "time": [np.datetime64("2024-04-01T00:00:00")],
        },
    )
    ds["spatial_ref"] = xr.DataArray(0)
    ds["spatial_ref"].attrs["crs_wkt"] = CRS.from_epsg(4326).to_wkt()
    ds["spatial_ref"].attrs["spatial_ref"] = CRS.from_epsg(4326).to_wkt()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


class SnowcoverUncertaintyTests(unittest.TestCase):
    def test_ingest_config_requires_netcdf_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "data_assimilation": {
                        "uncertainty": {
                            "scf": {
                                "enabled": True,
                                "ingest": {"uncertainty_variable": "uncertainty", "time_variable": "time"},
                            }
                        }
                    }
                },
            )
            with self.assertRaises(ValueError) as ctx:
                _load_uncertainty_ingest_config(project_dir)
            self.assertIn("scf_variable", str(ctx.exception))

    def test_ingest_config_reads_variable_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "data_assimilation": {
                        "uncertainty": {
                            "scf": {
                                "enabled": True,
                                "ingest": {
                                    "scf_variable": "fsc",
                                    "uncertainty_variable": "uncertainty",
                                    "time_variable": "time",
                                },
                            }
                        }
                    }
                },
            )
            cfg = _load_uncertainty_ingest_config(project_dir)
            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.scf_variable, "fsc")
            self.assertEqual(cfg.uncertainty_variable, "uncertainty")
            self.assertEqual(cfg.time_variable, "time")
            self.assertEqual(cfg.uncertainty_source, "product")
            self.assertIsNone(cfg.internal_config)

    def test_ingest_config_reads_internal_uncertainty_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "obs": {
                        "snowcover": {
                            "dir": "obs/snowcover",
                            "classes": {
                                "valid": [0, 50, 100],
                                "cloud": [205],
                                "water": [210],
                                "nodata": [255],
                            },
                        }
                    },
                    "data_assimilation": {
                        "uncertainty": {
                            "scf": {
                                "enabled": True,
                                "ingest": {
                                    "scf_variable": "fsc",
                                    "time_variable": "time",
                                    "uncertainty_source": "internal",
                                },
                                "u_min": 5.0,
                                "u_max": 20.0,
                                "penalties": [
                                    {
                                        "name": "forest",
                                        "source": "landcover",
                                        "classes": [8, 9, 10, 11, 12],
                                        "penalty": 20.0,
                                    }
                                ],
                            }
                        }
                    },
                },
            )

            cfg = _load_uncertainty_ingest_config(project_dir)

            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.scf_variable, "fsc")
            self.assertIsNone(cfg.uncertainty_variable)
            self.assertEqual(cfg.time_variable, "time")
            self.assertEqual(cfg.uncertainty_source, "internal")
            self.assertIsNotNone(cfg.internal_config)
            assert cfg.internal_config is not None
            self.assertAlmostEqual(cfg.internal_config.u_min, 5.0, places=6)
            self.assertAlmostEqual(cfg.internal_config.u_max, 20.0, places=6)
            self.assertEqual(cfg.internal_config.penalties[0].classes, (8, 9, 10, 11, 12))

    def test_ingest_config_internal_rejects_product_uncertainty_variable(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "obs": {
                        "snowcover": {
                            "dir": "obs/snowcover",
                            "classes": {
                                "valid": [0, 50, 100],
                                "cloud": [205],
                                "water": [],
                                "nodata": [],
                            },
                        }
                    },
                    "data_assimilation": {
                        "uncertainty": {
                            "scf": {
                                "enabled": True,
                                "ingest": {
                                    "scf_variable": "fsc",
                                    "uncertainty_variable": "uncertainty",
                                    "time_variable": "time",
                                    "uncertainty_source": "internal",
                                },
                                "u_min": 5.0,
                                "u_max": 20.0,
                                "penalties": [
                                    {"source": "fsc", "classes": [50], "penalty": 20.0},
                                ],
                            }
                        }
                    },
                },
            )

            with self.assertRaises(ValueError) as ctx:
                _load_uncertainty_ingest_config(project_dir)
            self.assertIn("uncertainty_variable must not be set", str(ctx.exception))

    def test_duplicate_netcdf_days_raise(self):
        with self.assertRaises(ValueError) as ctx:
            _normalize_netcdf_times(
                ["2024-04-01T01:00:00Z", "2024-04-01T23:00:00Z"],
                source_name="obs.nc",
            )
        self.assertIn("multiple timesteps map to the same day", str(ctx.exception))

    def test_cloud_pixels_are_excluded_from_uncertainty_aggregation(self):
        classes = SnowcoverClasses(
            valid=[0, 50, 100],
            cloud=[205],
            water=[],
            nodata=[],
        )
        data = np.array([[50, 205], [0, 100]], dtype=np.float32)
        mask = np.zeros_like(data, dtype=bool)
        unc = np.array([[60, 99], [10, 20]], dtype=np.float32)
        unc_mask = np.zeros_like(data, dtype=bool)

        row = _build_stats_row(
            date_key="2024-04-01",
            region_id="roi",
            tile="UNKNOWN",
            source_name="test",
            data=data,
            mask=mask,
            nodata=None,
            roi_mask=np.ones_like(data, dtype=bool),
            source_mask=np.zeros_like(data, dtype=bool),
            classes=classes,
            unc_data=unc,
            unc_mask=unc_mask,
            unc_nodata=None,
            require_uncertainty=True,
        )
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["n_valid"], 3)
        self.assertEqual(row["n_cloud"], 1)
        self.assertEqual(row["n_invalid"], 1)
        self.assertAlmostEqual(float(row["invalid_fraction"]), 0.25, places=6)
        self.assertAlmostEqual(float(row["unc_mean"]), 30.0, places=6)
        self.assertEqual(int(row["unc_n_valid"]), 3)

    def test_invalid_fraction_counts_nan_nodata_pixels(self):
        classes = SnowcoverClasses(
            valid=[0, 50, 100],
            cloud=[205],
            water=[210],
            nodata=[255],
        )
        data = np.array([[50, np.nan], [255, 100]], dtype=np.float32)
        mask = np.zeros_like(data, dtype=bool)

        row = _build_stats_row(
            date_key="2024-04-01",
            region_id="roi",
            tile="UNKNOWN",
            source_name="test",
            data=data,
            mask=mask,
            nodata=np.nan,
            roi_mask=np.ones_like(data, dtype=bool),
            source_mask=np.zeros_like(data, dtype=bool),
            classes=classes,
            require_uncertainty=False,
        )

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["n_valid"], 2)
        self.assertEqual(row["n_cloud"], 0)
        self.assertEqual(row["n_invalid"], 2)
        self.assertAlmostEqual(float(row["cloud_fraction"]), 0.0, places=6)
        self.assertAlmostEqual(float(row["invalid_fraction"]), 0.5, places=6)

    def test_summarize_snowcover_directory_reads_eurac_style_netcdf_crs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_dir = root / "setup"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            input_dir = setup_dir / "obs" / "snowcover"
            roi_path = setup_dir / "env" / "roi.gpkg"
            _write_project_yaml(
                project_dir,
                {
                    "obs": {
                        "snowcover": {
                            "dir": "obs/snowcover",
                            "classes": {
                                "valid": [0, 100],
                                "cloud": [205, 255],
                                "water": [210],
                                "nodata": [215],
                            },
                        }
                    },
                    "data_assimilation": {
                        "uncertainty": {
                            "scf": {
                                "enabled": True,
                                "ingest": {
                                    "scf_variable": "fsc",
                                    "uncertainty_variable": "uncertainty",
                                    "time_variable": "time",
                                },
                            }
                        }
                    },
                },
            )
            gdf = gpd.GeoDataFrame(
                {"id": [1]},
                geometry=[box(0.0, 0.0, 2.0, 2.0)],
                crs="EPSG:4326",
            )
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_file(roi_path, driver="GPKG")
            _write_eurac_style_scf_netcdf(input_dir / "SnowFLAKES_20240401_v3_eurac.nc")

            outputs = summarize_snowcover_directory(
                setup_dir=setup_dir,
                input_dir=input_dir,
                aoi=roi_path,
                project_label="project_2024_2025",
                output_root=setup_dir / "obs" / "summaries",
                landcover_cfg=LandcoverMaskConfig(
                    enabled=False,
                    path=None,
                    classes=tuple(),
                    project_crs=CRS.from_epsg(4326),
                ),
            )

            self.assertEqual(outputs, [input_dir / "SnowFLAKES_20240401_v3_eurac.nc"])
            df = np.genfromtxt(
                setup_dir / "obs" / "summaries" / "project_2024_2025" / "scf_summary.csv",
                delimiter=",",
                names=True,
                dtype=None,
                encoding="utf-8",
            )
            self.assertEqual(int(df["n_valid"]), 2)
            self.assertEqual(int(df["n_cloud"]), 2)
            self.assertAlmostEqual(float(df["cloud_fraction"]), 0.5, places=6)
            self.assertAlmostEqual(float(df["scf"]), 0.5, places=6)
            expected_unc_mean = float(
                np.mean(np.array([10.12345, 40.98765], dtype=np.float32))
            )
            self.assertAlmostEqual(float(df["unc_mean"]), expected_unc_mean, places=6)
            self.assertNotEqual(float(df["unc_mean"]), round(expected_unc_mean, 3))
            self.assertEqual(float(df["unc_min"]), 10.123)
            self.assertEqual(float(df["unc_max"]), 40.988)
            self.assertEqual(str(df["source"]), "SnowFLAKES_20240401_v3_eurac.nc@2024-04-01T00:00:00Z")

    def test_out_of_range_uncertainty_raises(self):
        classes = SnowcoverClasses(
            valid=[0, 50, 100],
            cloud=[205],
            water=[],
            nodata=[],
        )
        data = np.array([[50, 205], [0, 100]], dtype=np.float32)
        mask = np.zeros_like(data, dtype=bool)
        unc = np.array([[120, 99], [10, 20]], dtype=np.float32)
        unc_mask = np.zeros_like(data, dtype=bool)

        with self.assertRaises(ValueError) as ctx:
            _build_stats_row(
                date_key="2024-04-01",
                region_id="roi",
                tile="UNKNOWN",
                source_name="test",
                data=data,
                mask=mask,
                nodata=None,
                roi_mask=np.ones_like(data, dtype=bool),
                source_mask=np.zeros_like(data, dtype=bool),
                classes=classes,
                unc_data=unc,
                unc_mask=unc_mask,
                unc_nodata=None,
                require_uncertainty=True,
            )
        self.assertIn("out of [0,100]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

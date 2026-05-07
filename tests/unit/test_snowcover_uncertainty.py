from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

from openamundsen_da.observer.snowcover import (
    SnowcoverClasses,
    _build_stats_row,
    _load_uncertainty_ingest_config,
    _normalize_netcdf_times,
)


def _write_project_yaml(project_dir: Path, payload: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    yml = project_dir / f"{project_dir.name}.yml"
    y = YAML()
    with yml.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


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

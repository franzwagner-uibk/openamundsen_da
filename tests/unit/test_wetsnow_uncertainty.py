from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

from openamundsen_da.methods.wet_snow.area import (
    _build_wetsnow_summary_row,
    _load_wet_snow_uncertainty_ingest_config,
)


def _write_project_yaml(project_dir: Path, payload: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    yml = project_dir / f"{project_dir.name}.yml"
    y = YAML()
    with yml.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


class WetSnowUncertaintyTests(unittest.TestCase):
    def test_ingest_config_requires_netcdf_variables(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "data_assimilation": {
                        "uncertainty": {
                            "wet_snow": {
                                "enabled": True,
                                "ingest": {"uncertainty_variable": "uncertainty", "time_variable": "time"},
                            }
                        }
                    }
                },
            )
            with self.assertRaises(ValueError) as ctx:
                _load_wet_snow_uncertainty_ingest_config(project_dir)
            self.assertIn("wet_snow_variable", str(ctx.exception))

    def test_ingest_config_reads_variable_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "data_assimilation": {
                        "uncertainty": {
                            "wet_snow": {
                                "enabled": True,
                                "ingest": {
                                    "wet_snow_variable": "wet_snow",
                                    "uncertainty_variable": "uncertainty",
                                    "time_variable": "time",
                                },
                            }
                        }
                    }
                },
            )
            cfg = _load_wet_snow_uncertainty_ingest_config(project_dir)
            self.assertTrue(cfg.enabled)
            self.assertEqual(cfg.wet_snow_variable, "wet_snow")
            self.assertEqual(cfg.uncertainty_variable, "uncertainty")
            self.assertEqual(cfg.time_variable, "time")

    def test_uncertainty_aggregation_uses_valid_non_excluded_pixels(self):
        arr = np.ma.array(
            np.array(
                [
                    [110, 125],
                    [200, 210],
                ],
                dtype=np.float32,
            ),
            mask=np.zeros((2, 2), dtype=bool),
        )
        unc = np.ma.array(
            np.array(
                [
                    [40, 50],
                    [99, 99],
                ],
                dtype=np.float32,
            ),
            mask=np.zeros((2, 2), dtype=bool),
        )
        row = _build_wetsnow_summary_row(
            date_key="2024-04-01",
            region_id="roi",
            tile="UNKNOWN",
            source_name="wet.tif",
            arr=arr,
            wet_values=[110],
            valid_values=[110, 125, 200, 210],
            exclude_values=[200, 210],
            unc_arr=unc,
            unc_nodata=None,
            require_uncertainty=True,
        )
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["n_valid"], 2)
        self.assertEqual(row["n_wet"], 1)
        self.assertAlmostEqual(float(row["unc_mean"]), 45.0, places=6)
        self.assertEqual(int(row["unc_n_valid"]), 2)

    def test_out_of_range_uncertainty_raises(self):
        arr = np.ma.array(
            np.array([[110, 125]], dtype=np.float32),
            mask=np.zeros((1, 2), dtype=bool),
        )
        unc = np.ma.array(
            np.array([[101, 20]], dtype=np.float32),
            mask=np.zeros((1, 2), dtype=bool),
        )
        with self.assertRaises(ValueError) as ctx:
            _build_wetsnow_summary_row(
                date_key="2024-04-01",
                region_id="roi",
                tile="UNKNOWN",
                source_name="wet.tif",
                arr=arr,
                wet_values=[110],
                valid_values=[110, 125],
                exclude_values=[],
                unc_arr=unc,
                unc_nodata=None,
                require_uncertainty=True,
            )
        self.assertIn("out of [0,100]", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

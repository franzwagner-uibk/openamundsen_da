from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd
from ruamel.yaml import YAML

from openamundsen_da.methods.pf.assimilate_station import (
    assimilate_station_hs_for_date,
    assimilate_station_swe_for_date,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = YAML()
    with path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


def _write_series(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_project_config(project_dir: Path) -> None:
    _write_yaml(
        project_dir / f"{project_dir.name}.yml",
        {
            "obs": {"stations": {"dir": "obs/stations"}},
            "data_assimilation": {
                "station": {
                    "default_station_uncertainty_pct": 25,
                    "min_station_uncertainty_pct": 10,
                    "hs_sigma_abs_min": 0.10,
                    "swe_sigma_abs_min": 20.0,
                    "single_station_factor": 2.0,
                }
            },
        },
    )


class AssimilateStationTests(unittest.TestCase):
    def test_station_hs_assimilation_combines_multiple_stations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_dir = root / "setup_root"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            step_dir = project_dir / "steps" / "step_00_init"
            obs_dir = setup_dir / "obs" / "stations"
            prior_root = step_dir / "ensembles" / "prior"

            _write_project_config(project_dir)
            _write_series(
                obs_dir / "stations_da_metadata.csv",
                [
                    {"station_id": "station_a", "station_uncertainty_pct": 10},
                    {"station_id": "station_b", "station_uncertainty_pct": 50},
                ],
            )
            _write_series(
                obs_dir / "station_a.csv",
                [{"time": "2024-01-03 00:00:00", "snow_depth": 1.0, "swe": 100.0}],
            )
            _write_series(
                obs_dir / "station_b.csv",
                [{"time": "2024-01-03 00:00:00", "snow_depth": 0.5, "swe": 40.0}],
            )
            _write_series(
                obs_dir / "station_c.csv",
                [{"time": "2024-01-03 00:00:00", "snow_depth": 0.2, "swe": 20.0}],
            )

            _write_series(
                prior_root / "member_001" / "results" / "point_station_a.csv",
                [{"time": "2024-01-03 00:00:00", "snow_depth": 1.02, "swe": 101.0}],
            )
            _write_series(
                prior_root / "member_001" / "results" / "point_station_b.csv",
                [{"time": "2024-01-03 00:00:00", "snow_depth": 0.55, "swe": 39.0}],
            )
            _write_series(
                prior_root / "member_001" / "results" / "point_station_c.csv",
                [{"time": "2024-01-03 00:00:00", "snow_depth": 0.18, "swe": 18.0}],
            )
            _write_series(
                prior_root / "member_002" / "results" / "point_station_a.csv",
                [{"time": "2024-01-03 00:00:00", "snow_depth": 1.30, "swe": 130.0}],
            )
            _write_series(
                prior_root / "member_002" / "results" / "point_station_b.csv",
                [{"time": "2024-01-03 00:00:00", "snow_depth": 0.20, "swe": 10.0}],
            )
            _write_series(
                prior_root / "member_002" / "results" / "point_station_c.csv",
                [{"time": "2024-01-03 00:00:00", "snow_depth": 0.05, "swe": 5.0}],
            )

            result = assimilate_station_hs_for_date(
                setup_dir=setup_dir,
                step_dir=step_dir,
                ensemble="prior",
                date=datetime(2024, 1, 3),
            )

            self.assertEqual(set(result.weights["member_id"]), {"member_001", "member_002"})
            self.assertEqual(int(result.weights["n_stations"].iloc[0]), 3)
            self.assertAlmostEqual(float(result.weights["weight"].sum()), 1.0, places=6)
            weights = result.weights.set_index("member_id")["weight"]
            self.assertGreater(float(weights["member_001"]), float(weights["member_002"]))

            self.assertEqual(len(result.diagnostics), 6)
            self.assertIn("uncertainty_source", result.diagnostics.columns)
            station_c = result.diagnostics[result.diagnostics["station_id"] == "station_c"]
            self.assertTrue((station_c["uncertainty_source"] == "default").all())
            self.assertTrue((pd.to_numeric(station_c["sigma"], errors="coerce") >= 0.10).all())

    def test_single_station_case_inflates_sigma(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_dir = root / "setup_root"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            step_dir = project_dir / "steps" / "step_00_init"
            obs_dir = setup_dir / "obs" / "stations"
            prior_root = step_dir / "ensembles" / "prior"

            _write_project_config(project_dir)
            _write_series(
                obs_dir / "stations_da_metadata.csv",
                [{"station_id": "station_a", "station_uncertainty_pct": 10}],
            )
            _write_series(
                obs_dir / "station_a.csv",
                [{"time": "2024-02-01 00:00:00", "snow_depth": 1.0, "swe": 30.0}],
            )

            _write_series(
                prior_root / "member_001" / "results" / "point_station_a.csv",
                [{"time": "2024-02-01 00:00:00", "snow_depth": 1.05, "swe": 28.0}],
            )
            _write_series(
                prior_root / "member_002" / "results" / "point_station_a.csv",
                [{"time": "2024-02-01 00:00:00", "snow_depth": 0.80, "swe": 20.0}],
            )

            result = assimilate_station_hs_for_date(
                setup_dir=setup_dir,
                step_dir=step_dir,
                ensemble="prior",
                date=datetime(2024, 2, 1),
            )

            diag = result.diagnostics
            self.assertTrue(diag["single_station_inflated"].all())
            sigma_base = float(diag["sigma_base"].iloc[0])
            sigma = float(diag["sigma"].iloc[0])
            self.assertAlmostEqual(sigma_base, 0.10, places=6)
            self.assertAlmostEqual(sigma, 0.20, places=6)

    def test_station_swe_uses_absolute_sigma_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_dir = root / "setup_root"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            step_dir = project_dir / "steps" / "step_00_init"
            obs_dir = setup_dir / "obs" / "stations"
            prior_root = step_dir / "ensembles" / "prior"

            _write_project_config(project_dir)
            _write_series(
                obs_dir / "stations_da_metadata.csv",
                [{"station_id": "station_a", "station_uncertainty_pct": 10}],
            )
            _write_series(
                obs_dir / "station_a.csv",
                [{"time": "2024-03-01 00:00:00", "snow_depth": 0.1, "swe": 5.0}],
            )
            _write_series(
                prior_root / "member_001" / "results" / "point_station_a.csv",
                [{"time": "2024-03-01 00:00:00", "snow_depth": 0.1, "swe": 6.0}],
            )
            _write_series(
                prior_root / "member_002" / "results" / "point_station_a.csv",
                [{"time": "2024-03-01 00:00:00", "snow_depth": 0.1, "swe": 20.0}],
            )

            result = assimilate_station_swe_for_date(
                setup_dir=setup_dir,
                step_dir=step_dir,
                ensemble="prior",
                date=datetime(2024, 3, 1),
            )

            sigma_base = float(result.diagnostics["sigma_base"].iloc[0])
            self.assertAlmostEqual(sigma_base, 20.0, places=6)


if __name__ == "__main__":
    unittest.main()

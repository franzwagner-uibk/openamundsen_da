from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from ruamel.yaml import YAML

from openamundsen_da.observer.satellite_scf import (
    generate_project_from_summary as generate_scf_obs,
)
from openamundsen_da.observer.satellite_wet_snow_s1 import (
    generate_project_from_summary as generate_wet_obs,
)


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = YAML()
    with path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


def _write_step(step_dir: Path, *, start_date: str, end_date: str) -> None:
    _write_yaml(
        step_dir / f"{step_dir.name}.yml",
        {
            "start_date": start_date,
            "end_date": end_date,
        },
    )


class FractionObsPrepareTests(unittest.TestCase):
    def test_scf_summary_writes_product_tagged_obs(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            summary_csv = setup_dir / "obs" / project_dir.name / "scf_summary.csv"

            _write_yaml(
                setup_dir / "setup_root.yml",
                {
                    "obs": {
                        "snowcover": {"product_tag": "FSC"},
                        "wetsnow": {"product_tag": "SWS"},
                    }
                },
            )
            _write_yaml(
                project_dir / "project_2024_2025.yml",
                {
                    "obs": {
                        "snowcover": {"product_tag": "FSC"},
                        "wetsnow": {"product_tag": "SWS"},
                    },
                    "data_assimilation": {
                        "assimilation_events": [
                            {"date": "2024-12-15", "variable": "scf"},
                        ]
                    },
                },
            )
            _write_step(
                project_dir / "steps" / "step_00_init",
                start_date="2024-12-10 00:00:00",
                end_date="2024-12-20 21:00:00",
            )
            _write_step(
                project_dir / "steps" / "step_01_20241220-20250101",
                start_date="2024-12-21 00:00:00",
                end_date="2025-01-01 21:00:00",
            )

            summary_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"date": "2024-12-15", "scf": 0.42, "n_valid": 100}]).to_csv(summary_csv, index=False)

            generate_scf_obs(project_dir=project_dir, summary_csv=summary_csv, product=None, overwrite=True)

            out_csv = project_dir / "steps" / "step_00_init" / "obs" / "obs_scf_FSC_20241215.csv"
            self.assertTrue(out_csv.is_file())
            out_df = pd.read_csv(out_csv)
            self.assertEqual(float(out_df.iloc[0]["scf"]), 0.42)

    def test_wet_summary_writes_product_tagged_obs(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            summary_csv = setup_dir / "obs" / project_dir.name / "wet_snow_summary.csv"

            _write_yaml(
                setup_dir / "setup_root.yml",
                {
                    "obs": {
                        "snowcover": {"product_tag": "FSC"},
                        "wetsnow": {"product_tag": "SWS"},
                    }
                },
            )
            _write_yaml(
                project_dir / "project_2024_2025.yml",
                {
                    "obs": {
                        "snowcover": {"product_tag": "FSC"},
                        "wetsnow": {"product_tag": "SWS"},
                    },
                    "data_assimilation": {
                        "assimilation_events": [
                            {"date": "2025-03-05", "variable": "wet_snow"},
                        ]
                    },
                },
            )
            _write_step(
                project_dir / "steps" / "step_00_init",
                start_date="2025-03-01 06:00:00",
                end_date="2025-03-10 21:00:00",
            )
            _write_step(
                project_dir / "steps" / "step_01_20250310-20250320",
                start_date="2025-03-11 00:00:00",
                end_date="2025-03-20 21:00:00",
            )

            summary_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [{"date": "2025-03-05", "wet_snow_fraction": 0.18, "n_valid": 50}]
            ).to_csv(summary_csv, index=False)

            generate_wet_obs(project_dir=project_dir, summary_csv=summary_csv, product=None, overwrite=True)

            out_csv = project_dir / "steps" / "step_00_init" / "obs" / "obs_wet_snow_SWS_20250305.csv"
            self.assertTrue(out_csv.is_file())
            out_df = pd.read_csv(out_csv)
            self.assertEqual(float(out_df.iloc[0]["wet_snow_fraction"]), 0.18)


if __name__ == "__main__":
    unittest.main()

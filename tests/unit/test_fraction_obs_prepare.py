from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from ruamel.yaml import YAML

from openamundsen_da.observer.satellite_scf import (
    cli_main as scf_cli_main,
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
    def test_multiple_scenes_require_observation_time_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            summary_csv = setup_dir / "obs" / "summaries" / "scf_summary.csv"
            _write_yaml(setup_dir / "setup_root.yml", {"timestep": "3h", "timezone": 0})
            project_payload = {
                "obs": {"snowcover": {"dir": "obs/snowcover", "product_tag": "FSC"}},
                "data_assimilation": {
                    "assimilation_events": [{"date": "2024-12-15", "variable": "scf"}]
                },
            }
            project_yaml = project_dir / "project_2024_2025.yml"
            _write_yaml(project_yaml, project_payload)
            _write_step(
                project_dir / "steps" / "step_00_init",
                start_date="2024-12-15 00:00:00",
                end_date="2024-12-15 21:00:00",
            )
            _write_step(
                project_dir / "steps" / "step_01_next",
                start_date="2024-12-16 00:00:00",
                end_date="2024-12-20 21:00:00",
            )
            summary_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "date": "2024-12-15",
                        "acquisition_time": "2024-12-15T09:00:00Z",
                        "scf": 0.4,
                    },
                    {
                        "date": "2024-12-15",
                        "acquisition_time": "2024-12-15T12:00:00Z",
                        "scf": 0.6,
                    },
                ]
            ).to_csv(summary_csv, index=False)

            with self.assertRaisesRegex(ValueError, "Several scf observation scenes"):
                generate_scf_obs(
                    project_dir=project_dir,
                    summary_csv=summary_csv,
                    product=None,
                    overwrite=True,
                )

            project_payload["data_assimilation"]["assimilation_events"][0]["observation_time"] = (
                "2024-12-15T12:00:00Z"
            )
            _write_yaml(project_yaml, project_payload)
            generate_scf_obs(
                project_dir=project_dir,
                summary_csv=summary_csv,
                product=None,
                overwrite=True,
            )
            output = pd.read_csv(
                project_dir / "steps" / "step_00_init" / "obs" / "obs_scf_FSC_20241215.csv"
            )
            self.assertEqual(float(output.iloc[0]["scf"]), 0.6)
            self.assertEqual(output.iloc[0]["matched_model_time"], "2024-12-15T12:00:00")

    def test_scf_cli_records_summary_path_in_project_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2024_2025_ens23"
            summary_csv = setup_dir / "obs" / "summaries" / "project_2024_2025" / "scf_summary.csv"

            _write_yaml(
                setup_dir / "setup_root.yml",
                {"timestep": "3h", "timezone": 0},
            )
            _write_yaml(
                project_dir / "project_2024_2025_ens23.yml",
                {
                    "obs": {"snowcover": {"dir": "obs/snowcover", "product_tag": "FSC"}},
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

            rc = scf_cli_main(["--project-dir", str(project_dir), "--summary-csv", str(summary_csv), "--overwrite"])

            self.assertEqual(rc, 0)
            y = YAML(typ="safe")
            with (project_dir / "project_2024_2025_ens23.yml").open("r", encoding="utf-8") as f:
                data = y.load(f)
            self.assertEqual(data["obs"]["snowcover"]["summary_csv"], "obs/summaries/project_2024_2025/scf_summary.csv")

    def test_scf_summary_writes_product_tagged_obs(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            summary_csv = setup_dir / "obs" / project_dir.name / "scf_summary.csv"

            _write_yaml(
                setup_dir / "setup_root.yml",
                {
                    "timestep": "3h",
                    "timezone": 0,
                },
            )
            _write_yaml(
                project_dir / "project_2024_2025.yml",
                {
                    "obs": {
                        "snowcover": {"dir": "obs/snowcover", "product_tag": "FSC"},
                        "wetsnow": {"dir": "obs/wetsnow", "product_tag": "SWS"},
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
                    "timestep": "3h",
                    "timezone": 0,
                },
            )
            _write_yaml(
                project_dir / "project_2024_2025.yml",
                {
                    "obs": {
                        "snowcover": {"dir": "obs/snowcover", "product_tag": "FSC"},
                        "wetsnow": {"dir": "obs/wetsnow", "product_tag": "SWS"},
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

    def test_wet_summary_writes_wet_snow_line_obs(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            summary_csv = setup_dir / "obs" / project_dir.name / "wet_snow_summary.csv"

            _write_yaml(
                setup_dir / "setup_root.yml",
                {
                    "timestep": "3h",
                    "timezone": 0,
                },
            )
            _write_yaml(
                project_dir / "project_2024_2025.yml",
                {
                    "obs": {
                        "snowcover": {"dir": "obs/snowcover", "product_tag": "FSC"},
                        "wetsnow": {"dir": "obs/wetsnow", "product_tag": "SWS"},
                    },
                    "data_assimilation": {
                        "assimilation_events": [
                            {"date": "2025-03-05", "variable": "wet_snow_line"},
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
                [
                    {
                        "date": "2025-03-05",
                        "wet_snow_fraction": 0.18,
                        "wet_snow_line": 2450.0,
                        "wet_snow_line_n_wet": 120,
                        "wet_snow_line_wet_bands": 2,
                    }
                ]
            ).to_csv(summary_csv, index=False)

            generate_wet_obs(project_dir=project_dir, summary_csv=summary_csv, product=None, overwrite=True)

            out_csv = project_dir / "steps" / "step_00_init" / "obs" / "obs_wet_snow_line_SWS_20250305.csv"
            self.assertTrue(out_csv.is_file())
            out_df = pd.read_csv(out_csv)
            self.assertEqual(float(out_df.iloc[0]["wet_snow_line"]), 2450.0)


if __name__ == "__main__":
    unittest.main()

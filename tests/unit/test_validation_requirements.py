import tempfile
import unittest
from datetime import date
from pathlib import Path

from ruamel.yaml import YAML

from openamundsen_da.util.da_events import AssimilationEvent
from openamundsen_da.util.validation import validate_assimilation_requirements


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = YAML()
    with path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


class ValidateAssimilationRequirementsTests(unittest.TestCase):
    def test_missing_scf_output_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_dir = root / "setup_root"
            project_dir = setup_dir / "projects" / "project_2022_2023"
            step0 = project_dir / "steps" / "step_00_init"
            step1 = project_dir / "steps" / "step_01_a"

            _write_yaml(
                setup_dir / "setup_root.yml",
                {"output_data": {"grids": {"variables": []}}},
            )
            (step0 / "obs").mkdir(parents=True, exist_ok=True)
            (step1 / "obs").mkdir(parents=True, exist_ok=True)
            (step0 / "obs" / "obs_scf_SNOWCOVER_20221003.csv").write_text("date,scf\n2022-10-03,0.5\n", encoding="ascii")

            events = [AssimilationEvent(date=date(2022, 10, 3), variable="scf", product="SNOWCOVER")]
            with self.assertRaises(ValueError) as ctx:
                validate_assimilation_requirements(
                    setup_dir=setup_dir,
                    project_dir=project_dir,
                    steps=[step0, step1],
                    events=events,
                )
            self.assertIn("snow depth daily output", str(ctx.exception))

    def test_missing_obs_file_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_dir = root / "setup_root"
            project_dir = setup_dir / "projects" / "project_2022_2023"
            step0 = project_dir / "steps" / "step_00_init"
            step1 = project_dir / "steps" / "step_01_a"

            _write_yaml(
                setup_dir / "setup_root.yml",
                {
                    "output_data": {
                        "grids": {
                            "variables": [
                                {"var": "snow.depth", "name": "snowdepth_daily"},
                            ]
                        }
                    }
                },
            )
            (step0 / "obs").mkdir(parents=True, exist_ok=True)
            (step1 / "obs").mkdir(parents=True, exist_ok=True)

            events = [AssimilationEvent(date=date(2022, 10, 3), variable="scf", product="SNOWCOVER")]
            with self.assertRaises(ValueError) as ctx:
                validate_assimilation_requirements(
                    setup_dir=setup_dir,
                    project_dir=project_dir,
                    steps=[step0, step1],
                    events=events,
                )
            msg = str(ctx.exception)
            self.assertIn("missing obs CSV", msg)
            self.assertIn("obs_scf_SNOWCOVER_20221003.csv", msg)

    def test_passes_when_required_outputs_and_obs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_dir = root / "setup_root"
            project_dir = setup_dir / "projects" / "project_2022_2023"
            step0 = project_dir / "steps" / "step_00_init"
            step1 = project_dir / "steps" / "step_01_a"
            step2 = project_dir / "steps" / "step_02_b"

            _write_yaml(
                setup_dir / "setup_root.yml",
                {
                    "output_data": {
                        "grids": {
                            "variables": [
                                {"var": "snow.depth", "name": "snowdepth_daily"},
                                {"var": "snow.liquid_water_content", "name": "liquid_water_content"},
                            ]
                        }
                    }
                },
            )
            (step0 / "obs").mkdir(parents=True, exist_ok=True)
            (step1 / "obs").mkdir(parents=True, exist_ok=True)
            (step2 / "obs").mkdir(parents=True, exist_ok=True)
            (step0 / "obs" / "obs_scf_SNOWCOVER_20221003.csv").write_text("date,scf\n2022-10-03,0.5\n", encoding="ascii")
            (step1 / "obs" / "obs_wet_snow_WETSNOW_20221005.csv").write_text(
                "date,wet_snow_fraction\n2022-10-05,0.3\n",
                encoding="ascii",
            )

            events = [
                AssimilationEvent(date=date(2022, 10, 3), variable="scf", product="SNOWCOVER"),
                AssimilationEvent(date=date(2022, 10, 5), variable="wet_snow", product="WETSNOW"),
            ]
            validate_assimilation_requirements(
                setup_dir=setup_dir,
                project_dir=project_dir,
                steps=[step0, step1, step2],
                events=events,
            )

    def test_station_assimilation_requires_station_obs_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_dir = root / "setup_root"
            project_dir = setup_dir / "projects" / "project_2022_2023"
            step0 = project_dir / "steps" / "step_00_init"
            step1 = project_dir / "steps" / "step_01_a"

            _write_yaml(
                setup_dir / "setup_root.yml",
                {"output_data": {"grids": {"variables": []}}},
            )
            _write_yaml(
                project_dir / "project_2022_2023.yml",
                {
                    "obs": {"stations": {"dir": "obs/stations"}},
                    "data_assimilation": {
                        "station": {
                            "default_station_uncertainty_pct": 25,
                            "min_station_uncertainty_pct": 10,
                            "single_station_factor": 2.0,
                        }
                    },
                },
            )
            step0.mkdir(parents=True, exist_ok=True)
            step1.mkdir(parents=True, exist_ok=True)

            events = [AssimilationEvent(date=date(2022, 10, 3), variable="station_hs", product="STATION")]
            with self.assertRaises(ValueError) as ctx:
                validate_assimilation_requirements(
                    setup_dir=setup_dir,
                    project_dir=project_dir,
                    steps=[step0, step1],
                    events=events,
                )
            self.assertIn("Station assimilation requires observation directory", str(ctx.exception))

    def test_station_assimilation_passes_with_station_obs_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_dir = root / "setup_root"
            project_dir = setup_dir / "projects" / "project_2022_2023"
            step0 = project_dir / "steps" / "step_00_init"
            step1 = project_dir / "steps" / "step_01_a"

            _write_yaml(
                setup_dir / "setup_root.yml",
                {"output_data": {"grids": {"variables": []}}},
            )
            _write_yaml(
                project_dir / "project_2022_2023.yml",
                {
                    "obs": {"stations": {"dir": "obs/stations"}},
                    "data_assimilation": {
                        "station": {
                            "default_station_uncertainty_pct": 25,
                            "min_station_uncertainty_pct": 10,
                            "single_station_factor": 2.0,
                        }
                    },
                },
            )
            (setup_dir / "obs" / "stations").mkdir(parents=True, exist_ok=True)
            (setup_dir / "obs" / "stations" / "station_1.csv").write_text(
                "time,snow_depth\n2022-10-03 00:00:00,0.4\n",
                encoding="ascii",
            )
            step0.mkdir(parents=True, exist_ok=True)
            step1.mkdir(parents=True, exist_ok=True)

            events = [AssimilationEvent(date=date(2022, 10, 3), variable="station_hs", product="STATION")]
            validate_assimilation_requirements(
                setup_dir=setup_dir,
                project_dir=project_dir,
                steps=[step0, step1],
                events=events,
            )


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from ruamel.yaml import YAML

from openamundsen_da.util.da_events import AssimilationEvent
from openamundsen_da.util.validation import validate_assimilation_requirements


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = YAML()
    with path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


def _write_ascii_grid(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="AAIGrid",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs="EPSG:25832",
        transform=from_origin(0, 2, 1, 1),
        nodata=-9999,
    ) as dataset:
        dataset.write(values, 1)


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
            _write_yaml(
                project_dir / "project_2022_2023.yml",
                {"obs": {"snowcover": {"product_tag": "SNOWCOVER", "summary_csv": "obs/summaries/project_2022_2023/scf_summary.csv"}}},
            )
            (setup_dir / "obs" / "summaries" / "project_2022_2023").mkdir(parents=True, exist_ok=True)
            (setup_dir / "obs" / "summaries" / "project_2022_2023" / "scf_summary.csv").write_text(
                "date,scf\n2022-10-03,0.5\n",
                encoding="ascii",
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
            self.assertIn("instantaneous snow depth output", str(ctx.exception))

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
                    "timestep": "3h",
                    "timezone": 1,
                    "output_data": {
                        "grids": {
                            "variables": [
                                {"var": "snow.depth", "name": "snowdepth_daily"},
                            ]
                        }
                    }
                },
            )
            _write_yaml(
                project_dir / "project_2022_2023.yml",
                {"obs": {"snowcover": {"product_tag": "SNOWCOVER", "summary_csv": "obs/summaries/project_2022_2023/scf_summary.csv"}}},
            )
            (setup_dir / "obs" / "summaries" / "project_2022_2023").mkdir(parents=True, exist_ok=True)
            (setup_dir / "obs" / "summaries" / "project_2022_2023" / "scf_summary.csv").write_text(
                "date,scf\n2022-10-03,0.5\n",
                encoding="ascii",
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
                                {"var": "snow.depth", "name": "snowdepth_instantaneous"},
                                {
                                    "var": "snow.liquid_water_content",
                                    "name": "liquid_water_content_instantaneous",
                                },
                            ]
                        }
                    }
                },
            )
            _write_yaml(
                project_dir / "project_2022_2023.yml",
                {
                    "obs": {
                        "snowcover": {
                            "product_tag": "SNOWCOVER",
                            "summary_csv": "obs/summaries/project_2022_2023/scf_summary.csv",
                        },
                        "wetsnow": {
                            "product_tag": "WETSNOW",
                            "summary_csv": "obs/summaries/project_2022_2023/wet_snow_summary.csv",
                        },
                    }
                },
            )
            (setup_dir / "obs" / "summaries" / "project_2022_2023").mkdir(parents=True, exist_ok=True)
            (setup_dir / "obs" / "summaries" / "project_2022_2023" / "scf_summary.csv").write_text(
                "date,scf\n2022-10-03,0.5\n",
                encoding="ascii",
            )
            (setup_dir / "obs" / "summaries" / "project_2022_2023" / "wet_snow_summary.csv").write_text(
                "date,wet_snow_fraction\n2022-10-05,0.3\n",
                encoding="ascii",
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
            _write_yaml(
                step0 / "step_00.yml",
                {"start_date": "2022-10-01 00:00:00", "end_date": "2022-10-03 21:00:00"},
            )
            _write_yaml(
                step1 / "step_01.yml",
                {"start_date": "2022-10-04 00:00:00", "end_date": "2022-10-10 21:00:00"},
            )

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
                {
                    "timestep": "3h",
                    "timezone": 1,
                    "output_data": {
                        "timeseries": {
                            "add_default_points": False,
                            "points": [{"name": "STATION_1", "x": 0.5, "y": 0.5}],
                        },
                        "grids": {"variables": []},
                    }
                },
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
            (setup_dir / "obs" / "stations" / "stations_da_metadata.csv").write_text(
                "station_id,station_uncertainty_pct,hs_sigma_abs_min,use_for_da,use_for_benchmark\n"
                "station_1,25,0.1,true,false\n",
                encoding="ascii",
            )
            _write_yaml(
                step0 / "step_00.yml",
                {"start_date": "2022-10-01 00:00:00", "end_date": "2022-10-03 21:00:00"},
            )
            _write_yaml(
                step1 / "step_01.yml",
                {"start_date": "2022-10-04 00:00:00", "end_date": "2022-10-10 21:00:00"},
            )

            events = [AssimilationEvent(date=date(2022, 10, 3), variable="station_hs", product="STATION")]
            validate_assimilation_requirements(
                setup_dir=setup_dir,
                project_dir=project_dir,
                steps=[step0, step1],
                events=events,
            )

    def test_station_event_without_value_inside_half_timestep_fails_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2022_2023"
            step0 = project_dir / "steps" / "step_00_init"
            step1 = project_dir / "steps" / "step_01_a"
            _write_yaml(
                setup_dir / "setup_root.yml",
                {
                    "timestep": "3h",
                    "timezone": 1,
                    "output_data": {
                        "timeseries": {
                            "add_default_points": False,
                            "points": [{"name": "station_1", "x": 0.5, "y": 0.5}],
                        },
                        "grids": {"variables": []},
                    },
                },
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
            obs_dir = setup_dir / "obs" / "stations"
            obs_dir.mkdir(parents=True)
            (obs_dir / "station_1.csv").write_text(
                "time,snow_depth\n2022-10-03 02:00:00,0.4\n",
                encoding="ascii",
            )
            (obs_dir / "stations_da_metadata.csv").write_text(
                "station_id,station_uncertainty_pct,hs_sigma_abs_min,use_for_da,use_for_benchmark\n"
                "station_1,25,0.1,true,false\n",
                encoding="ascii",
            )
            _write_yaml(
                step0 / "step_00.yml",
                {"start_date": "2022-10-01 00:00:00", "end_date": "2022-10-03 21:00:00"},
            )
            _write_yaml(
                step1 / "step_01.yml",
                {"start_date": "2022-10-04 00:00:00", "end_date": "2022-10-10 21:00:00"},
            )
            events = [
                AssimilationEvent(
                    date=date(2022, 10, 3),
                    variable="station_hs",
                    product="STATION",
                )
            ]

            with self.assertRaisesRegex(ValueError, "no active station observation within half"):
                validate_assimilation_requirements(
                    setup_dir,
                    project_dir,
                    [step0, step1],
                    events,
                )

    def test_station_identity_reports_active_missing_series_and_point_but_exempts_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2022_2023"
            step0 = project_dir / "steps" / "step_00_init"
            step1 = project_dir / "steps" / "step_01_a"
            _write_yaml(
                setup_dir / "setup_root.yml",
                {
                    "timestep": "3h",
                    "timezone": 1,
                    "output_data": {
                        "timeseries": {
                            "add_default_points": False,
                            "points": [{"name": "station_da", "x": 0.5, "y": 0.5}],
                        },
                        "grids": {"variables": []},
                    }
                },
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
            obs_dir = setup_dir / "obs" / "stations"
            obs_dir.mkdir(parents=True)
            (obs_dir / "station_da.csv").write_text("time,snow_depth\n", encoding="ascii")
            (obs_dir / "disabled_station.csv").write_text("time,snow_depth\n", encoding="ascii")
            (obs_dir / "stations_da_metadata.csv").write_text(
                "station_id,station_uncertainty_pct,hs_sigma_abs_min,use_for_da,use_for_benchmark\n"
                "station_da,25,0.1,true,false\n"
                "Station_Benchmark,25,0.1,false,true\n"
                "disabled_station,25,0.1,false,false\n",
                encoding="ascii",
            )
            _write_yaml(
                step0 / "step_00.yml",
                {"start_date": "2022-10-01 00:00:00", "end_date": "2022-10-03 21:00:00"},
            )
            _write_yaml(
                step1 / "step_01.yml",
                {"start_date": "2022-10-04 00:00:00", "end_date": "2022-10-10 21:00:00"},
            )

            events = [AssimilationEvent(date=date(2022, 10, 3), variable="station_hs", product="STATION")]
            with self.assertRaises(ValueError) as ctx:
                validate_assimilation_requirements(setup_dir, project_dir, [step0, step1], events)

            message = str(ctx.exception)
            self.assertIn("Station_Benchmark", message)
            self.assertIn("missing same-ID observation CSVs", message)
            self.assertIn("missing same-ID model output points", message)
            self.assertNotIn("disabled_station", message)

    def test_station_identity_resolves_default_meteo_points_inside_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            import geopandas as gpd
            from shapely.geometry import box

            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2022_2023"
            step0 = project_dir / "steps" / "step_00_init"
            step1 = project_dir / "steps" / "step_01_a"
            _write_yaml(
                setup_dir / "setup_root.yml",
                {
                    "timestep": "3h",
                    "timezone": 1,
                    "domain": "test",
                    "resolution": 1,
                    "crs": "EPSG:25832",
                    "input_data": {"grids": {"dir": "grids"}},
                    "output_data": {
                        "timeseries": {"add_default_points": True, "points": []},
                        "grids": {"variables": []},
                    },
                },
            )
            _write_ascii_grid(setup_dir / "grids" / "dem_test_1.asc", np.ones((2, 2), dtype="float32"))
            env_dir = setup_dir / "env"
            env_dir.mkdir()
            gpd.GeoDataFrame(
                {"id": ["roi"]},
                geometry=[box(0, 0, 2, 2)],
                crs="EPSG:25832",
            ).to_file(env_dir / "roi.gpkg", driver="GPKG")
            meteo_dir = setup_dir / "meteo"
            meteo_dir.mkdir()
            (meteo_dir / "stations.csv").write_text("id,x,y\n04140864,0.5,1.5\n", encoding="ascii")
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
            obs_dir = setup_dir / "obs" / "stations"
            obs_dir.mkdir(parents=True)
            (obs_dir / "04140864.csv").write_text(
                "time,snow_depth\n2022-10-03 00:00:00,0.4\n",
                encoding="ascii",
            )
            (obs_dir / "stations_da_metadata.csv").write_text(
                "station_id,station_uncertainty_pct,hs_sigma_abs_min,use_for_da,use_for_benchmark\n"
                "04140864,25,0.1,true,false\n",
                encoding="ascii",
            )
            _write_yaml(
                step0 / "step_00.yml",
                {"start_date": "2022-10-01 00:00:00", "end_date": "2022-10-03 21:00:00"},
            )
            _write_yaml(
                step1 / "step_01.yml",
                {"start_date": "2022-10-04 00:00:00", "end_date": "2022-10-10 21:00:00"},
            )

            events = [AssimilationEvent(date=date(2022, 10, 3), variable="station_hs", product="STATION")]
            validate_assimilation_requirements(setup_dir, project_dir, [step0, step1], events)
            self.assertTrue((setup_dir / "grids" / "roi_test_1.asc").is_file())


if __name__ == "__main__":
    unittest.main()

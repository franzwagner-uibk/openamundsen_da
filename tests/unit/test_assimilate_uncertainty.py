from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import CRS
from rasterio.transform import from_origin
from ruamel.yaml import YAML

from openamundsen_da.methods.pf.assimilate_fraction import (
    FractionModelEvaluation,
    LikelihoodParams,
    ScfUncertaintyAssimilationConfig,
    WetSnowUncertaintyAssimilationConfig,
    _compute_sigma,
    _read_likelihood_from_project,
    _read_obs,
    _read_scf_uncertainty_assimilation_config,
    _read_wet_snow_uncertainty_assimilation_config,
    assimilate_fraction_for_date,
    assimilate_scf_for_date,
    assimilate_wet_snow_for_date,
)
from openamundsen_da.methods.pf.fraction_support import (
    ObservationSupportMask,
    _source_dataset_ref,
    load_observation_support_mask,
)


def _write_project_yaml(project_dir: Path, payload: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    yml = project_dir / f"{project_dir.name}.yml"
    y = YAML()
    with yml.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


def _write_two_slice_eurac_scf(path: Path) -> None:
    ds = xr.Dataset(
        {
            "fsc": (
                ("band", "y", "x"),
                np.array(
                    [
                        [[205.0, 205.0], [205.0, 205.0]],
                        [[100.0, 205.0], [0.0, 255.0]],
                    ],
                    dtype=np.float32,
                ),
            ),
        },
        coords={
            "band": [1, 2],
            "x": np.array([50.0, 150.0], dtype=np.float32),
            "y": np.array([150.0, 50.0], dtype=np.float32),
            "time": [
                np.datetime64("2024-04-01T00:00:00"),
                np.datetime64("2024-04-02T00:00:00"),
            ],
        },
    )
    ds["spatial_ref"] = xr.DataArray(0)
    ds["spatial_ref"].attrs["crs_wkt"] = CRS.from_epsg(25832).to_wkt()
    ds["spatial_ref"].attrs["spatial_ref"] = CRS.from_epsg(25832).to_wkt()
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


class AssimilateUncertaintyTests(unittest.TestCase):
    def test_source_dataset_ref_selects_scf_variable_from_netcdf(self):
        class _Container:
            count = 0
            subdatasets = (
                "netcdf:/tmp/scene.nc:uncertainty",
                "netcdf:/tmp/scene.nc:fsc",
            )

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        with patch("openamundsen_da.methods.pf.fraction_support.rasterio.open", return_value=_Container()):
            ref = _source_dataset_ref(
                Path("/tmp/scene.nc"),
                token="scene.nc@2024-04-01T00:00:00Z",
                observable="scf",
            )

        self.assertEqual(ref, "netcdf:/tmp/scene.nc:fsc")

    def test_likelihood_config_missing_block_uses_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(project_dir, {"data_assimilation": {}})

            params = _read_likelihood_from_project(project_dir, "scf")

            self.assertEqual(params, LikelihoodParams())

    def test_likelihood_config_invalid_value_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "data_assimilation": {
                        "likelihood": {
                            "scf": {
                                "obs_sigma": "not-a-number",
                            }
                        }
                    }
                },
            )

            with self.assertRaises(ValueError) as ctx:
                _read_likelihood_from_project(project_dir, "scf")
            self.assertIn("obs_sigma", str(ctx.exception))

    def test_likelihood_config_reads_min_support_coverage_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "data_assimilation": {
                        "likelihood": {
                            "scf": {
                                "min_support_coverage_ratio": 0.35,
                            }
                        }
                    }
                },
            )

            params = _read_likelihood_from_project(project_dir, "scf")

            self.assertAlmostEqual(params.min_support_coverage_ratio, 0.35, places=6)

    def test_likelihood_config_reads_min_model_finite_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "data_assimilation": {
                        "likelihood": {
                            "wet_snow_line": {
                                "min_model_finite_fraction": 0.9,
                            }
                        }
                    }
                },
            )

            params = _read_likelihood_from_project(project_dir, "wet_snow_line")

            self.assertAlmostEqual(params.min_model_finite_fraction, 0.9, places=6)

    def test_likelihood_config_rejects_invalid_min_model_finite_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "data_assimilation": {
                        "likelihood": {
                            "wet_snow_line": {
                                "min_model_finite_fraction": 1.2,
                            }
                        }
                    }
                },
            )

            with self.assertRaises(ValueError) as ctx:
                _read_likelihood_from_project(project_dir, "wet_snow_line")
            self.assertIn(
                "project.data_assimilation.likelihood.wet_snow_line.min_model_finite_fraction",
                str(ctx.exception),
            )

    def test_likelihood_config_rejects_invalid_min_support_coverage_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "data_assimilation": {
                        "likelihood": {
                            "scf": {
                                "min_support_coverage_ratio": 1.5,
                            }
                        }
                    }
                },
            )

            with self.assertRaises(ValueError) as ctx:
                _read_likelihood_from_project(project_dir, "scf")
            self.assertIn("min_support_coverage_ratio", str(ctx.exception))

    def test_uncertainty_assimilation_config_missing_block_raises_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "data_assimilation": {
                        "uncertainty": {
                            "scf": {
                                "enabled": True,
                            }
                        }
                    }
                },
            )
            with self.assertRaises(ValueError) as ctx:
                _read_scf_uncertainty_assimilation_config(project_dir)
            self.assertIn("uncertainty.scf.assimilation", str(ctx.exception))

    def test_wet_snow_uncertainty_assimilation_config_missing_block_raises_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "data_assimilation": {
                        "uncertainty": {
                            "wet_snow": {
                                "enabled": True,
                            }
                        }
                    }
                },
            )
            with self.assertRaises(ValueError) as ctx:
                _read_wet_snow_uncertainty_assimilation_config(project_dir)
            self.assertIn("uncertainty.wet_snow.assimilation", str(ctx.exception))

    def test_uncertainty_layer_sigma_uses_uncertainty_metric(self):
        sigma = _compute_sigma(
            obs={"unc_mean": 60.0, "n_valid": 100, "cloud_fraction": 0.0},
            y=0.5,
            prm=LikelihoodParams(min_sigma=0.03),
            sigma_mode="uncertainty_layer",
            uncertainty_metric="unc_mean",
            obs_path=Path("obs.csv"),
        )
        self.assertAlmostEqual(sigma, 0.6, places=6)

    def test_uncertainty_layer_sigma_respects_min_sigma_floor(self):
        sigma = _compute_sigma(
            obs={"unc_mean": 1.0, "n_valid": 100, "cloud_fraction": 0.0},
            y=0.5,
            prm=LikelihoodParams(min_sigma=0.03),
            sigma_mode="uncertainty_layer",
            uncertainty_metric="unc_mean",
            obs_path=Path("obs.csv"),
        )
        self.assertAlmostEqual(sigma, 0.03, places=6)

    def test_uncertainty_layer_out_of_range_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _compute_sigma(
                obs={"unc_mean": 120.0},
                y=0.5,
                prm=LikelihoodParams(min_sigma=0.03),
                sigma_mode="uncertainty_layer",
                uncertainty_metric="unc_mean",
                obs_path=Path("obs.csv"),
            )
        self.assertIn("out of [0,100]", str(ctx.exception))

    def test_read_obs_requires_uncertainty_metric_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "obs.csv"
            pd.DataFrame([{"date": "2024-04-01", "scf": 0.4, "n_valid": 100}]).to_csv(csv_path, index=False)
            with self.assertRaises(ValueError) as ctx:
                _read_obs(csv_path, "scf", uncertainty_metric="unc_mean")
            self.assertIn("missing required uncertainty metric", str(ctx.exception).lower())

    def test_assimilate_scf_uses_tagged_obs_candidate_when_uncertainty_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            step_dir = project_dir / "steps" / "step_00_init"
            step_dir.mkdir(parents=True, exist_ok=True)

            captured: dict[str, object] = {}

            def _fake_assimilate_fraction(**kwargs):
                captured.update(kwargs)
                return pd.DataFrame(
                    [
                        {
                            "member_id": "member_0001",
                            "value_model": 0.4,
                            "value_obs": 0.5,
                            "residual": 0.1,
                            "sigma": 0.1,
                            "log_weight": -1.0,
                            "weight": 1.0,
                        }
                    ]
                )

            with (
                patch("openamundsen_da.methods.pf.assimilate_fraction.infer_project_dir", return_value=project_dir),
                patch("openamundsen_da.methods.pf.assimilate_fraction.load_hofx_from_project", return_value=("depth_threshold", "hs", None)),
                patch(
                    "openamundsen_da.methods.pf.assimilate_fraction._read_scf_uncertainty_assimilation_config",
                    return_value=ScfUncertaintyAssimilationConfig(
                        enabled=True,
                        sigma_mode="formula",
                        aggregate_metric="unc_mean",
                    ),
                ),
                patch("openamundsen_da.methods.pf.assimilate_fraction.resolve_obs_product_tag", return_value="SNOWCOVER"),
                patch(
                    "openamundsen_da.methods.pf.assimilate_fraction.assimilate_fraction_for_date",
                    side_effect=_fake_assimilate_fraction,
                ),
            ):
                assimilate_scf_for_date(
                    setup_dir=Path(tmp) / "setup",
                    step_dir=step_dir,
                    ensemble="prior",
                    date=datetime(2024, 4, 3),
                    aoi=Path(tmp) / "roi.gpkg",
                    landcover_cfg=object(),  # avoids resolve_landcover_mask call
                    obs_csv=None,
                    product="SNOWCOVER",
                )

            obs_candidates = captured["obs_candidates"]
            self.assertIsInstance(obs_candidates, list)
            self.assertEqual(len(obs_candidates), 1)
            self.assertTrue(str(obs_candidates[0]).endswith("obs_scf_SNOWCOVER_20240403.csv"))

    def test_assimilate_wet_snow_uses_tagged_obs_candidate_when_uncertainty_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            step_dir = project_dir / "steps" / "step_00_init"
            step_dir.mkdir(parents=True, exist_ok=True)

            captured: dict[str, object] = {}

            def _fake_assimilate_fraction(**kwargs):
                captured.update(kwargs)
                return pd.DataFrame(
                    [
                        {
                            "member_id": "member_0001",
                            "value_model": 0.2,
                            "value_obs": 0.3,
                            "residual": 0.1,
                            "sigma": 0.1,
                            "log_weight": -1.0,
                            "weight": 1.0,
                        }
                    ]
                )

            with (
                patch("openamundsen_da.methods.pf.assimilate_fraction.infer_project_dir", return_value=project_dir),
                patch(
                    "openamundsen_da.methods.pf.assimilate_fraction._read_wet_snow_uncertainty_assimilation_config",
                    return_value=WetSnowUncertaintyAssimilationConfig(
                        enabled=True,
                        sigma_mode="formula",
                        aggregate_metric="unc_mean",
                    ),
                ),
                patch("openamundsen_da.methods.pf.assimilate_fraction.resolve_obs_product_tag", return_value="WETSNOW"),
                patch(
                    "openamundsen_da.methods.pf.assimilate_fraction.assimilate_fraction_for_date",
                    side_effect=_fake_assimilate_fraction,
                ),
            ):
                assimilate_wet_snow_for_date(
                    setup_dir=Path(tmp) / "setup",
                    step_dir=step_dir,
                    ensemble="prior",
                    date=datetime(2024, 5, 11),
                    aoi=Path(tmp) / "roi.gpkg",
                    landcover_cfg=object(),  # avoids resolve_landcover_mask call
                    obs_csv=None,
                    product="WETSNOW",
                )

            obs_candidates = captured["obs_candidates"]
            self.assertIsInstance(obs_candidates, list)
            self.assertEqual(len(obs_candidates), 1)
            self.assertTrue(str(obs_candidates[0]).endswith("obs_wet_snow_WETSNOW_20240511.csv"))

    def test_support_gate_writes_uniform_weights_and_support_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            step_dir = project_dir / "steps" / "step_00_init"
            obs_dir = step_dir / "obs"
            member_1 = step_dir / "ensembles" / "prior" / "member_0001"
            member_2 = step_dir / "ensembles" / "prior" / "member_0002"
            for path in (obs_dir, member_1 / "results", member_2 / "results"):
                path.mkdir(parents=True, exist_ok=True)
            y = YAML()
            with (setup_dir / "demo.yml").open("w", encoding="utf-8") as f:
                y.dump({"crs": "EPSG:25832", "domain": "demo", "resolution": 100}, f)
            _write_project_yaml(project_dir, {"data_assimilation": {}})
            obs_csv = obs_dir / "obs_scf_20240401.csv"
            pd.DataFrame(
                [
                    {
                        "date": "2024-04-01",
                        "scf": 0.6,
                        "n_valid": 10,
                        "cloud_fraction": 0.0,
                        "source": "scene.tif",
                    }
                ]
            ).to_csv(obs_csv, index=False)

            def _model_eval(_results_dir, _aoi_path, _dt, _support_info):
                return FractionModelEvaluation(
                    value_model=0.4,
                    value_model_full_roi=0.8,
                    value_model_obs_support=0.4,
                    full_roi_n_valid=100,
                    obs_support_n_valid=20,
                )

            with (
                patch(
                    "openamundsen_da.methods.pf.assimilate_fraction._read_likelihood_from_project",
                    return_value=LikelihoodParams(min_support_coverage_ratio=0.5),
                ),
                patch(
                    "openamundsen_da.methods.pf.assimilate_fraction.resolve_landcover_mask",
                    return_value=None,
                ),
                patch(
                    "openamundsen_da.methods.pf.assimilate_fraction.load_observation_support_mask",
                    return_value=ObservationSupportMask(
                        mask=np.ones((2, 2), dtype=bool),
                        eligible_mask=np.ones((2, 2), dtype=bool),
                        n_valid=1,
                        n_eligible=4,
                        coverage_ratio=0.25,
                    ),
                ),
            ):
                df = assimilate_fraction_for_date(
                    project_dir=project_dir,
                    step_dir=step_dir,
                    ensemble="prior",
                    date=datetime(2024, 4, 1),
                    aoi=Path(tmp) / "roi.gpkg",
                    obs_csv=obs_csv,
                    value_col="scf",
                    observable="scf",
                    obs_candidates=[obs_csv],
                    model_eval=_model_eval,
                )

            self.assertEqual(list(df["weight"]), [0.5, 0.5])
            self.assertEqual(list(df["log_weight"]), [0.0, 0.0])
            self.assertTrue(pd.isna(df["sigma"]).all())
            self.assertTrue(bool(df["support_gate_triggered"].iloc[0]))
            self.assertAlmostEqual(float(df["obs_support_coverage_ratio"].iloc[0]), 0.25, places=6)
            self.assertAlmostEqual(float(df["min_support_coverage_ratio"].iloc[0]), 0.5, places=6)
            self.assertEqual(float(df["value_model_full_roi"].iloc[0]), 0.8)
            self.assertEqual(float(df["value_model_obs_support"].iloc[0]), 0.4)
            self.assertEqual(int(df["obs_support_n_valid"].iloc[0]), 20)

    def test_load_observation_support_mask_falls_back_to_subdomain_manifest_raw_snowcover_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_dir = root / "setup"
            raw_snowcover_dir = setup_dir / "obs" / "snowcover"
            project_root = setup_dir / "projects" / "project_2024_2025"
            subdomain_root = project_root / "subdomains"
            sub_setup_dir = subdomain_root / "sd_01"
            sub_project_dir = sub_setup_dir / "projects" / "project_2024_2025"
            obs_csv = sub_project_dir / "steps" / "step_00_init" / "obs" / "obs_scf_SNOWCOVER_20240401.csv"
            local_obs_dir = sub_setup_dir / "obs" / "snowcover"
            grid_dir = sub_setup_dir / "grids"

            raw_snowcover_dir.mkdir(parents=True, exist_ok=True)
            local_obs_dir.mkdir(parents=True, exist_ok=True)
            grid_dir.mkdir(parents=True, exist_ok=True)
            obs_csv.parent.mkdir(parents=True, exist_ok=True)

            y = YAML()
            with (sub_setup_dir / f"{sub_setup_dir.name}.yml").open("w", encoding="utf-8") as f:
                y.dump(
                    {
                        "domain": "demo",
                        "resolution": 100,
                        "crs": "EPSG:25832",
                        "input_data": {"grids": {"dir": "grids"}},
                    },
                    f,
                )
            _write_project_yaml(
                sub_project_dir,
                {
                    "obs": {
                        "snowcover": {
                            "dir": "obs/snowcover",
                            "classes": {
                                "valid": [0, 100],
                                "cloud": [205],
                                "water": [],
                                "nodata": [255],
                            },
                        }
                    },
                    "data_assimilation": {},
                },
            )

            transform = from_origin(0.0, 200.0, 100.0, 100.0)
            roi = np.array([[1, 1], [1, 1]], dtype=np.int16)
            with rasterio.open(
                grid_dir / "dem_demo_100.tif",
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=1,
                dtype="float32",
                crs="EPSG:25832",
                transform=transform,
                nodata=-9999.0,
            ) as dst:
                dst.write(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 1)
            with rasterio.open(
                grid_dir / "roi_demo_100.asc",
                "w",
                driver="AAIGrid",
                width=2,
                height=2,
                count=1,
                dtype="uint8",
                crs="EPSG:25832",
                transform=transform,
                nodata=0,
            ) as dst:
                dst.write(roi.astype(np.uint8), 1)

            raw_raster = raw_snowcover_dir / "scene_20240401.tif"
            with rasterio.open(
                raw_raster,
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=1,
                dtype="uint8",
                crs="EPSG:25832",
                transform=transform,
                nodata=255,
            ) as dst:
                dst.write(np.array([[100, 0], [205, 255]], dtype=np.uint8), 1)

            manifest = {
                "run_mode": "subdomain",
                "setup_dir": str(setup_dir),
                "project_dir": str(project_root),
                "project_name": "project_2024_2025",
                "setup_yaml": str(setup_dir / "demo.yml"),
                "project_yaml": str(project_root / "project_2024_2025.yml"),
                "subdomain_root": str(subdomain_root),
                "regions_path": str(setup_dir / "env" / "subdomains.gpkg"),
                "id_field": "id",
                "crs": "EPSG:25832",
                "grid_rows": 2,
                "grid_cols": 2,
                "grid_transform": list(transform)[:6],
                "grid_resolution": 100.0,
                "grid_domain": "demo",
                "clip_mode": "window",
                "station_buffer_m": 0.0,
                "roi_buffer_m": 0.0,
                "grid_buffer_m": 0.0,
                "raw_snowcover_dir": str(raw_snowcover_dir),
                "raw_wetsnow_dir": str(setup_dir / "obs" / "wetsnow"),
                "subdomains": {
                    "sd_01": {
                        "id": "sd_01",
                        "label": "sd_01",
                        "setup_dir": str(sub_setup_dir),
                        "setup_yaml": str(sub_setup_dir / f"{sub_setup_dir.name}.yml"),
                        "project_dir": str(sub_project_dir),
                        "project_yaml": str(sub_project_dir / "project_2024_2025.yml"),
                        "project_name": "project_2024_2025",
                        "grids_dir": str(grid_dir),
                        "meteo_dir": str(sub_setup_dir / "meteo"),
                        "obs_stations_dir": str(sub_setup_dir / "obs" / "stations"),
                        "roi_raster_path": str(grid_dir / "roi_demo_100.asc"),
                        "roi_vector_path": str(sub_setup_dir / "env" / "roi.gpkg"),
                        "window": {"row_off": 0, "col_off": 0, "height": 2, "width": 2},
                        "transform": list(transform)[:6],
                        "bounds": [0.0, 0.0, 200.0, 200.0],
                        "crs": "EPSG:25832",
                        "status": "pending",
                    }
                },
            }
            (subdomain_root / "subdomain_manifest.json").parent.mkdir(parents=True, exist_ok=True)
            (subdomain_root / "subdomain_manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            pd.DataFrame([{"date": "2024-04-01", "scf": 0.5, "n_valid": 2, "source": raw_raster.name}]).to_csv(
                obs_csv,
                index=False,
            )

            support = load_observation_support_mask(
                setup_dir=sub_setup_dir,
                project_dir=sub_project_dir,
                obs_csv=obs_csv,
                observable="scf",
                landcover_cfg=None,
            )

            self.assertEqual(support.n_valid, 2)
            self.assertEqual(support.n_eligible, 4)
            self.assertAlmostEqual(support.coverage_ratio, 0.5, places=6)
            np.testing.assert_array_equal(
                support.mask.astype(np.uint8),
                np.array([[1, 1], [0, 0]], dtype=np.uint8),
            )

    def test_load_observation_support_mask_reads_timestamped_eurac_netcdf_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            setup_dir = root / "setup"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            obs_dir = setup_dir / "obs" / "snowcover"
            grid_dir = setup_dir / "grids"
            obs_csv = project_dir / "steps" / "step_00_init" / "obs" / "obs_scf_SNOWCOVER_20240402.csv"

            obs_dir.mkdir(parents=True, exist_ok=True)
            grid_dir.mkdir(parents=True, exist_ok=True)
            obs_csv.parent.mkdir(parents=True, exist_ok=True)
            y = YAML()
            with (setup_dir / "demo.yml").open("w", encoding="utf-8") as f:
                y.dump(
                    {
                        "domain": "demo",
                        "resolution": 100,
                        "crs": "EPSG:25832",
                        "input_data": {"grids": {"dir": "grids"}},
                    },
                    f,
                )
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

            transform = from_origin(0.0, 200.0, 100.0, 100.0)
            with rasterio.open(
                grid_dir / "dem_demo_100.tif",
                "w",
                driver="GTiff",
                width=2,
                height=2,
                count=1,
                dtype="float32",
                crs="EPSG:25832",
                transform=transform,
                nodata=-9999.0,
            ) as dst:
                dst.write(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 1)
            with rasterio.open(
                grid_dir / "roi_demo_100.asc",
                "w",
                driver="AAIGrid",
                width=2,
                height=2,
                count=1,
                dtype="uint8",
                crs="EPSG:25832",
                transform=transform,
                nodata=0,
            ) as dst:
                dst.write(np.ones((2, 2), dtype=np.uint8), 1)

            nc_path = obs_dir / "SnowFLAKES_20240402_v3_eurac.nc"
            _write_two_slice_eurac_scf(nc_path)
            pd.DataFrame(
                [
                    {
                        "date": "2024-04-02",
                        "scf": 0.5,
                        "n_valid": 2,
                        "source": f"{nc_path.name}@2024-04-02T00:00:00Z",
                    }
                ]
            ).to_csv(obs_csv, index=False)

            support = load_observation_support_mask(
                setup_dir=setup_dir,
                project_dir=project_dir,
                obs_csv=obs_csv,
                observable="scf",
                landcover_cfg=None,
            )

            self.assertEqual(support.n_valid, 2)
            self.assertEqual(support.n_eligible, 4)
            np.testing.assert_array_equal(
                support.mask.astype(np.uint8),
                np.array([[1, 0], [1, 0]], dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()

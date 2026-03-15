from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from ruamel.yaml import YAML

from openamundsen_da.methods.pf.assimilate_fraction import (
    LikelihoodParams,
    ScfUncertaintyAssimilationConfig,
    WetSnowUncertaintyAssimilationConfig,
    _compute_sigma,
    _read_likelihood_from_project,
    _read_obs,
    _read_scf_uncertainty_assimilation_config,
    _read_wet_snow_uncertainty_assimilation_config,
    assimilate_scf_for_date,
    assimilate_wet_snow_for_date,
)


def _write_project_yaml(project_dir: Path, payload: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    yml = project_dir / f"{project_dir.name}.yml"
    y = YAML()
    with yml.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


class AssimilateUncertaintyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

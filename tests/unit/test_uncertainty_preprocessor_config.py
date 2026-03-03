from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

from openamundsen_da.observer.scf_uncertainty import (
    PenaltyRule as ScfPenaltyRule,
    ScfClassMapping,
    ScfUncertaintyConfig,
    _build_uncertainty as build_scf_uncertainty,
    _load_project_config as load_scf_uncertainty_config,
)
from openamundsen_da.observer.wetsnow_uncertainty import (
    _load_project_config as load_wetsnow_uncertainty_config,
)


def _write_project_yaml(project_dir: Path, payload: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    yml = project_dir / f"{project_dir.name}.yml"
    y = YAML()
    with yml.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


class UncertaintyPreprocessorConfigTests(unittest.TestCase):
    def test_scf_defaults_map_cloud_and_water_to_max_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup"
            project_dir = setup_dir / "projects" / "project_2024_2025"
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
                                "penalties": [
                                    {"source": "fsc", "classes": [50], "penalty": 20.0},
                                ]
                            }
                        }
                    },
                },
            )
            cfg, _ = load_scf_uncertainty_config(project_dir)
            self.assertEqual(cfg.class_mapping.base_classes, (0, 50, 100))
            self.assertEqual(cfg.class_mapping.max_uncertainty_classes, (205, 210))
            self.assertEqual(cfg.class_mapping.nodata_classes, (255,))
            self.assertEqual(cfg.penalties[0].classes, (50,))

    def test_scf_penalty_groups_are_resolved_from_obs_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup"
            project_dir = setup_dir / "projects" / "project_2024_2025"
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
                                "penalties": [
                                    {"source": "fsc", "groups": ["cloud", "water"], "penalty": 20.0},
                                ]
                            }
                        }
                    },
                },
            )
            cfg, _ = load_scf_uncertainty_config(project_dir)
            self.assertEqual(cfg.penalties[0].classes, (205, 210))

    def test_wetsnow_defaults_keep_exclude_as_nodata(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            _write_project_yaml(
                project_dir,
                {
                    "obs": {
                        "wetsnow": {
                            "dir": "obs/wetsnow",
                            "classes": {
                                "wet": [110],
                                "valid": [110, 125, 200, 210],
                                "exclude": [200, 210],
                            },
                        }
                    },
                    "data_assimilation": {
                        "uncertainty": {
                            "wet_snow": {
                                "penalties": [
                                    {"source": "wet_snow", "classes": [110], "penalty": 20.0},
                                ]
                            }
                        }
                    },
                },
            )
            cfg, _ = load_wetsnow_uncertainty_config(project_dir)
            self.assertEqual(cfg.class_mapping.base_classes, (110, 125))
            self.assertEqual(cfg.class_mapping.max_uncertainty_classes, ())
            self.assertEqual(cfg.class_mapping.nodata_classes, (200, 210))
            self.assertEqual(cfg.penalties[0].classes, (110,))

    def test_scf_build_uncertainty_sets_cloud_and_water_to_100(self):
        cfg = ScfUncertaintyConfig(
            enabled=True,
            input_dir=Path("."),
            u_min=10.0,
            u_max=20.0,
            nodata_value=255.0,
            class_mapping=ScfClassMapping(
                base_classes=tuple(range(0, 101)),
                max_uncertainty_classes=(205, 210),
                nodata_classes=(255,),
            ),
            penalties=(
                ScfPenaltyRule(
                    name="noop",
                    source="fsc",
                    classes=(999,),
                    penalty=0.0,
                    enabled=True,
                    input_dir=None,
                ),
            ),
        )
        fsc = np.array(
            [
                [50.0, 205.0, 210.0],
                [0.0, 100.0, 255.0],
            ],
            dtype=np.float32,
        )
        unc, _ = build_scf_uncertainty(
            fsc=fsc,
            landcover_resampled=None,
            shadow_by_rule={},
            cfg=cfg,
        )
        self.assertAlmostEqual(float(unc[0, 0]), 20.0, places=6)
        self.assertAlmostEqual(float(unc[0, 1]), 100.0, places=6)
        self.assertAlmostEqual(float(unc[0, 2]), 100.0, places=6)
        self.assertAlmostEqual(float(unc[1, 0]), 10.0, places=6)
        self.assertAlmostEqual(float(unc[1, 1]), 10.0, places=6)
        self.assertAlmostEqual(float(unc[1, 2]), 255.0, places=6)


if __name__ == "__main__":
    unittest.main()

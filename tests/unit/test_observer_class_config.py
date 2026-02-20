from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ruamel.yaml import YAML

from openamundsen_da.observer.snowcover import _load_classes as load_snowcover_classes
from openamundsen_da.observer.class_config import load_wetsnow_classes


def _write_project_yaml(project_dir: Path, payload: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = project_dir / f"{project_dir.name}.yml"
    y = YAML()
    with yaml_path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


class ObserverClassConfigTests(unittest.TestCase):
    def test_snowcover_classes_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_ci_2022_2023"
            _write_project_yaml(
                project_dir,
                {
                    "obs": {
                        "snowcover": {"dir": "obs/snowcover"},
                    }
                },
            )
            with self.assertRaises(ValueError) as ctx:
                load_snowcover_classes(project_dir)
            self.assertIn("project.obs.snowcover.classes", str(ctx.exception))

    def test_snowcover_classes_are_loaded_from_project_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_ci_2022_2023"
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
                    }
                },
            )
            classes = load_snowcover_classes(project_dir)
            self.assertEqual(classes.valid, [0, 50, 100])
            self.assertEqual(classes.cloud, [205])
            self.assertEqual(classes.water, [210])
            self.assertEqual(classes.nodata, [255])

    def test_wetsnow_classes_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_ci_2022_2023"
            _write_project_yaml(
                project_dir,
                {
                    "obs": {
                        "wetsnow": {"dir": "obs/wetsnow"},
                    }
                },
            )

            with self.assertRaises(ValueError) as ctx_observer:
                load_wetsnow_classes(project_dir)
            self.assertIn("project.obs.wetsnow.classes", str(ctx_observer.exception))

    def test_wetsnow_classes_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_ci_2022_2023"
            _write_project_yaml(
                project_dir,
                {
                    "obs": {
                        "wetsnow": {
                            "dir": "obs/wetsnow",
                            "classes": {
                                "wet": [1, 2],
                                "valid": [1, 2, 3, 4, 255],
                                "exclude": [5, 6],
                            },
                        }
                    }
                },
            )

            wet_obs = load_wetsnow_classes(project_dir)
            self.assertEqual(wet_obs, ([1, 2], [1, 2, 3, 4, 255], [5, 6]))


if __name__ == "__main__":
    unittest.main()

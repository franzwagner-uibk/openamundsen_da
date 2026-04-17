import tempfile
import unittest
from pathlib import Path

from ruamel.yaml import YAML

from openamundsen_da.methods.h_of_x.model_scf import load_hofx_from_project


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = YAML()
    with path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


class HofxConfigTests(unittest.TestCase):
    def test_loads_hofx_from_data_assimilation_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp)
            project_dir = setup_dir / "projects" / "project_test"
            _write_yaml(
                project_dir / "project_test.yml",
                {
                    "data_assimilation": {
                        "h_of_x": {
                            "method": "logistic",
                            "variable": "hs",
                            "params": {"h0": 0.1, "k": 12.0},
                        }
                    }
                },
            )

            method, variable, params = load_hofx_from_project(project_dir)

            self.assertEqual(method, "logistic")
            self.assertEqual(variable, "hs")
            self.assertEqual(params.h0, 0.1)
            self.assertEqual(params.k, 12.0)

    def test_top_level_hofx_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp)
            project_dir = setup_dir / "projects" / "project_test"
            _write_yaml(
                project_dir / "project_test.yml",
                {
                    "h_of_x": {
                        "method": "depth_threshold",
                        "variable": "hs",
                        "params": {"h0": 0.05},
                    }
                },
            )

            with self.assertRaisesRegex(ValueError, "data_assimilation.h_of_x"):
                load_hofx_from_project(project_dir)


if __name__ == "__main__":
    unittest.main()

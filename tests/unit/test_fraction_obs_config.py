from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ruamel.yaml import YAML

from openamundsen_da.observer.fraction_obs import resolve_obs_product_tag


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = YAML()
    with path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


class FractionObsConfigTests(unittest.TestCase):
    def test_resolve_product_tag_from_project_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            _write_yaml(
                project_dir / "project_2024_2025.yml",
                {
                    "obs": {
                        "snowcover": {"product_tag": "FSC"},
                        "wetsnow": {"product_tag": "SWS"},
                    }
                },
            )
            self.assertEqual(resolve_obs_product_tag("scf", project_dir=project_dir), "FSC")
            self.assertEqual(resolve_obs_product_tag("wet_snow", project_dir=project_dir), "SWS")

    def test_missing_product_tag_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup"
            project_dir = setup_dir / "projects" / "project_2024_2025"
            _write_yaml(
                project_dir / "project_2024_2025.yml",
                {
                    "obs": {
                        "snowcover": {},
                        "wetsnow": {"product_tag": "SWS"},
                    }
                },
            )
            with self.assertRaises(ValueError) as ctx:
                resolve_obs_product_tag("scf", project_dir=project_dir)
            self.assertIn("project.obs.snowcover.product_tag", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


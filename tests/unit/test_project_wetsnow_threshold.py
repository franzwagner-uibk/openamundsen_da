from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ruamel.yaml import YAML

from openamundsen_da.pipeline.project import _load_wet_snow_classification_config, _load_wet_snow_threshold_percent


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = YAML()
    with path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


class ProjectWetSnowThresholdTests(unittest.TestCase):
    def test_threshold_is_loaded_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_yaml(
                project_dir / "project_2024_2025.yml",
                {"data_assimilation": {"wet_snow": {"classification_threshold_percent": 0.5}}},
            )
            self.assertEqual(_load_wet_snow_threshold_percent(project_dir), 0.5)

    def test_missing_threshold_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_yaml(
                project_dir / "project_2024_2025.yml",
                {"data_assimilation": {"wet_snow": {}}},
            )
            with self.assertRaises(ValueError) as ctx:
                _load_wet_snow_threshold_percent(project_dir)
            self.assertIn("classification_threshold_percent", str(ctx.exception))

    def test_amount_method_does_not_require_fraction_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "setup" / "projects" / "project_2024_2025"
            _write_yaml(
                project_dir / "project_2024_2025.yml",
                {"data_assimilation": {"wet_snow": {"classification_method": "liquid_water_amount"}}},
            )
            cfg = _load_wet_snow_classification_config(project_dir)
            self.assertEqual(cfg.method, "liquid_water_amount")
            self.assertEqual(cfg.liquid_water_amount_threshold_mm, 5.0)


if __name__ == "__main__":
    unittest.main()

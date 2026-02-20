from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ruamel.yaml import YAML

from openamundsen_da.observer.hrwsi_download import load_hrwsi_config


def _write_project_yaml(project_dir: Path, payload: dict) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = project_dir / f"{project_dir.name}.yml"
    y = YAML()
    with yaml_path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


def _base_project_payload() -> dict:
    return {
        "copernicus_download": {
            "endpoint_url": "https://s3.example.org",
            "bucket": "HRWSI",
            "access_key": "${HRWSI_ACCESS_KEY}",
            "secret_key": "env:HRWSI_SECRET_KEY",
            "tiles": ["32TPS", "32TPT"],
            "snowcover": {
                "product": "FSC",
                "filename_suffix": "_FSCTOC.tif",
                "output_dir": "obs/snowcover",
            },
            "wetsnow": {
                "product": "SWS",
                "filename_suffix": "_WSM.tif",
                "output_dir": "obs/wetsnow",
            },
        }
    }


class HrwsiDownloadConfigTests(unittest.TestCase):
    def test_load_hrwsi_config_resolves_env_credentials(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup"
            project_dir = setup_dir / "projects" / "project_ci_2024_2025"
            _write_project_yaml(project_dir, _base_project_payload())

            with mock.patch.dict(
                os.environ,
                {"HRWSI_ACCESS_KEY": "access_from_env", "HRWSI_SECRET_KEY": "secret_from_env"},
                clear=False,
            ):
                cfg = load_hrwsi_config(setup_dir, project_dir)

            self.assertEqual(cfg.access_key, "access_from_env")
            self.assertEqual(cfg.secret_key, "secret_from_env")
            self.assertEqual(cfg.tiles, ("32TPS", "32TPT"))
            self.assertEqual(cfg.snowcover.output_dir, setup_dir / "obs/snowcover")
            self.assertEqual(cfg.wetsnow.output_dir, setup_dir / "obs/wetsnow")

    def test_load_hrwsi_config_errors_for_missing_env_var(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup"
            project_dir = setup_dir / "projects" / "project_ci_2024_2025"
            _write_project_yaml(project_dir, _base_project_payload())

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("HRWSI_ACCESS_KEY", None)
                os.environ.pop("HRWSI_SECRET_KEY", None)
                with self.assertRaises(ValueError) as ctx:
                    load_hrwsi_config(setup_dir, project_dir)

            self.assertIn("environment variable", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

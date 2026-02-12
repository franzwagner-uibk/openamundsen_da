from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from openamundsen_da.io.paths import find_project_yaml, infer_project_dir, infer_setup_dir


class YamlDiscoveryTests(unittest.TestCase):
    def test_find_project_yaml_prefers_project_name(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "projects" / "project_alpha"
            project_dir.mkdir(parents=True, exist_ok=True)
            preferred = project_dir / "project_alpha.yml"
            fallback = project_dir / "project.yml"
            preferred.write_text("a: 1\n", encoding="utf-8")
            fallback.write_text("a: 2\n", encoding="utf-8")
            self.assertEqual(find_project_yaml(project_dir), preferred)

    def test_find_project_yaml_uses_project_fallback(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "projects" / "project_beta"
            project_dir.mkdir(parents=True, exist_ok=True)
            fallback = project_dir / "project.yml"
            fallback.write_text("a: 1\n", encoding="utf-8")
            self.assertEqual(find_project_yaml(project_dir), fallback)

    def test_find_project_yaml_rejects_unrelated_single_yaml(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp) / "projects" / "project_gamma"
            project_dir.mkdir(parents=True, exist_ok=True)
            (project_dir / "00.yml").write_text("a: 1\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                find_project_yaml(project_dir)

    def test_infer_project_dir_from_step_dir(self):
        with TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "rofental"
            setup_dir.mkdir(parents=True, exist_ok=True)
            (setup_dir / "rofental.yml").write_text("a: 1\n", encoding="utf-8")

            project_dir = setup_dir / "projects" / "project_ci_2022_2023"
            step_dir = project_dir / "steps" / "step_00_init"
            step_dir.mkdir(parents=True, exist_ok=True)

            (project_dir / "project_ci_2022_2023.yml").write_text("start_date: 2023-03-12\n", encoding="utf-8")
            (step_dir / "00.yml").write_text("start_date: 2023-03-12\n", encoding="utf-8")

            self.assertEqual(infer_project_dir(step_dir), project_dir.resolve())
            self.assertEqual(infer_setup_dir(step_dir), setup_dir.resolve())


if __name__ == "__main__":
    unittest.main()

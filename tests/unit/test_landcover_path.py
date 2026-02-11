import tempfile
import unittest
from pathlib import Path

from openamundsen_da.util.landcover_mask import _derive_landcover_path


class LandcoverPathTests(unittest.TestCase):
    def test_prefers_exact_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            grids = project_dir / "grids"
            grids.mkdir(parents=True, exist_ok=True)
            exact = grids / "lc_demo_100.asc"
            exact.write_text("ncols 1\nnrows 1\n", encoding="ascii")
            (grids / "lc_demo_100_alt.asc").write_text("ncols 1\nnrows 1\n", encoding="ascii")

            resolved = _derive_landcover_path(project_dir, "demo", 100)

            self.assertEqual(resolved, exact)

    def test_allows_single_suffix_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            grids = project_dir / "grids"
            grids.mkdir(parents=True, exist_ok=True)
            candidate = grids / "lc_demo_100_large.asc"
            candidate.write_text("ncols 1\nnrows 1\n", encoding="ascii")

            resolved = _derive_landcover_path(project_dir, "demo", 100)

            self.assertEqual(resolved, candidate)

    def test_rejects_multiple_suffix_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            grids = project_dir / "grids"
            grids.mkdir(parents=True, exist_ok=True)
            (grids / "lc_demo_100_a.asc").write_text("ncols 1\nnrows 1\n", encoding="ascii")
            (grids / "lc_demo_100_b.asc").write_text("ncols 1\nnrows 1\n", encoding="ascii")

            with self.assertRaises(FileExistsError):
                _derive_landcover_path(project_dir, "demo", 100)


if __name__ == "__main__":
    unittest.main()

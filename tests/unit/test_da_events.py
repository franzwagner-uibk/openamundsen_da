import tempfile
import unittest
from datetime import date
from pathlib import Path

from ruamel.yaml import YAML

from openamundsen_da.util.da_events import _parse_event_date, load_assimilation_events


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = YAML()
    with path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


def _write_setup_with_product_tags(setup_dir: Path) -> None:
    _write_yaml(
        setup_dir / "setup_root.yml",
        {
            "obs": {
                "snowcover": {"product_tag": "SNOWCOVER"},
                "wetsnow": {"product_tag": "WETSNOW"},
            }
        },
    )


class DaEventsTests(unittest.TestCase):
    def test_parse_event_date(self):
        self.assertEqual(_parse_event_date("2022-10-03"), date(2022, 10, 3))
        with self.assertRaises(ValueError):
            _parse_event_date("2022/10/03")

    def test_load_events_sorts_and_normalizes_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2022_2023"
            _write_setup_with_product_tags(setup_dir)
            _write_yaml(
                project_dir / "project_2022_2023.yml",
                {
                    "obs": {
                        "snowcover": {"product_tag": "SNOWCOVER"},
                        "wetsnow": {"product_tag": "WETSNOW"},
                    },
                    "data_assimilation": {
                        "assimilation_events": [
                            {"date": "2022-10-05", "variable": "scf", "product": "SNOWCOVER"},
                            {"date": "2022-10-03", "variable": "wet_snow", "product": "WETSNOW"},
                            {"date": "2022-10-04", "variable": "scf"},
                        ]
                    }
                },
            )

            events = load_assimilation_events(project_dir)

            self.assertEqual([e.date for e in events], [date(2022, 10, 3), date(2022, 10, 4), date(2022, 10, 5)])
            self.assertEqual(events[0].variable, "wet_snow")
            self.assertEqual(events[0].product, "WETSNOW")
            self.assertEqual(events[1].variable, "scf")
            self.assertEqual(events[1].product, "SNOWCOVER")
            self.assertEqual(events[2].variable, "scf")
            self.assertEqual(events[2].product, "SNOWCOVER")

    def test_missing_variable_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2022_2023"
            _write_setup_with_product_tags(setup_dir)
            _write_yaml(
                project_dir / "project_2022_2023.yml",
                {"data_assimilation": {"assimilation_events": [{"date": "2022-10-04"}]}},
            )
            with self.assertRaises(ValueError) as ctx:
                load_assimilation_events(project_dir)
            self.assertIn("assimilation_events[1].variable", str(ctx.exception))

    def test_non_mapping_event_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2022_2023"
            _write_setup_with_product_tags(setup_dir)
            _write_yaml(
                project_dir / "project_2022_2023.yml",
                {"data_assimilation": {"assimilation_events": ["2022-10-04"]}},
            )
            with self.assertRaises(ValueError) as ctx:
                load_assimilation_events(project_dir)
            self.assertIn("Expected mapping", str(ctx.exception))

    def test_no_events_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup_dir = Path(tmp) / "setup_root"
            project_dir = setup_dir / "projects" / "project_2022_2023"
            _write_setup_with_product_tags(setup_dir)
            _write_yaml(project_dir / "project_2022_2023.yml", {"data_assimilation": {"assimilation_events": []}})
            with self.assertRaises(ValueError):
                load_assimilation_events(project_dir)


if __name__ == "__main__":
    unittest.main()

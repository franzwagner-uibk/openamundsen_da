import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from ruamel.yaml import YAML

from openamundsen_da.util.da_events import _parse_event_date, load_assimilation_events


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = YAML()
    with path.open("w", encoding="utf-8") as f:
        y.dump(payload, f)


class DaEventsTests(unittest.TestCase):
    def test_parse_event_date(self):
        self.assertEqual(_parse_event_date("2022-10-03"), date(2022, 10, 3))
        with self.assertRaises(ValueError):
            _parse_event_date("2022/10/03")

    def test_load_events_sorts_and_normalizes_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            season_dir = Path(tmp) / "project" / "propagation" / "season_2022_2023"
            _write_yaml(
                season_dir / "season.yml",
                {
                    "data_assimilation": {
                        "assimilation_events": [
                            {"date": "2022-10-05", "variable": "scf", "product": "MOD10A1"},
                            {"date": "2022-10-03", "variable": "wet_snow", "product": "S1"},
                            {"date": "2022-10-04"},
                        ]
                    }
                },
            )

            def _default_tag(var: str, **_: object) -> str:
                if var == "wet_snow":
                    return "WETSNOW"
                return "SNOWCOVER"

            with patch("openamundsen_da.util.da_events.resolve_obs_product_tag", side_effect=_default_tag):
                events = load_assimilation_events(season_dir)

            self.assertEqual([e.date for e in events], [date(2022, 10, 3), date(2022, 10, 4), date(2022, 10, 5)])
            self.assertEqual(events[0].variable, "wet_snow")
            self.assertEqual(events[0].product, "WETSNOW")
            self.assertEqual(events[1].variable, "scf")
            self.assertEqual(events[1].product, "SNOWCOVER")
            self.assertEqual(events[2].variable, "scf")
            self.assertEqual(events[2].product, "SNOWCOVER")

    def test_no_events_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            season_dir = Path(tmp) / "project" / "propagation" / "season_2022_2023"
            _write_yaml(season_dir / "season.yml", {"data_assimilation": {"assimilation_events": []}})
            with self.assertRaises(ValueError):
                load_assimilation_events(season_dir)


if __name__ == "__main__":
    unittest.main()

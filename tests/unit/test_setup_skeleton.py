import unittest
from datetime import timedelta

from openamundsen_da.pipeline.season_skeleton import _parse_timestep


class ParseTimestepTests(unittest.TestCase):
    def test_parse_hours(self):
        self.assertEqual(_parse_timestep("3H"), timedelta(hours=3))
        self.assertEqual(_parse_timestep("h"), timedelta(hours=1))

    def test_parse_days(self):
        self.assertEqual(_parse_timestep("1D"), timedelta(days=1))
        self.assertEqual(_parse_timestep("2d"), timedelta(days=2))

    def test_rejects_unsupported_format(self):
        with self.assertRaises(ValueError):
            _parse_timestep("30min")


if __name__ == "__main__":
    unittest.main()

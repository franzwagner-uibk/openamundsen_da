import unittest
from datetime import datetime, timedelta
from pathlib import Path

from openamundsen_da.io.paths import list_steps_sorted, read_step_config
from openamundsen_da.pipeline.project_skeleton import (
    _parse_timestep,
    create_project_skeleton,
    plan_project_steps,
)


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


def test_pure_step_plan_matches_materialized_skeleton(tmp_path: Path) -> None:
    setup = tmp_path / "setup"
    project = setup / "projects" / "demo"
    project.mkdir(parents=True)
    (setup / "setup_config.yml").write_text("timestep: 3H\n", encoding="utf-8")
    (project / "demo.yml").write_text(
        "start_date: '2023-10-01 00:00:00'\n"
        "end_date: '2024-09-30 21:00:00'\n"
        "data_assimilation:\n"
        "  assimilation_events:\n"
        "    - {date: '2023-10-07', variable: scf}\n"
        "    - {date: '2023-10-13', variable: station_hs}\n",
        encoding="utf-8",
    )

    planned = plan_project_steps(setup, project)

    assert not (project / "steps").exists()
    assert [step.name for step in planned] == [
        "step_00_init",
        "step_01_20231007-20231013",
        "step_02_20231013-20240930",
    ]
    create_project_skeleton(setup, project)
    materialized = [
        (
            step.name,
            datetime.fromisoformat(str(read_step_config(step)["start_date"])),
            datetime.fromisoformat(str(read_step_config(step)["end_date"])),
        )
        for step in list_steps_sorted(project)
    ]

    assert materialized == [
        (step.name, step.start, step.end)
        for step in planned
    ]


if __name__ == "__main__":
    unittest.main()

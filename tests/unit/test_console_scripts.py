import tomllib
from pathlib import Path


def test_project_skeleton_console_script_is_published():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["scripts"]["oa-da-project-skeleton"] == (
        "openamundsen_da.pipeline.project_skeleton:cli"
    )

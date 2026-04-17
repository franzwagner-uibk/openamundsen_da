import tomllib
from pathlib import Path


def test_project_skeleton_console_script_is_published():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["scripts"]["oa-da-project-skeleton"] == (
        "openamundsen_da.pipeline.project_skeleton:cli"
    )


def test_result_overview_console_script_is_published():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["scripts"]["oa-da-plot-result-overview"] == (
        "openamundsen_da.methods.viz.plots.result_overview:cli_main"
    )


def test_benchmark_console_script_is_published():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["scripts"]["oa-da-benchmark"] == (
        "openamundsen_da.benchmark.pipeline:cli"
    )


def test_project_maps_console_script_is_published():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["scripts"]["oa-da-plot-project-maps"] == (
        "openamundsen_da.methods.viz.project_maps:cli_main"
    )


def test_fetch_overview_geojson_console_script_is_published():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["scripts"]["oa-da-fetch-overview-geojson"] == (
        "openamundsen_da.methods.viz.project_maps.overview:cli_main"
    )

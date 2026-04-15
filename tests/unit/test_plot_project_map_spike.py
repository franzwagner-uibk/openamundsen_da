from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "dev" / "plot_project_map_spike.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("plot_project_map_spike", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_derive_setup_dir_returns_parent_setup_root(tmp_path: Path) -> None:
    module = _load_script_module()
    project_dir = tmp_path / "setup" / "projects" / "project_2022_2023"
    project_dir.mkdir(parents=True)

    resolved = module._derive_setup_dir(project_dir)

    assert resolved == (tmp_path / "setup").resolve()


def test_default_output_path_points_into_project_results_maps() -> None:
    module = _load_script_module()
    project_dir = Path("/tmp/example/projects/project_2022_2023")

    output_path = module._default_output_path(project_dir, "2023-06-02")

    assert output_path == project_dir / "results" / "maps" / "overview" / "spike_snow_depth_2023-06-02.png"


def test_nice_ceiling_rounds_up_to_requested_step() -> None:
    module = _load_script_module()

    assert module._nice_ceiling(1.02, step=0.25, minimum=0.5) == 1.25
    assert module._nice_ceiling(0.11, step=0.25, minimum=0.5) == 0.5

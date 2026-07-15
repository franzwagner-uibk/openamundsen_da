from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys

from PIL import Image


def _load_module():
    script = Path(__file__).parents[2] / "scripts" / "release" / "validate_manuscript_reference.py"
    spec = importlib.util.spec_from_file_location("validate_manuscript_reference", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract() -> dict:
    path = (
        Path(__file__).parents[1]
        / "baselines"
        / "rofental_es30_manuscript_contract.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_refresh_module(monkeypatch):
    script = Path(__file__).parents[2] / "scripts" / "release" / "refresh_manuscript_outputs.py"
    monkeypatch.syspath_prepend(str(script.parent))
    spec = importlib.util.spec_from_file_location("refresh_manuscript_outputs", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shipped_config_matches_manuscript_parameter_contract() -> None:
    module = _load_module()
    setup = Path(__file__).parents[2] / "examples" / "rofental"

    assert module._validate_config(setup, _contract()) == []


def test_tex_validation_allows_acknowledged_precipitation_exception() -> None:
    module = _load_module()
    contract = _contract()
    valid = "\n".join(contract["manuscript_required_literals"])
    acknowledged = contract["manuscript_acknowledged_exceptions"][0]["literal"]

    assert module._validate_tex(valid, contract) == []
    assert module._validate_tex(f"{valid}\n{acknowledged}", contract) == []


def test_result_validation_checks_ess_decision_and_benchmark_values(tmp_path: Path) -> None:
    module = _load_module()
    weights = tmp_path / "weights.csv"
    weights.write_text("weight\n0.5\n0.5\n", encoding="utf-8")
    summary = (
        tmp_path
        / "projects"
        / "project_2022_2023"
        / "results"
        / "benchmark"
        / "tables"
        / "update_summary.csv"
    )
    summary.parent.mkdir(parents=True)
    summary.write_text(
        "assimilation_date,variable,stream,prior_crpss,posterior_crpss\n"
        "2023-02-21,station_hs,assimilation_fit,0.15,0.84\n",
        encoding="utf-8",
    )
    contract = {
        "ess": [{"path": "weights.csv", "expected": 2.0, "resampled": True}],
        "benchmark_claims": [
            {
                "date": "2023-02-21",
                "variable": "station_hs",
                "prior_crpss": 0.15,
                "posterior_crpss": 0.84,
            }
        ],
    }

    assert module._validate_results(tmp_path, contract) == []


def test_figure_validation_checks_run_and_manuscript_pixels(tmp_path: Path) -> None:
    module = _load_module()
    run_figure = tmp_path / "run" / "figure.png"
    run_figure.parent.mkdir()
    Image.new("RGBA", (2, 3), (1, 2, 3, 255)).save(run_figure)
    record = {"name": "fig03.png", "source": "run/figure.png", **module._image_record(run_figure)}
    manuscript = tmp_path / "manuscript"
    (manuscript / "assets").mkdir(parents=True)
    shutil.copy2(run_figure, manuscript / "assets" / "fig03.png")

    assert module._validate_figures(tmp_path, {"figures": [record]}, manuscript) == []

    Image.new("RGBA", (2, 3), (4, 5, 6, 255)).save(manuscript / "assets" / "fig03.png")
    differences = module._validate_figures(tmp_path, {"figures": [record]}, manuscript)
    assert any("manuscript:fig03.png:pixels_sha256" in item for item in differences)


def test_figure_validation_accepts_whitelisted_run_record_only(tmp_path: Path) -> None:
    module = _load_module()
    run_figure = tmp_path / "run" / "figure.png"
    run_figure.parent.mkdir()
    Image.new("RGBA", (2, 3), (4, 5, 6, 255)).save(run_figure)
    accepted = module._image_record(run_figure)
    canonical = tmp_path / "canonical.png"
    Image.new("RGBA", (2, 3), (1, 2, 3, 255)).save(canonical)
    record = {
        "name": "fig03.png",
        "source": "run/figure.png",
        **module._image_record(canonical),
        "accepted_run_records": [accepted],
    }

    assert module._validate_figures(tmp_path, {"figures": [record]}, None) == []


def test_input_manifest_locks_selected_source_run() -> None:
    module = _load_module()
    contract = _contract()
    manifest_path = (
        Path(__file__).parents[1]
        / "baselines"
        / "rofental_es30_manuscript_inputs"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert module._validate_input_manifest(contract, manifest) == []


def test_frozen_input_validation_checks_file_hash(tmp_path: Path) -> None:
    module = _load_module()
    frozen = tmp_path / "data.txt"
    frozen.write_text("selected\n", encoding="utf-8")
    manifest = {
        "files": [
            {
                "path": "data.txt",
                "sha256": module._sha256(frozen),
            }
        ]
    }

    assert module._validate_frozen_inputs(tmp_path, manifest) == []
    frozen.write_text("drifted\n", encoding="utf-8")
    assert module._validate_frozen_inputs(tmp_path, manifest)[0].startswith(
        "frozen manuscript input differs: data.txt"
    )


def test_publication_overlay_is_checksum_validated(tmp_path: Path, monkeypatch) -> None:
    module = _load_refresh_module(monkeypatch)
    source_root = tmp_path / "source"
    run_root = tmp_path / "run"
    source = source_root / "source.csv"
    source.parent.mkdir()
    source.write_text("publication\n", encoding="utf-8")
    run_root.mkdir()
    contract = {
        "publication_analysis_overlay": [
            {
                "source": "source.csv",
                "path": "analysis/input.csv",
                "sha256": module._sha256(source),
            }
        ]
    }

    written = module.apply_publication_overlay(run_root, contract, source_root=source_root)

    assert written == ((run_root / "analysis" / "input.csv").resolve(),)
    assert written[0].read_text(encoding="utf-8") == "publication\n"

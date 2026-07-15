from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openamundsen_da import cli


@pytest.mark.parametrize(
    "arguments",
    [
        ["subdomains", "pipeline", "/tmp/project"],
        ["subdomains", "plot", "/tmp/project"],
        ["subdomains", "model-pipeline", "/tmp/setup"],
        ["subdomains", "model-prepare", "/tmp/setup"],
    ],
)
def test_removed_subdomain_aliases_are_rejected(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(arguments)

    assert excinfo.value.code == 2


def test_da_prepare_dispatches_directly_to_staged_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "winter"
    regions = setup_dir / "env" / "subdomains.gpkg"
    regions.parent.mkdir(parents=True)
    regions.write_bytes(b"regions")
    project_dir.mkdir(parents=True)
    called: dict[str, object] = {}

    def fake_prepare(**kwargs):
        called.update(kwargs)
        return SimpleNamespace(
            subdomain_root=project_dir / "subdomains",
            subdomains={"sd_01": object(), "sd_02": object()},
        )

    monkeypatch.setattr("openamundsen_da.subdomain.prepare.prepare_subdomains", fake_prepare)

    exit_code = cli.main(["subdomains", "prepare", str(project_dir), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert called["setup_dir"] == setup_dir.resolve()
    assert called["project_dir"] == project_dir.resolve()
    assert called["regions_path"] == regions.resolve()
    assert payload["result"]["subdomain_count"] == 2


def test_model_prepare_keeps_plain_model_root_separate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    setup_dir = tmp_path / "setup"
    regions = setup_dir / "env" / "subdomains.gpkg"
    regions.parent.mkdir(parents=True)
    regions.write_bytes(b"regions")
    called: dict[str, object] = {}

    def fake_prepare(**kwargs):
        called.update(kwargs)
        return SimpleNamespace(
            subdomain_root=setup_dir / "subdomains" / "model",
            subdomains={"sd_01": object()},
        )

    monkeypatch.setattr(
        "openamundsen_da.subdomain.prepare.prepare_model_subdomains",
        fake_prepare,
    )

    exit_code = cli.main(["subdomains", "model", "prepare", str(setup_dir)])

    assert exit_code == 0
    assert called["setup_dir"] == setup_dir.resolve()
    assert called["regions_path"] == regions.resolve()


def test_da_merge_dispatches_to_manifest_owned_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "setup" / "projects" / "winter"
    project_dir.mkdir(parents=True)
    called: dict[str, object] = {}

    def fake_merge(**kwargs):
        called.update(kwargs)
        return [project_dir / "results" / "grids" / "da_output_grids.nc"]

    monkeypatch.setattr("openamundsen_da.subdomain.merge.merge_grids", fake_merge)

    exit_code = cli.main(["subdomains", "merge", str(project_dir)])

    assert exit_code == 0
    assert called["manifest_path"] == project_dir / "subdomains" / "subdomain_manifest.json"


def test_model_merge_dispatches_to_plain_model_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    setup_dir = tmp_path / "setup"
    setup_dir.mkdir()
    called: dict[str, object] = {}

    def fake_merge(**kwargs):
        called.update(kwargs)
        return [setup_dir / "subdomains" / "model" / "results" / "grids" / "output_grids.nc"]

    monkeypatch.setattr("openamundsen_da.subdomain.merge.merge_model_grids", fake_merge)

    exit_code = cli.main(["subdomains", "model", "merge", str(setup_dir)])

    assert exit_code == 0
    assert called["manifest_path"] == setup_dir / "subdomains" / "model" / "subdomain_manifest.json"


def test_subdomain_project_must_follow_setup_projects_layout(tmp_path: Path) -> None:
    project_dir = tmp_path / "winter"
    project_dir.mkdir()

    assert cli.main(["subdomains", "merge", str(project_dir)]) == 1

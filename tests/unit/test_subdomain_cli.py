from __future__ import annotations

from pathlib import Path

from openamundsen_da.subdomain import cli as subdomain_cli


def test_prepare_uses_setup_default_subdomain_root(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    regions = tmp_path / "regions.gpkg"
    setup_dir.mkdir(parents=True, exist_ok=True)
    project_dir.mkdir(parents=True, exist_ok=True)
    regions.write_text("", encoding="utf-8")

    called: dict = {}

    def _fake_prepare_subdomains(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("openamundsen_da.subdomain.prepare.prepare_subdomains", _fake_prepare_subdomains)

    rc = subdomain_cli.cli(
        [
            "prepare",
            "--setup-dir",
            str(setup_dir),
            "--project-dir",
            str(project_dir),
            "--regions",
            str(regions),
        ]
    )

    assert rc == 0
    assert called["setup_dir"] == setup_dir
    assert called["project_dir"] == project_dir
    assert called["subdomain_root"] == setup_dir / "subdomains"


def test_run_resolves_manifest_from_setup_dir(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    manifest = setup_dir / "subdomains" / "subdomain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    called: dict = {}

    def _fake_run_subdomains(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("openamundsen_da.subdomain.run.run_subdomains", _fake_run_subdomains)

    rc = subdomain_cli.cli(["run", "--setup-dir", str(setup_dir), "--no-perf-monitor"])

    assert rc == 0
    assert called["manifest_path"] == manifest


def test_merge_uses_default_output_layout(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    manifest = setup_dir / "subdomains" / "subdomain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    called_grids: dict = {}
    called_points: dict = {}

    def _fake_merge_grids(**kwargs):
        called_grids.update(kwargs)
        return []

    def _fake_merge_points(**kwargs):
        called_points.update(kwargs)
        return []

    monkeypatch.setattr("openamundsen_da.subdomain.merge.merge_grids", _fake_merge_grids)
    monkeypatch.setattr("openamundsen_da.subdomain.merge.merge_points", _fake_merge_points)

    rc = subdomain_cli.cli(["merge", "--setup-dir", str(setup_dir)])

    assert rc == 0
    assert called_grids["out_dir"] == setup_dir / "subdomains" / "merged" / "grids"
    assert called_points["out_dir"] == setup_dir / "subdomains" / "merged" / "points"


def test_resolve_manifest_from_subdomain_root(tmp_path: Path) -> None:
    subdomain_root = tmp_path / "custom_subdomains"
    manifest = subdomain_root / "subdomain_manifest.json"
    subdomain_root.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    resolved = subdomain_cli._resolve_manifest(
        manifest_arg=None,
        setup_dir=None,
        subdomain_root=subdomain_root,
    )
    assert resolved == manifest

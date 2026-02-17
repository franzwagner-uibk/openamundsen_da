from __future__ import annotations

from pathlib import Path
import sys
import types

from openamundsen_da.subdomain import cli as subdomain_cli


def _write_project_yaml(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / f"{project_dir.name}.yml").write_text(
        "start_date: '2022-10-01'\nend_date: '2022-10-02'\ndata_assimilation: {}\n",
        encoding="utf-8",
    )


def test_prepare_uses_setup_default_subdomain_root(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    regions = tmp_path / "regions.gpkg"
    setup_dir.mkdir(parents=True, exist_ok=True)
    _write_project_yaml(project_dir)
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
    assert called["subdomain_root"] == project_dir / "subdomains"


def test_run_resolves_manifest_from_project_dir(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)
    manifest = project_dir / "subdomains" / "subdomain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    called: dict = {}

    def _fake_run_subdomains(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("openamundsen_da.subdomain.run.run_subdomains", _fake_run_subdomains)

    rc = subdomain_cli.cli(["run", "--project-dir", str(project_dir), "--no-perf-monitor"])

    assert rc == 0
    assert called["manifest_path"] == manifest


def test_merge_uses_default_output_layout(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)
    manifest = project_dir / "subdomains" / "subdomain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "{}",
        encoding="utf-8",
    )

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
    monkeypatch.setattr(
        "openamundsen_da.subdomain.manifest.SubdomainManifest.load",
        lambda _path: type("M", (), {"project_dir": project_dir})(),
    )

    rc = subdomain_cli.cli(["merge", "--project-dir", str(project_dir)])

    assert rc == 0
    assert called_grids["out_dir"] == project_dir / "merged" / "grids"
    assert called_points["out_dir"] == project_dir / "merged" / "points"


def test_resolve_manifest_from_subdomain_root(tmp_path: Path) -> None:
    subdomain_root = tmp_path / "custom_subdomains"
    manifest = subdomain_root / "subdomain_manifest.json"
    subdomain_root.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    resolved = subdomain_cli._resolve_manifest(
        manifest_arg=None,
        project_dir=None,
        subdomain_root=subdomain_root,
    )
    assert resolved == manifest


def test_plot_defaults_to_snow_depth_obs_column(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)
    manifest = project_dir / "subdomains" / "subdomain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    called: dict = {}

    def _fake_plot_station_comparisons(**kwargs):
        called.update(kwargs)
        return []

    fake_plot_module = types.SimpleNamespace(plot_station_comparisons=_fake_plot_station_comparisons)
    monkeypatch.setitem(sys.modules, "openamundsen_da.subdomain.plot", fake_plot_module)

    rc = subdomain_cli.cli(["plot", "--project-dir", str(project_dir)])

    assert rc == 0
    assert called["manifest_path"] == manifest
    assert called["obs_column"] == "snow_depth"

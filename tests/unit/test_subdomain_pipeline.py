from __future__ import annotations

from pathlib import Path

from openamundsen_da.subdomain import pipeline as pipeline_mod


def _pipeline_paths(tmp_path: Path) -> dict[str, Path]:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    regions_path = setup_dir / "env" / "subdomains.gpkg"
    subdomain_root = project_dir / "subdomains"
    project_dir.mkdir(parents=True)
    regions_path.parent.mkdir(parents=True)
    regions_path.write_bytes(b"")
    return {
        "setup_dir": setup_dir,
        "project_dir": project_dir,
        "regions_path": regions_path,
        "subdomain_root": subdomain_root,
    }


def _patch_common_pipeline_steps(monkeypatch, order: list[str]) -> None:
    monkeypatch.setattr(pipeline_mod, "ensure_run_mode", lambda *args, **kwargs: "subdomain")
    monkeypatch.setattr(
        pipeline_mod,
        "prepare_subdomains",
        lambda **kwargs: order.append("prepare"),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "run_subdomains",
        lambda **kwargs: order.append("run"),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "write_subdomain_reports",
        lambda **kwargs: order.append("subdomain_reports"),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "render_project_report_best_effort",
        lambda project_dir: order.append("report"),
    )


def test_subdomain_pipeline_defers_cleanup_until_after_top_level_maps(monkeypatch, tmp_path: Path) -> None:
    paths = _pipeline_paths(tmp_path)
    order: list[str] = []
    merge_kwargs: dict = {}

    _patch_common_pipeline_steps(monkeypatch, order)

    def _merge(**kwargs):
        order.append("merge")
        merge_kwargs.update(kwargs)

    monkeypatch.setattr(pipeline_mod, "merge_grids", _merge)
    monkeypatch.setattr(pipeline_mod, "project_maps_enabled", lambda project_dir: True)
    monkeypatch.setattr(
        pipeline_mod,
        "render_project_maps",
        lambda project_dir: order.append("maps") or [Path(project_dir) / "results" / "maps" / "map.png"],
    )
    monkeypatch.setattr(
        pipeline_mod,
        "cleanup_deferred_compact_grid_artifacts",
        lambda **kwargs: order.append("cleanup") or (3, 1000),
    )

    pipeline_mod.run_pipeline(**paths, perf_monitor=False)

    assert merge_kwargs["defer_compact_cleanup"] is True
    assert order == ["prepare", "run", "subdomain_reports", "merge", "maps", "cleanup", "report"]


def test_subdomain_pipeline_keeps_deferred_grids_when_top_level_maps_fail(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _pipeline_paths(tmp_path)
    order: list[str] = []

    _patch_common_pipeline_steps(monkeypatch, order)
    monkeypatch.setattr(pipeline_mod, "merge_grids", lambda **kwargs: order.append("merge"))
    monkeypatch.setattr(pipeline_mod, "project_maps_enabled", lambda project_dir: True)

    def _fail_maps(project_dir):
        order.append("maps")
        raise RuntimeError("render failed")

    monkeypatch.setattr(pipeline_mod, "render_project_maps", _fail_maps)
    monkeypatch.setattr(
        pipeline_mod,
        "cleanup_deferred_compact_grid_artifacts",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("cleanup should wait for successful maps")),
    )

    pipeline_mod.run_pipeline(**paths, perf_monitor=False)

    assert order == ["prepare", "run", "subdomain_reports", "merge", "maps", "report"]


def test_subdomain_pipeline_runs_final_cleanup_when_plotting_is_explicitly_skipped(
    monkeypatch,
    tmp_path: Path,
) -> None:
    paths = _pipeline_paths(tmp_path)
    order: list[str] = []

    _patch_common_pipeline_steps(monkeypatch, order)
    monkeypatch.setattr(pipeline_mod, "merge_grids", lambda **kwargs: order.append("merge"))
    monkeypatch.setattr(
        pipeline_mod,
        "project_maps_enabled",
        lambda project_dir: (_ for _ in ()).throw(AssertionError("project maps should not be inspected")),
    )
    monkeypatch.setattr(
        pipeline_mod,
        "cleanup_deferred_compact_grid_artifacts",
        lambda **kwargs: order.append("cleanup") or (3, 1000),
    )

    pipeline_mod.run_pipeline(**paths, skip_plot=True, perf_monitor=False)

    assert order == ["prepare", "run", "subdomain_reports", "merge", "cleanup", "report"]

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from openamundsen_da.subdomain import merge as merge_mod


def test_merge_grids_uses_compact_da_summary(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    sub_project_dir = project_dir / "subdomains" / "sd_01" / "projects" / "project_2022_2023"
    compact_da = sub_project_dir / "results" / "grids" / "da_output_grids.nc"
    compact_da.parent.mkdir(parents=True, exist_ok=True)
    compact_da.write_bytes(b"compact")

    sub = SimpleNamespace(id="sd_01", project_dir=sub_project_dir)
    manifest = SimpleNamespace(
        run_mode="subdomain",
        project_dir=project_dir,
        subdomains={"sd_01": sub},
        grid_rows=1,
        grid_cols=1,
        grid_transform=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        crs="EPSG:31254",
    )

    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(merge_mod, "ensure_run_mode", lambda *args, **kwargs: "subdomain")
    monkeypatch.setattr(
        merge_mod,
        "_expected_coverage_mask",
        lambda *_args, **_kwargs: np.zeros((1, 1), dtype=bool),
    )
    monkeypatch.setattr(merge_mod, "output_retention_mode", lambda *_args, **_kwargs: "full")
    monkeypatch.setattr(merge_mod, "write_da_output_grids", lambda **_kwargs: None)

    calls: list[tuple[str, list[Path]]] = []

    def _fake_merge_netcdf(*, output_name: str, nc_paths, out_dir: Path, **kwargs):
        source_paths = [Path(path) for _, path in nc_paths]
        calls.append((output_name, source_paths))
        out_path = Path(out_dir) / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"merged")
        return out_path

    monkeypatch.setattr(merge_mod, "_merge_netcdf", _fake_merge_netcdf)

    written = merge_mod.merge_grids(manifest_path=tmp_path / "manifest.json")

    merged_da = project_dir / "results" / "grids" / "da_output_grids.nc"
    assert merged_da.is_file()
    assert written == [merged_da]
    assert calls == [("da_output_grids.nc", [compact_da])]

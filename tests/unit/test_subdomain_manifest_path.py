from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openamundsen_da.subdomain import merge as merge_mod
from openamundsen_da.subdomain.manifest import require_canonical_manifest_path


def test_manifest_path_rejects_mixed_container_mount_aliases(tmp_path: Path) -> None:
    canonical = tmp_path / "data" / "project" / "subdomains"
    manifest = SimpleNamespace(subdomain_root=canonical)

    assert require_canonical_manifest_path(
        manifest,
        canonical / "subdomain_manifest.json",
    ) == canonical / "subdomain_manifest.json"

    with pytest.raises(ValueError, match="different setup/project alias"):
        require_canonical_manifest_path(
            manifest,
            tmp_path / "setup" / "project" / "subdomains" / "subdomain_manifest.json",
        )


def test_tracked_merge_rejects_alias_before_operation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "data" / "project" / "subdomains"
    alias = tmp_path / "setup" / "project" / "subdomains" / "subdomain_manifest.json"
    manifest = SimpleNamespace(subdomain_root=canonical)
    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda _cls, _path: manifest),
    )
    called = False

    def operation(*, manifest_path: Path) -> list[Path]:
        nonlocal called
        called = True
        return []

    operation.__name__ = "merge_grids"
    wrapped = merge_mod._tracked_merge(operation)

    with pytest.raises(ValueError, match="different setup/project alias"):
        wrapped(manifest_path=alias)
    assert called is False

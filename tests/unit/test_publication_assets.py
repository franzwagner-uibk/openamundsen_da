from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
from PIL import Image


def _load_release_module(name: str, monkeypatch):
    scripts_dir = Path(__file__).parents[2] / "scripts" / "release"
    monkeypatch.syspath_prepend(str(scripts_dir))
    script = scripts_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manuscript_profile_replaces_broad_mirror_with_three_declared_outputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_release_module("render_manuscript_profile", monkeypatch)
    root = tmp_path / "setup"
    project_dir = root / "projects" / module.PROJECT_NAME
    paper_root = project_dir / "results" / "paper"
    stale = paper_root / "plots" / "results" / "result_overview.png"
    stale.parent.mkdir(parents=True)
    stale.write_text("stale", encoding="utf-8")

    def _fake_weights(_project_dir, *, output, show_figure_title):
        assert show_figure_title is False
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("weights", encoding="utf-8")
        return output

    def _fake_maps(*, project_dir, output_root, names, strip_figure_titles):
        assert names == {"da_7", "da_8"}
        assert strip_figure_titles is True
        outputs = []
        for name in sorted(names):
            output = output_root / "da_events" / f"{name}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(name, encoding="utf-8")
            outputs.append(output)
        return outputs

    monkeypatch.setattr(module, "plot_setup_weights_overview", _fake_weights)
    monkeypatch.setattr(module, "render_project_map_profile", _fake_maps)

    outputs = module.render_manuscript_profile(root)

    assert tuple(path.relative_to(paper_root) for path in outputs) == tuple(
        sorted(module.PROFILE_RELATIVE_OUTPUTS)
    )
    assert module._existing_profile_files(paper_root) == tuple(
        sorted(module.PROFILE_RELATIVE_OUTPUTS)
    )
    assert not stale.exists()


def test_manuscript_profile_keeps_previous_outputs_when_rendering_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_release_module("render_manuscript_profile", monkeypatch)
    root = tmp_path / "setup"
    project_dir = root / "projects" / module.PROJECT_NAME
    paper_root = project_dir / "results" / "paper"
    previous = paper_root / "maps" / "da_events" / "da_7.png"
    previous.parent.mkdir(parents=True)
    previous.write_text("previous", encoding="utf-8")

    def _fail_weights(*_args, **_kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(module, "plot_setup_weights_overview", _fail_weights)

    with pytest.raises(RuntimeError, match="render failed"):
        module.render_manuscript_profile(root)

    assert previous.read_text(encoding="utf-8") == "previous"
    assert not list((project_dir / "results").glob(".manuscript-profile-*"))


def test_manuscript_profile_restores_previous_outputs_when_swap_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_release_module("render_manuscript_profile", monkeypatch)
    root = tmp_path / "setup"
    project_dir = root / "projects" / module.PROJECT_NAME
    paper_root = project_dir / "results" / "paper"
    previous = paper_root / "maps" / "da_events" / "da_7.png"
    previous.parent.mkdir(parents=True)
    previous.write_text("previous", encoding="utf-8")

    def _fake_weights(_project_dir, *, output, show_figure_title):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("weights", encoding="utf-8")
        return output

    def _fake_maps(*, project_dir, output_root, names, strip_figure_titles):
        for name in sorted(names):
            output = output_root / "da_events" / f"{name}.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(name, encoding="utf-8")

    original_replace = module.Path.replace

    def _fail_install(path, target):
        if (
            path.name.startswith(".manuscript-profile-")
            and not path.name.startswith(".manuscript-profile-backup-")
            and target == paper_root
        ):
            raise OSError("swap failed")
        return original_replace(path, target)

    monkeypatch.setattr(module, "plot_setup_weights_overview", _fake_weights)
    monkeypatch.setattr(module, "render_project_map_profile", _fake_maps)
    monkeypatch.setattr(module.Path, "replace", _fail_install)

    with pytest.raises(OSError, match="swap failed"):
        module.render_manuscript_profile(root)

    assert previous.read_text(encoding="utf-8") == "previous"
    assert not list((project_dir / "results").glob(".manuscript-profile-*"))


def test_publication_staging_copies_only_selected_assets_and_keeps_extras(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_release_module("stage_publication_assets", monkeypatch)
    root = tmp_path / "run"
    source = root / "results" / "figure.png"
    source.parent.mkdir(parents=True)
    Image.new("RGBA", (3, 2), (1, 2, 3, 255)).save(source)
    destination = tmp_path / "assets"
    destination.mkdir()
    manual = destination / "fig01.png"
    Image.new("RGBA", (2, 2), (9, 8, 7, 255)).save(manual)
    extra = destination / "keep-me.png"
    extra.write_bytes(b"unlisted")
    manifest = {
        "figures": [
            {"name": "fig01.png", "source": "manual", **module._image_record(manual)},
            {
                "name": "fig03.png",
                "source": "results/figure.png",
                **module._image_record(source),
            },
        ]
    }

    actions, errors = module.plan_stage(
        root=root,
        destination=destination,
        manifest=manifest,
        target="manuscript",
    )

    assert errors == ()
    assert [(action.destination.name, action.operation) for action in actions] == [
        ("fig01.png", "VALIDATE"),
        ("fig03.png", "COPY"),
    ]
    assert not (destination / "fig03.png").exists()

    copied = module.apply_stage(actions, errors)

    assert copied == (destination / "fig03.png",)
    assert module._image_record(destination / "fig03.png") == module._image_record(source)
    assert extra.read_bytes() == b"unlisted"


def test_publication_staging_uses_separate_target_manifests(monkeypatch) -> None:
    module = _load_release_module("stage_publication_assets", monkeypatch)

    assert module._manifest_path("manuscript") == module.DEFAULT_ASSET_MANIFEST
    assert module._manifest_path("tutorial") == module.DEFAULT_TUTORIAL_ASSET_MANIFEST
    assert module._manifest_path("tutorial", Path("custom.json")) == Path("custom.json")


def test_publication_staging_blocks_source_hash_drift(tmp_path: Path, monkeypatch) -> None:
    module = _load_release_module("stage_publication_assets", monkeypatch)
    root = tmp_path / "run"
    source = root / "results" / "figure.png"
    source.parent.mkdir(parents=True)
    Image.new("RGBA", (3, 2), (1, 2, 3, 255)).save(source)
    record = {
        "destination": "figure.png",
        "source": "results/figure.png",
        **module._image_record(source),
    }
    Image.new("RGBA", (3, 2), (4, 5, 6, 255)).save(source)

    actions, errors = module.plan_stage(
        root=root,
        destination=tmp_path / "tutorial",
        manifest={"tutorial_assets": [record]},
        target="tutorial",
    )

    assert actions[0].operation == "BLOCKED"
    assert any("selected source differs" in error for error in errors)
    assert not (tmp_path / "tutorial" / "figure.png").exists()


def test_publication_staging_preserves_canonical_destination_for_accepted_variant(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_release_module("stage_publication_assets", monkeypatch)
    root = tmp_path / "run"
    source = root / "results" / "figure.png"
    source.parent.mkdir(parents=True)
    Image.new("RGBA", (3, 2), (4, 5, 6, 255)).save(source)
    destination = tmp_path / "assets"
    destination.mkdir()
    canonical = destination / "figure.png"
    Image.new("RGBA", (3, 2), (1, 2, 3, 255)).save(canonical)
    record = {
        "destination": "figure.png",
        "source": "results/figure.png",
        **module._image_record(canonical),
        "accepted_run_records": [module._image_record(source)],
    }

    actions, errors = module.plan_stage(
        root=root,
        destination=destination,
        manifest={"tutorial_assets": [record]},
        target="tutorial",
    )

    assert errors == ()
    assert actions[0].operation == "PRESERVE"
    assert module.apply_stage(actions, errors) == ()
    assert module._image_record(canonical) == {
        key: record[key] for key in module.IMAGE_KEYS
    }


def test_publication_staging_blocks_accepted_variant_without_canonical_destination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_release_module("stage_publication_assets", monkeypatch)
    root = tmp_path / "run"
    source = root / "results" / "figure.png"
    source.parent.mkdir(parents=True)
    Image.new("RGBA", (3, 2), (4, 5, 6, 255)).save(source)
    canonical = tmp_path / "canonical.png"
    Image.new("RGBA", (3, 2), (1, 2, 3, 255)).save(canonical)
    record = {
        "destination": "figure.png",
        "source": "results/figure.png",
        **module._image_record(canonical),
        "accepted_run_records": [module._image_record(source)],
    }

    actions, errors = module.plan_stage(
        root=root,
        destination=tmp_path / "assets",
        manifest={"tutorial_assets": [record]},
        target="tutorial",
    )

    assert actions[0].operation == "BLOCKED"
    assert any("canonical destination differs" in error for error in errors)


def test_publication_staging_preserves_runtime_specific_plot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_release_module("stage_publication_assets", monkeypatch)
    root = tmp_path / "run"
    source = root / "results" / "performance.png"
    source.parent.mkdir(parents=True)
    Image.new("RGBA", (3, 2), (8, 7, 6, 255)).save(source)
    destination = tmp_path / "assets"
    destination.mkdir()
    canonical = destination / "performance.png"
    Image.new("RGBA", (3, 2), (1, 2, 3, 255)).save(canonical)
    record = {
        "destination": "performance.png",
        "source": "results/performance.png",
        "source_policy": "runtime_specific",
        **module._image_record(canonical),
    }

    actions, errors = module.plan_stage(
        root=root,
        destination=destination,
        manifest={"tutorial_assets": [record]},
        target="tutorial",
    )

    assert errors == ()
    assert actions[0].operation == "PRESERVE"
    assert module.apply_stage(actions, errors) == ()

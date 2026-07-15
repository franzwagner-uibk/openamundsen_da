"""One supported command tree for openAMUNDSEN-DA workflows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from openamundsen_da.exceptions import OpenAmundsenDAError


def _version() -> str:
    try:
        return version("openamundsen-da")
    except PackageNotFoundError:
        from openamundsen_da import __version__

        return __version__


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable JSON envelope")


def _project_leaf(
    subparsers: argparse._SubParsersAction,
    name: str,
    *,
    help_text: str,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name, help=help_text, description=help_text)
    parser.add_argument("project_dir", type=Path, metavar="PROJECT_DIR")
    _add_json(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the complete supported CLI parser."""
    parser = argparse.ArgumentParser(
        prog="openamundsen-da",
        description="Prepare, run and inspect openAMUNDSEN data-assimilation projects.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    commands = parser.add_subparsers(dest="command", required=True)

    observations = commands.add_parser("observations", help="Preprocess configured observation products")
    observation_commands = observations.add_subparsers(dest="observation_command", required=True)
    snow = _project_leaf(
        observation_commands,
        "snow-cover",
        help_text="Preprocess configured snow-cover observations",
    )
    snow.add_argument("--overwrite", action="store_true")
    wet = _project_leaf(
        observation_commands,
        "wet-snow",
        help_text="Preprocess configured wet-snow observations",
    )
    wet.add_argument("--overwrite", action="store_true")

    prepare = _project_leaf(commands, "prepare", help_text="Prepare deterministic project steps and observations")
    prepare.add_argument("--overwrite", action="store_true")
    run = _project_leaf(commands, "run", help_text="Run a prepared single-domain project")
    run.add_argument("--max-workers", type=_positive_int)
    render = _project_leaf(commands, "render", help_text="Regenerate configured project outputs")
    render.add_argument("--max-workers", type=_positive_int)
    clean = _project_leaf(commands, "clean", help_text="Preview safe heavy restart-artifact cleanup")
    clean.add_argument("--apply", action="store_true", help="Apply the previewed cleanup")

    subdomains = commands.add_parser("subdomains", help="Run explicit subdomain workflows")
    subdomain_commands = subdomains.add_subparsers(dest="subdomain_command", required=True)
    sub_prepare = _project_leaf(
        subdomain_commands,
        "prepare",
        help_text="Prepare DA subdomains for a project",
    )
    sub_prepare.add_argument("--regions", type=Path)
    sub_prepare.add_argument("--station-buffer-km", type=float, default=50.0)
    sub_prepare.add_argument("--grid-buffer-m", type=float)
    sub_prepare.add_argument("--overwrite", action="store_true")
    sub_run = _project_leaf(subdomain_commands, "run", help_text="Run prepared DA subdomains")
    sub_run.add_argument("--max-workers", type=_positive_int)
    sub_run.add_argument("--inner-max-workers", type=_positive_int)
    sub_run.add_argument("--overwrite", action="store_true")
    sub_merge = _project_leaf(subdomain_commands, "merge", help_text="Merge compact DA subdomain outputs")
    sub_merge.add_argument("--coverage-sliver-tol-px", type=int, default=4)
    sub_merge.add_argument("--out-dir", type=Path)
    sub_render = _project_leaf(subdomain_commands, "render", help_text="Render merged DA subdomain outputs")
    sub_render.add_argument("--max-workers", type=_positive_int)

    model = subdomain_commands.add_parser("model", help="Tile one plain openAMUNDSEN simulation")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    model_prepare = model_commands.add_parser("prepare", help="Prepare plain-model subdomains")
    model_prepare.add_argument("setup_dir", type=Path, metavar="SETUP_DIR")
    model_prepare.add_argument("--regions", type=Path)
    model_prepare.add_argument("--station-buffer-km", type=float, default=50.0)
    model_prepare.add_argument("--grid-buffer-m", type=float)
    model_prepare.add_argument("--overwrite", action="store_true")
    _add_json(model_prepare)
    model_run = model_commands.add_parser("run", help="Run plain-model subdomains")
    model_run.add_argument("setup_dir", type=Path, metavar="SETUP_DIR")
    model_run.add_argument("--max-workers", type=_positive_int)
    model_run.add_argument("--overwrite", action="store_true")
    _add_json(model_run)
    model_merge = model_commands.add_parser("merge", help="Merge plain-model subdomain outputs")
    model_merge.add_argument("setup_dir", type=Path, metavar="SETUP_DIR")
    model_merge.add_argument("--coverage-sliver-tol-px", type=int, default=4)
    model_merge.add_argument("--out-dir", type=Path)
    _add_json(model_merge)
    return parser


def _subdomain_setup_dir(project_dir: Path) -> Path:
    if project_dir.parent.name != "projects":
        raise ValueError(f"Subdomain project must use <setup>/projects/<project>: {project_dir}")
    return project_dir.parent.parent.resolve()


def _default_regions_path(setup_dir: Path) -> Path:
    env_dir = setup_dir / "env"
    preferred = (env_dir / "subdomains.gpkg", env_dir / "roi.gpkg")
    return next((path for path in preferred if path.is_file()), preferred[0])


def _dispatch_subdomains(args: argparse.Namespace) -> object:
    if args.subdomain_command == "model":
        from openamundsen_da.subdomain.merge import merge_model_grids
        from openamundsen_da.subdomain.model import run_model_subdomains
        from openamundsen_da.subdomain.prepare import prepare_model_subdomains

        setup_dir = args.setup_dir.resolve()
        manifest_path = setup_dir / "subdomains" / "model" / "subdomain_manifest.json"
        if args.model_command == "prepare":
            manifest = prepare_model_subdomains(
                setup_dir=setup_dir,
                regions_path=(args.regions or _default_regions_path(setup_dir)).resolve(),
                station_buffer_m=float(args.station_buffer_km) * 1000.0,
                grid_buffer_m=args.grid_buffer_m,
                overwrite=args.overwrite,
            )
            return {
                "status": "completed",
                "manifest_path": manifest.subdomain_root / "subdomain_manifest.json",
                "subdomain_count": len(manifest.subdomains),
            }
        if args.model_command == "run":
            results = run_model_subdomains(
                manifest_path=manifest_path,
                max_workers=args.max_workers,
                overwrite=args.overwrite,
            )
            return {
                "status": "completed",
                "manifest_path": manifest_path,
                "completed": sum(result.status == "success" for result in results),
                "reused": sum(result.status == "skipped" for result in results),
            }
        outputs = merge_model_grids(
            manifest_path=manifest_path,
            coverage_sliver_tol_px=args.coverage_sliver_tol_px,
            out_dir=args.out_dir,
        )
        return {"status": "completed", "manifest_path": manifest_path, "outputs": outputs}

    project_dir = args.project_dir.resolve()
    manifest_path = project_dir / "subdomains" / "subdomain_manifest.json"
    if args.subdomain_command == "render":
        from openamundsen_da.subdomain.render import render_subdomain_outputs

        return render_subdomain_outputs(project_dir, max_workers=args.max_workers)
    setup_dir = _subdomain_setup_dir(project_dir)
    if args.subdomain_command == "prepare":
        from openamundsen_da.subdomain.prepare import prepare_subdomains

        manifest = prepare_subdomains(
            setup_dir=setup_dir,
            project_dir=project_dir,
            regions_path=(args.regions or _default_regions_path(setup_dir)).resolve(),
            station_buffer_m=float(args.station_buffer_km) * 1000.0,
            grid_buffer_m=args.grid_buffer_m,
            overwrite=args.overwrite,
        )
        return {
            "status": "completed",
            "manifest_path": manifest.subdomain_root / "subdomain_manifest.json",
            "subdomain_count": len(manifest.subdomains),
        }
    if args.subdomain_command == "run":
        from openamundsen_da.subdomain.run import run_subdomains

        results = run_subdomains(
            manifest_path=manifest_path,
            max_workers=args.max_workers,
            inner_max_workers=args.inner_max_workers,
            overwrite=args.overwrite,
        )
        return {
            "status": "completed",
            "manifest_path": manifest_path,
            "completed": sum(result.status == "success" for result in results),
            "reused": sum(result.status == "skipped" for result in results),
        }

    from openamundsen_da.subdomain.merge import merge_grids

    outputs = merge_grids(
        manifest_path=manifest_path,
        coverage_sliver_tol_px=args.coverage_sliver_tol_px,
        out_dir=args.out_dir,
    )
    return {"status": "completed", "manifest_path": manifest_path, "outputs": outputs}


def _dispatch(args: argparse.Namespace) -> object:
    if args.command == "observations":
        from openamundsen_da.observations import preprocess_snow_cover, preprocess_wet_snow

        operation = preprocess_snow_cover if args.observation_command == "snow-cover" else preprocess_wet_snow
        return operation(args.project_dir, overwrite=args.overwrite)
    if args.command == "subdomains":
        return _dispatch_subdomains(args)
    from openamundsen_da.api import clean_project, prepare_project, render_project, run_project

    operations: dict[str, Callable[..., object]] = {
        "prepare": prepare_project,
        "run": run_project,
        "render": render_project,
        "clean": clean_project,
    }
    kwargs: dict[str, object] = {}
    if args.command == "prepare":
        kwargs["overwrite"] = args.overwrite
    elif args.command in {"run", "render"}:
        kwargs["max_workers"] = args.max_workers
    elif args.command == "clean":
        kwargs["apply"] = args.apply
    return operations[args.command](args.project_dir, **kwargs)


def _human_summary(result: object) -> str:
    data = _jsonable(result)
    if not isinstance(data, dict):
        return str(data)
    status = data.get("status", "completed")
    primary = data.get("summary_path") or data.get("manifest_path") or data.get("project_dir")
    return f"{status}: {primary}" if primary else str(status)


def main(argv: list[str] | None = None) -> int:
    """Execute one CLI operation and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    command = " ".join(
        str(value)
        for value in (
            args.command,
            getattr(args, "observation_command", None),
            getattr(args, "subdomain_command", None),
            getattr(args, "model_command", None),
        )
        if value is not None
    )
    try:
        result = _dispatch(args)
    except (OpenAmundsenDAError, OSError, RuntimeError, ValueError) as exc:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "command": command,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                    sort_keys=True,
                )
            )
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "command": command, "result": _jsonable(result)}, sort_keys=True))
    else:
        print(_human_summary(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]

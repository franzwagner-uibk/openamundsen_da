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


def _append_option(arguments: list[str], name: str, value: object | None) -> None:
    if value is not None:
        arguments.extend([name, str(value)])


def _run_legacy_subdomain(arguments: list[str]) -> dict[str, Any]:
    from openamundsen_da.subdomain.cli import cli as subdomain_cli

    result = subdomain_cli(arguments)
    if result not in (None, 0):
        raise RuntimeError(f"Subdomain operation failed with exit code {result}")
    return {"status": "completed"}


def _dispatch_subdomains(args: argparse.Namespace) -> object:
    if args.subdomain_command == "model":
        setup_dir = args.setup_dir.resolve()
        legacy = [f"model-{args.model_command}", "--setup-dir", str(setup_dir)]
        if args.model_command == "prepare":
            _append_option(legacy, "--regions", args.regions)
            _append_option(legacy, "--station-buffer-km", args.station_buffer_km)
            _append_option(legacy, "--grid-buffer-m", args.grid_buffer_m)
        elif args.model_command == "run":
            _append_option(legacy, "--max-workers", args.max_workers)
        elif args.model_command == "merge":
            _append_option(legacy, "--coverage-sliver-tol-px", args.coverage_sliver_tol_px)
            _append_option(legacy, "--out-dir", args.out_dir)
        if getattr(args, "overwrite", False):
            legacy.append("--overwrite")
        return _run_legacy_subdomain(legacy)

    project_dir = args.project_dir.resolve()
    if args.subdomain_command == "render":
        from openamundsen_da.subdomain.render import render_subdomain_outputs

        return render_subdomain_outputs(project_dir, max_workers=args.max_workers)
    setup_dir = project_dir.parent.parent.resolve()
    legacy = [args.subdomain_command, "--project-dir", str(project_dir)]
    if args.subdomain_command == "prepare":
        legacy.extend(["--setup-dir", str(setup_dir)])
        _append_option(legacy, "--regions", args.regions)
        _append_option(legacy, "--station-buffer-km", args.station_buffer_km)
        _append_option(legacy, "--grid-buffer-m", args.grid_buffer_m)
    elif args.subdomain_command == "run":
        _append_option(legacy, "--max-workers", args.max_workers)
        _append_option(legacy, "--inner-max-workers", args.inner_max_workers)
    elif args.subdomain_command == "merge":
        _append_option(legacy, "--coverage-sliver-tol-px", args.coverage_sliver_tol_px)
        _append_option(legacy, "--out-dir", args.out_dir)
    if getattr(args, "overwrite", False):
        legacy.append("--overwrite")
    return _run_legacy_subdomain(legacy)


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

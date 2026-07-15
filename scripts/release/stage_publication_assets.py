#!/usr/bin/env python3
"""Preview or stage manifest-selected manuscript and tutorial image assets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

from validate_manuscript_reference import DEFAULT_ASSET_MANIFEST, _image_record, _read_json


class PublicationAssetError(RuntimeError):
    """Raised when a selected publication asset is missing or differs."""


@dataclass(frozen=True)
class StageAction:
    destination: Path
    source: Path | None
    operation: str


def _relative_path(value: Any, *, field: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise PublicationAssetError(f"Invalid relative {field}: {value!r}")
    return path


IMAGE_KEYS = ("file_sha256", "width", "height", "mode", "pixels_sha256")
LAYOUT_KEYS = ("width", "height", "mode")


def _image_differences(
    path: Path,
    record: Mapping[str, Any],
    *,
    keys: Sequence[str] = IMAGE_KEYS,
) -> list[str]:
    if not path.is_file():
        return [f"missing image: {path}"]
    actual = _image_record(path)
    differences: list[str] = []
    for key in keys:
        if actual[key] != record[key]:
            differences.append(
                f"{path}:{key}: expected {record[key]!r}, got {actual[key]!r}"
            )
    return differences


def _matches_record(path: Path, record: Mapping[str, Any]) -> bool:
    if not path.is_file():
        return False
    actual = _image_record(path)
    return all(actual[key] == record[key] for key in IMAGE_KEYS)


def _target_records(manifest: Mapping[str, Any], target: str) -> tuple[Mapping[str, Any], ...]:
    if target == "manuscript":
        records = manifest.get("figures", ())
    elif target == "tutorial":
        records = manifest.get("tutorial_assets", ())
    else:  # pragma: no cover - argparse and callers constrain this value
        raise PublicationAssetError(f"Unsupported publication target: {target}")
    if not isinstance(records, list) or not records:
        raise PublicationAssetError(f"Asset manifest has no {target} records")
    return tuple(records)


def plan_stage(
    *,
    root: Path,
    destination: Path,
    manifest: Mapping[str, Any],
    target: str,
) -> tuple[tuple[StageAction, ...], tuple[str, ...]]:
    """Return a deterministic, non-mutating staging plan and validation errors."""
    root = Path(root).resolve(strict=True)
    destination = Path(destination).resolve()
    actions: list[StageAction] = []
    errors: list[str] = []

    for record in _target_records(manifest, target):
        destination_value = record.get("destination", record.get("name"))
        relative_destination = _relative_path(destination_value, field="destination")
        destination_path = destination / relative_destination
        source_value = record.get("source")

        if source_value == "manual":
            differences = _image_differences(destination_path, record)
            if differences:
                errors.extend(f"immutable manual asset differs: {item}" for item in differences)
            actions.append(StageAction(destination_path, None, "VALIDATE"))
            continue

        relative_source = _relative_path(source_value, field="source")
        source_path = root / relative_source
        source_policy = str(record.get("source_policy", "exact"))
        if source_policy not in {"exact", "runtime_specific"}:
            errors.append(
                f"unsupported source policy for {source_path}: {source_policy!r}"
            )
            actions.append(StageAction(destination_path, source_path, "BLOCKED"))
            continue
        source_differences = _image_differences(source_path, record)
        accepted_variant = any(
            _matches_record(source_path, accepted)
            for accepted in record.get("accepted_run_records", ())
        )
        runtime_variant = source_policy == "runtime_specific" and not _image_differences(
            source_path,
            record,
            keys=LAYOUT_KEYS,
        )
        if source_differences and not accepted_variant and not runtime_variant:
            errors.extend(f"selected source differs: {item}" for item in source_differences)
            actions.append(StageAction(destination_path, source_path, "BLOCKED"))
            continue

        if source_differences:
            destination_differences = _image_differences(destination_path, record)
            if destination_differences:
                errors.extend(
                    f"canonical destination differs: {item}" for item in destination_differences
                )
                actions.append(StageAction(destination_path, source_path, "BLOCKED"))
            else:
                actions.append(StageAction(destination_path, source_path, "PRESERVE"))
            continue

        destination_differences = _image_differences(destination_path, record)
        operation = "COPY" if destination_differences else "UNCHANGED"
        actions.append(StageAction(destination_path, source_path, operation))

    actions.sort(key=lambda action: str(action.destination))
    return tuple(actions), tuple(errors)


def apply_stage(actions: Sequence[StageAction], errors: Sequence[str]) -> tuple[Path, ...]:
    """Apply a validated plan without deleting any destination files."""
    if errors:
        raise PublicationAssetError("Asset staging is blocked:\n- " + "\n- ".join(errors))
    copied: list[Path] = []
    for action in actions:
        if action.operation != "COPY":
            continue
        if action.source is None:  # pragma: no cover - guarded by operation
            raise PublicationAssetError(f"Copy action has no source: {action.destination}")
        action.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(action.source, action.destination)
        copied.append(action.destination)
    return tuple(copied)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Validated manuscript reference setup root")
    parser.add_argument("--target", required=True, choices=("manuscript", "tutorial"))
    parser.add_argument(
        "--destination",
        required=True,
        type=Path,
        help="Explicit destination root (for example manuscript assets/ or tutorial image directory)",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument("--apply", action="store_true", help="Copy changed selected assets")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = _read_json(args.manifest.resolve(strict=True))
        actions, errors = plan_stage(
            root=args.root,
            destination=args.destination,
            manifest=manifest,
            target=args.target,
        )
        for action in actions:
            source = f" <- {action.source}" if action.source is not None else ""
            print(f"{action.operation} {action.destination}{source}")
        if errors:
            print("\n".join(errors))
            return 1
        if not args.apply:
            print("Preview only; pass --apply to copy selected changed assets")
            return 0
        copied = apply_stage(actions, errors)
    except (OSError, PublicationAssetError, ValueError) as exc:
        print(f"publication asset staging error: {exc}")
        return 2
    print(f"Copied {len(copied)} selected asset(s); no unlisted files were removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

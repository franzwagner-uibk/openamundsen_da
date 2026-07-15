#!/usr/bin/env python3
"""Capture and compare deterministic scientific-output fingerprints.

CSV values and ordering, decoded NetCDF arrays and decoded image pixels are
compared. Container metadata such as PNG chunks and NetCDF history is ignored.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import rasterio
import xarray as xr
import yaml
from PIL import Image


SCHEMA_VERSION = 1
SUPPORTED_KINDS = {
    "ascii_grid",
    "auto",
    "binary",
    "csv",
    "geotiff",
    "image",
    "json",
    "netcdf",
    "yaml",
}


class FingerprintError(ValueError):
    """Raised when a fingerprint specification or artifact is invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_json(value: Any) -> str:
    return _digest_bytes(_canonical_json(value))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FingerprintError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FingerprintError(f"JSON root must be an object: {path}")
    return value


def _relative_artifact(root: Path, artifact: Path) -> tuple[Path, Path]:
    resolved_root = root.resolve(strict=True)
    if artifact.is_symlink():
        raise FingerprintError(f"Symlink artifacts are not allowed: {artifact}")
    resolved_artifact = artifact.resolve(strict=True)
    try:
        relative = resolved_artifact.relative_to(resolved_root)
    except ValueError as exc:
        raise FingerprintError(f"Artifact escapes the fingerprint root: {artifact}") from exc
    if not resolved_artifact.is_file():
        raise FingerprintError(f"Artifact is not a regular file: {artifact}")
    return relative, resolved_artifact


def _auto_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".asc":
        return "ascii_grid"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image"
    if suffix in {".tif", ".tiff"}:
        return "geotiff"
    if suffix in {".nc", ".nc4", ".netcdf"}:
        return "netcdf"
    if suffix == ".json":
        return "json"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    return "binary"


def _csv_record(path: Path, *, ignore_columns: Sequence[str]) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.reader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise FingerprintError(f"Cannot read CSV file {path}: {exc}") from exc
    if not rows:
        raise FingerprintError(f"CSV file has no header: {path}")
    header = rows[0]
    if len(set(header)) != len(header):
        raise FingerprintError(f"CSV header contains duplicate columns: {path}")
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise FingerprintError(
                f"CSV row {row_number} has {len(row)} fields; expected {len(header)}: {path}"
            )
    ignored = set(ignore_columns)
    keep_indexes = [index for index, name in enumerate(header) if name not in ignored]
    kept_rows = [[row[index] for index in keep_indexes] for row in rows[1:]]
    kept_header = [header[index] for index in keep_indexes]
    return {
        "kind": "csv",
        "columns_sha256": _digest_json(kept_header),
        "ignored_columns": sorted(ignored.intersection(header)),
        "row_count": len(kept_rows),
        "data_sha256": _digest_json(kept_rows),
    }


def _ascii_grid_record(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            header: dict[str, float] = {}
            data_lines: list[str] = []
            header_keys = {
                "ncols",
                "nrows",
                "xllcenter",
                "xllcorner",
                "yllcenter",
                "yllcorner",
                "cellsize",
                "nodata_value",
            }
            for line in stream:
                fields = line.split(maxsplit=1)
                key = fields[0].lower() if fields else ""
                if key in header_keys and len(fields) == 2:
                    header[key] = float(fields[1].replace(",", "."))
                    continue
                data_lines.append(line)
                data_lines.extend(stream)
                break
            values = np.loadtxt((line.replace(",", ".") for line in data_lines))
    except (OSError, UnicodeError, ValueError) as exc:
        raise FingerprintError(f"Cannot read ESRI ASCII grid {path}: {exc}") from exc
    expected_shape = (int(header.get("nrows", -1)), int(header.get("ncols", -1)))
    if values.size == int(np.prod(expected_shape)):
        values = values.reshape(expected_shape)
    if values.shape != expected_shape:
        raise FingerprintError(
            f"ASCII grid shape differs from header for {path}: {values.shape} != {expected_shape}"
        )
    return {
        "kind": "ascii_grid",
        "header": {key: _json_scalar(value) for key, value in sorted(header.items())},
        "data_sha256": _array_digest(values),
    }


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if isinstance(value, complex):
        return {"real": _json_scalar(value.real), "imag": _json_scalar(value.imag)}
    if isinstance(value, (bytes, bytearray)):
        return {"bytes_hex": bytes(value).hex()}
    return str(value)


def _array_digest(values: np.ndarray[Any, Any]) -> str:
    array = np.asarray(values)
    if array.dtype.kind in {"O", "U", "S", "M", "m"}:
        flattened = [_json_scalar(value) for value in array.reshape(-1).tolist()]
        return _digest_json({"shape": list(array.shape), "values": flattened})
    normalized = np.ascontiguousarray(array)
    if normalized.dtype.kind in {"f", "c"} and np.isnan(normalized).any():
        normalized = normalized.copy()
        normalized[np.isnan(normalized)] = np.nan
    if normalized.dtype.byteorder == ">" or (
        normalized.dtype.byteorder == "=" and not np.little_endian
    ):
        normalized = normalized.astype(normalized.dtype.newbyteorder("<"), copy=False)
    header = _canonical_json({"dtype": str(normalized.dtype), "shape": list(normalized.shape)})
    return _digest_bytes(header + b"\0" + normalized.tobytes(order="C"))


def _encoding_record(array: xr.DataArray) -> dict[str, Any]:
    keys = ("dtype", "_FillValue", "zlib", "complevel", "shuffle", "chunksizes")
    return {
        key: _json_scalar(array.encoding[key])
        for key in keys
        if key in array.encoding and array.encoding[key] is not None
    }


def _data_array_record(array: xr.DataArray) -> dict[str, Any]:
    values = np.asarray(array.values)
    return {
        "dimensions": list(array.dims),
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "data_sha256": _array_digest(values),
        "encoding": _encoding_record(array),
    }


def _netcdf_record(path: Path) -> dict[str, Any]:
    try:
        with xr.open_dataset(path, decode_cf=True, mask_and_scale=True) as dataset:
            dataset.load()
            dimensions = {name: int(size) for name, size in sorted(dataset.sizes.items())}
            coordinates = {
                name: _data_array_record(dataset.coords[name]) for name in sorted(dataset.coords)
            }
            variables = {
                name: _data_array_record(dataset[name]) for name in sorted(dataset.data_vars)
            }
    except Exception as exc:
        raise FingerprintError(f"Cannot decode NetCDF file {path}: {exc}") from exc
    return {
        "kind": "netcdf",
        "dimensions": dimensions,
        "coordinates": coordinates,
        "variables": variables,
    }


def _image_record(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            rgba = image.convert("RGBA")
            pixels = rgba.tobytes()
            width, height = rgba.size
    except (OSError, ValueError) as exc:
        raise FingerprintError(f"Cannot decode image file {path}: {exc}") from exc
    return {
        "kind": "image",
        "mode": "RGBA",
        "width": width,
        "height": height,
        "pixels_sha256": _digest_bytes(pixels),
    }


def _geotiff_record(path: Path) -> dict[str, Any]:
    try:
        with rasterio.open(path) as dataset:
            values = dataset.read()
            crs = dataset.crs.to_wkt() if dataset.crs is not None else None
            transform = list(dataset.transform)[:6]
            nodata = [_json_scalar(value) for value in dataset.nodatavals]
            descriptions = list(dataset.descriptions)
            color_interpretation = [item.name for item in dataset.colorinterp]
    except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
        raise FingerprintError(f"Cannot decode GeoTIFF file {path}: {exc}") from exc
    return {
        "kind": "geotiff",
        "bands": int(values.shape[0]),
        "height": int(values.shape[1]),
        "width": int(values.shape[2]),
        "dtype": str(values.dtype),
        "crs_wkt": crs,
        "transform": transform,
        "nodata": nodata,
        "descriptions": descriptions,
        "color_interpretation": color_interpretation,
        "data_sha256": _array_digest(values),
    }


def _drop_json_keys(value: Any, ignored: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_json_keys(item, ignored)
            for key, item in sorted(value.items())
            if key not in ignored
        }
    if isinstance(value, list):
        return [_drop_json_keys(item, ignored) for item in value]
    return value


def _normalize_structure(value: Any) -> Any:
    if isinstance(value, Mapping):
        sorted_items = sorted(value.items(), key=lambda pair: str(pair[0]))
        return {str(key): _normalize_structure(item) for key, item in sorted_items}
    if isinstance(value, (list, tuple)):
        return [_normalize_structure(item) for item in value]
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    return _json_scalar(value)


def _json_record(path: Path, *, ignore_keys: Sequence[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FingerprintError(f"Cannot read JSON artifact {path}: {exc}") from exc
    normalized = _drop_json_keys(value, set(ignore_keys))
    return {
        "kind": "json",
        "ignored_keys": sorted(set(ignore_keys)),
        "data_sha256": _digest_json(normalized),
    }


def _yaml_record(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise FingerprintError(f"Cannot read YAML artifact {path}: {exc}") from exc
    normalized = _normalize_structure(value)
    return {
        "kind": "yaml",
        "data_sha256": _digest_json(normalized),
    }


def _binary_record(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise FingerprintError(f"Cannot read binary artifact {path}: {exc}") from exc
    return {"kind": "binary", "size": len(data), "data_sha256": _digest_bytes(data)}


def _artifact_record(path: Path, rule: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(rule.get("kind", "auto"))
    if kind not in SUPPORTED_KINDS:
        raise FingerprintError(f"Unsupported fingerprint kind {kind!r} for {path}")
    kind = _auto_kind(path) if kind == "auto" else kind
    if kind == "ascii_grid":
        return _ascii_grid_record(path)
    if kind == "csv":
        return _csv_record(path, ignore_columns=tuple(rule.get("ignore_columns", ())))
    if kind == "netcdf":
        return _netcdf_record(path)
    if kind == "image":
        return _image_record(path)
    if kind == "geotiff":
        return _geotiff_record(path)
    if kind == "json":
        return _json_record(path, ignore_keys=tuple(rule.get("ignore_keys", ())))
    if kind == "yaml":
        return _yaml_record(path)
    return _binary_record(path)


def _expand_spec(root: Path, spec: Mapping[str, Any]) -> list[tuple[Path, Mapping[str, Any]]]:
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise FingerprintError(f"Fingerprint spec schema_version must be {SCHEMA_VERSION}")
    entries = spec.get("files")
    if not isinstance(entries, list) or not entries:
        raise FingerprintError("Fingerprint spec requires a nonempty files list")
    exclusions = tuple(str(pattern) for pattern in spec.get("exclude", ()))
    selected: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for index, rule in enumerate(entries):
        if not isinstance(rule, dict):
            raise FingerprintError(f"files[{index}] must be an object")
        has_path = "path" in rule
        has_glob = "glob" in rule
        if has_path == has_glob:
            raise FingerprintError(f"files[{index}] requires exactly one of path or glob")
        candidates: Iterable[Path]
        if has_path:
            candidates = (root / str(rule["path"]),)
        else:
            candidates = root.glob(str(rule["glob"]))
        matched = False
        for candidate in candidates:
            if not candidate.is_file() and not candidate.is_symlink():
                continue
            relative, resolved = _relative_artifact(root, candidate)
            if any(fnmatch.fnmatch(relative.as_posix(), pattern) for pattern in exclusions):
                continue
            matched = True
            key = relative.as_posix()
            previous = selected.get(key)
            if previous is not None and dict(previous[1]) != rule:
                raise FingerprintError(f"Artifact has conflicting fingerprint rules: {key}")
            selected[key] = (resolved, rule)
        if not matched and not bool(rule.get("optional", False)):
            selector = rule.get("path", rule.get("glob"))
            raise FingerprintError(f"Fingerprint selector matched no files: {selector}")
    return [selected[key] for key in sorted(selected)]


def capture_fingerprint(
    root: Path,
    spec: Mapping[str, Any],
    *,
    provenance: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture a deterministic fingerprint for files selected by a specification."""
    root = root.resolve(strict=True)
    files = [
        {"path": path.relative_to(root).as_posix(), **_artifact_record(path, rule)}
        for path, rule in _expand_spec(root, spec)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "spec_sha256": _digest_json(spec),
        "provenance": dict(sorted((provenance or {}).items())),
        "files": files,
    }


def compare_fingerprints(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> list[str]:
    """Return deterministic human-readable differences between two fingerprints."""
    differences: list[str] = []
    if expected.get("schema_version") != actual.get("schema_version"):
        differences.append("schema_version differs")
    if expected.get("spec_sha256") != actual.get("spec_sha256"):
        differences.append("spec_sha256 differs")
    expected_files = {record["path"]: record for record in expected.get("files", ())}
    actual_files = {record["path"]: record for record in actual.get("files", ())}
    differences.extend(
        f"missing artifact: {path}" for path in sorted(expected_files.keys() - actual_files.keys())
    )
    differences.extend(
        f"unexpected artifact: {path}" for path in sorted(actual_files.keys() - expected_files.keys())
    )
    differences.extend(
        f"artifact differs: {path}"
        for path in sorted(expected_files.keys() & actual_files.keys())
        if expected_files[path] != actual_files[path]
    )
    return differences


def _metadata(values: Sequence[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise FingerprintError(f"Metadata must use KEY=VALUE: {value!r}")
        if key in metadata:
            raise FingerprintError(f"Duplicate metadata key: {key}")
        metadata[key] = item
    return metadata


def _capture_command(args: argparse.Namespace) -> int:
    spec = _read_json(args.spec)
    fingerprint = capture_fingerprint(args.root, spec, provenance=_metadata(args.metadata))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Captured {len(fingerprint['files'])} artifacts in {args.output}")
    return 0


def _compare_command(args: argparse.Namespace) -> int:
    expected = _read_json(args.expected)
    actual = _read_json(args.actual)
    differences = compare_fingerprints(expected, actual)
    if differences:
        print("\n".join(differences))
        return 1
    print(f"Fingerprints match ({len(expected.get('files', ()))} artifacts)")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture", help="Capture a fingerprint manifest")
    capture.add_argument("root", type=Path)
    capture.add_argument("spec", type=Path)
    capture.add_argument("output", type=Path)
    capture.add_argument("--metadata", action="append", default=[], metavar="KEY=VALUE")
    capture.set_defaults(handler=_capture_command)
    compare = subparsers.add_parser("compare", help="Compare two fingerprint manifests")
    compare.add_argument("expected", type=Path)
    compare.add_argument("actual", type=Path)
    compare.set_defaults(handler=_compare_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scientific fingerprint command-line utility."""
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except FingerprintError as exc:
        print(f"science fingerprint error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

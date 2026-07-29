#!/usr/bin/env python3
"""Validate that a completed Rofental run supports the manuscript exactly."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = REPO_ROOT / "tests" / "baselines" / "rofental_es30_manuscript_contract.json"
DEFAULT_ASSET_MANIFEST = (
    REPO_ROOT / "tests" / "baselines" / "rofental_es30_manuscript_assets.json"
)
SCHEMA_VERSION = 1
PROJECT_NAME = "project_2022_2023"
RESULT_TOLERANCE = 1e-9


class ManuscriptReferenceError(ValueError):
    """Raised when a reference-run contract or artifact cannot be validated."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManuscriptReferenceError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ManuscriptReferenceError(
            f"JSON schema_version must be {SCHEMA_VERSION}: {path}"
        )
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ManuscriptReferenceError(f"Cannot read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManuscriptReferenceError(f"YAML root must be a mapping: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ManuscriptReferenceError(f"Cannot read CSV {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_record(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            mode = image.mode
            width, height = image.size
            pixels = image.convert("RGBA").tobytes()
    except (OSError, ValueError) as exc:
        raise ManuscriptReferenceError(f"Cannot decode image {path}: {exc}") from exc
    return {
        "file_sha256": _sha256(path),
        "width": width,
        "height": height,
        "mode": mode,
        "pixels_sha256": hashlib.sha256(pixels).hexdigest(),
    }


def _normal(value: Any) -> Any:
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    return value


def _lookup(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    return _normal(value)


def _difference(actual: float, expected: float, *, tolerance: float) -> bool:
    return not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def _validate_config(root: Path, contract: Mapping[str, Any]) -> list[str]:
    setup_candidates = sorted(path for path in root.glob("*.yml") if path.is_file())
    if len(setup_candidates) != 1:
        return [f"expected one top-level setup YAML, found {len(setup_candidates)}"]
    setup = _read_yaml(setup_candidates[0])
    project = _read_yaml(root / "projects" / PROJECT_NAME / f"{PROJECT_NAME}.yml")
    sources = {"setup": setup, "project": project}
    differences: list[str] = []
    for record in contract.get("config_values", ()):  # type: ignore[assignment]
        source_name = str(record["file"])
        keys = tuple(str(key) for key in record["path"])
        try:
            actual = _lookup(sources[source_name], keys)
        except (KeyError, TypeError):
            differences.append(f"missing config value: {source_name}:{'.'.join(keys)}")
            continue
        expected = record["expected"]
        if actual != expected:
            differences.append(
                f"config differs: {source_name}:{'.'.join(keys)} "
                f"expected {expected!r}, got {actual!r}"
            )
    actual_events = project.get("data_assimilation", {}).get("assimilation_events")
    if actual_events != contract.get("events"):
        differences.append("assimilation event sequence differs from manuscript contract")
    return differences


def _read_ascii_grid(path: Path) -> tuple[dict[str, float], np.ndarray[Any, Any]]:
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
        raise ManuscriptReferenceError(f"Cannot read ESRI ASCII grid {path}: {exc}") from exc
    expected_shape = (int(header["nrows"]), int(header["ncols"]))
    if values.size == int(np.prod(expected_shape)):
        values = values.reshape(expected_shape)
    if values.shape != expected_shape:
        raise ManuscriptReferenceError(
            f"Grid shape differs from header for {path}: {values.shape} != {expected_shape}"
        )
    return header, values


def _validate_domain(root: Path, contract: Mapping[str, Any]) -> list[str]:
    expected = contract["domain"]
    dem_header, dem = _read_ascii_grid(root / "grids" / "dem_rofental_100.asc")
    roi_header, roi = _read_ascii_grid(root / "grids" / "roi_rofental_100.asc")
    if dem.shape != roi.shape:
        return [f"DEM and ROI shapes differ: {dem.shape} != {roi.shape}"]
    dem_nodata = dem_header.get("nodata_value")
    roi_nodata = roi_header.get("nodata_value")
    mask = np.isfinite(dem) & np.isfinite(roi) & (roi > 0)
    if dem_nodata is not None:
        mask &= dem != dem_nodata
    if roi_nodata is not None:
        mask &= roi != roi_nodata
    values = dem[mask]
    cell_size = float(dem_header["cellsize"])
    actual = {
        "roi_cells": int(values.size),
        "cell_size_m": cell_size,
        "area_km2": float(values.size * cell_size * cell_size / 1_000_000.0),
        "elevation_min_m": float(values.min()),
        "elevation_max_m": float(values.max()),
        "elevation_mean_m": float(values.mean()),
    }
    tolerance = float(expected["absolute_tolerance"])
    differences: list[str] = []
    for key, expected_value in expected.items():
        if key == "absolute_tolerance":
            continue
        actual_value = actual[key]
        if isinstance(expected_value, int):
            differs = actual_value != expected_value
        else:
            differs = _difference(float(actual_value), float(expected_value), tolerance=tolerance)
        if differs:
            differences.append(
                f"domain statistic differs for {key}: expected {expected_value}, got {actual_value}"
            )
    return differences


def _matching_row(
    rows: Sequence[Mapping[str, str]], keys: Mapping[str, str], *, source: Path
) -> Mapping[str, str]:
    matches = [row for row in rows if all(str(row.get(key)) == value for key, value in keys.items())]
    if len(matches) != 1:
        raise ManuscriptReferenceError(
            f"Expected one row matching {dict(keys)} in {source}, found {len(matches)}"
        )
    return matches[0]


def _validate_stations(root: Path, contract: Mapping[str, Any]) -> list[str]:
    station_path = root / "meteo" / "stations.csv"
    uncertainty_path = root / "obs" / "stations" / "stations_da_metadata.csv"
    stations = _read_csv(station_path)
    uncertainty = _read_csv(uncertainty_path)
    differences: list[str] = []
    for expected in contract.get("forcing_stations", ()):  # type: ignore[assignment]
        row = _matching_row(stations, {"id": str(expected["id"])}, source=station_path)
        for key in ("name", "alt"):
            actual: Any = row[key]
            if key == "alt":
                actual = float(actual)
            if actual != expected[key]:
                differences.append(
                    f"forcing station differs for {expected['id']}:{key}: "
                    f"expected {expected[key]!r}, got {actual!r}"
                )
    for expected in contract.get("station_uncertainty", ()):  # type: ignore[assignment]
        row = _matching_row(
            uncertainty,
            {"station_id": str(expected["station_id"])},
            source=uncertainty_path,
        )
        for key, expected_value in expected.items():
            if key == "station_id":
                continue
            actual = float(row[key])
            if _difference(actual, float(expected_value), tolerance=RESULT_TOLERANCE):
                differences.append(
                    f"station uncertainty differs for {expected['station_id']}:{key}: "
                    f"expected {expected_value}, got {actual}"
                )
    return differences


def _validate_results(
    root: Path,
    contract: Mapping[str, Any],
    *,
    claims_key: str = "benchmark_claims",
) -> list[str]:
    differences: list[str] = []
    project = root / "projects" / PROJECT_NAME
    ensemble_size = 30
    threshold = 0.7 * ensemble_size
    for expected in contract.get("ess", ()):  # type: ignore[assignment]
        weights_path = root / str(expected["path"])
        weights = [float(row["weight"]) for row in _read_csv(weights_path)]
        ess = 1.0 / sum(weight * weight for weight in weights)
        if _difference(ess, float(expected["expected"]), tolerance=RESULT_TOLERANCE):
            differences.append(
                f"ESS differs for {weights_path.name}: expected {expected['expected']}, got {ess}"
            )
        resampled = ess < threshold
        if resampled is not bool(expected["resampled"]):
            differences.append(
                f"resampling decision differs for {weights_path.name}: "
                f"expected {expected['resampled']}, got {resampled}"
            )

    summary_path = project / "results" / "benchmark" / "tables" / "update_summary.csv"
    summary = _read_csv(summary_path)
    for expected in contract.get(claims_key, ()):  # type: ignore[assignment]
        row = _matching_row(
            summary,
            {
                "assimilation_date": str(expected["date"]),
                "variable": str(expected["variable"]),
                "stream": "assimilation_fit",
            },
            source=summary_path,
        )
        for key, expected_value in expected.items():
            if key in {"date", "variable"}:
                continue
            actual = float(row[key])
            if _difference(actual, float(expected_value), tolerance=RESULT_TOLERANCE):
                differences.append(
                    f"benchmark differs for {expected['date']}:{expected['variable']}:{key}: "
                    f"expected {expected_value}, got {actual}"
                )
    return differences


def _validate_figure(
    path: Path,
    expected: Mapping[str, Any],
    label: str,
    *,
    accepted_records: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    if not path.is_file():
        return [f"missing figure: {label}: {path}"]
    actual = _image_record(path)
    keys = ("file_sha256", "width", "height", "mode", "pixels_sha256")
    if any(all(actual[key] == record[key] for key in keys) for record in accepted_records):
        return []
    return [
        f"figure differs for {label}:{key}: expected {expected[key]!r}, got {actual[key]!r}"
        for key in keys
        if actual[key] != expected[key]
    ]


def _validate_figures(
    root: Path,
    asset_manifest: Mapping[str, Any],
    manuscript_root: Path | None,
) -> list[str]:
    differences: list[str] = []
    for expected in asset_manifest.get("figures", ()):  # type: ignore[assignment]
        name = str(expected["name"])
        source = str(expected["source"])
        if source != "manual":
            differences.extend(
                _validate_figure(
                    root / source,
                    expected,
                    f"run:{name}",
                    accepted_records=expected.get("accepted_run_records", ()),
                )
            )
        if manuscript_root is not None:
            differences.extend(
                _validate_figure(manuscript_root / "assets" / name, expected, f"manuscript:{name}")
            )
    return differences


def _validate_tex(tex: str, contract: Mapping[str, Any]) -> list[str]:
    differences = [
        f"manuscript is missing required text: {literal}"
        for literal in contract.get("manuscript_required_literals", ())
        if str(literal) not in tex
    ]
    differences.extend(
        f"manuscript contains text incompatible with selected run: {literal}"
        for literal in contract.get("manuscript_forbidden_literals", ())
        if str(literal) in tex
    )
    return differences


def validate_reference(
    root: Path,
    contract: Mapping[str, Any],
    asset_manifest: Mapping[str, Any],
    *,
    manuscript_root: Path | None = None,
    stage: str = "publication",
) -> list[str]:
    """Return all differences between a completed run and the manuscript contract."""
    root = root.resolve(strict=True)
    if manuscript_root is not None:
        manuscript_root = manuscript_root.resolve(strict=True)
    if stage not in {"simulation", "publication"}:
        raise ManuscriptReferenceError(f"Unsupported validation stage: {stage}")
    differences: list[str] = []
    differences.extend(_validate_config(root, contract))
    differences.extend(_validate_domain(root, contract))
    differences.extend(_validate_stations(root, contract))
    differences.extend(
        _validate_results(
            root,
            contract,
            claims_key="benchmark_claims",
        )
    )
    if stage == "publication":
        differences.extend(_validate_figures(root, asset_manifest, manuscript_root))
    if stage == "publication" and manuscript_root is not None:
        tex_path = manuscript_root / "template.tex"
        try:
            tex = tex_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ManuscriptReferenceError(f"Cannot read manuscript {tex_path}: {exc}") from exc
        differences.extend(_validate_tex(tex, contract))
    return differences


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Completed Rofental setup root")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--asset-manifest", type=Path, default=DEFAULT_ASSET_MANIFEST)
    parser.add_argument(
        "--stage",
        choices=("simulation", "publication"),
        default="publication",
        help="Validate the completed selected simulation or the later publication-analysis state",
    )
    parser.add_argument(
        "--manuscript-root",
        type=Path,
        help="Optional openAMUNDSEN-DA manuscript repo containing template.tex and assets/",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        contract = _read_json(args.contract.resolve(strict=True))
        asset_manifest = _read_json(args.asset_manifest.resolve(strict=True))
        differences = validate_reference(
            args.root,
            contract,
            asset_manifest,
            manuscript_root=args.manuscript_root,
            stage=args.stage,
        )
    except (OSError, ManuscriptReferenceError) as exc:
        print(f"manuscript reference error: {exc}")
        return 2
    if differences:
        print("\n".join(differences))
        return 1
    print("Manuscript reference contract matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

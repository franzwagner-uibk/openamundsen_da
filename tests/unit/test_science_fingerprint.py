from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from PIL import Image, PngImagePlugin
from rasterio.transform import from_origin


def _load_fingerprint_module() -> ModuleType:
    script = Path(__file__).parents[2] / "scripts" / "release" / "science_fingerprint.py"
    spec = importlib.util.spec_from_file_location("science_fingerprint", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fixture(root: Path) -> dict[str, object]:
    (root / "values.csv").write_text("date,value,generated_at\n2026-01-01,1.5,first\n", encoding="utf-8")
    xr.Dataset(
        data_vars={"snow_depth": (("time", "y", "x"), np.array([[[1.0, np.nan]]]))},
        coords={"time": pd.date_range("2026-01-01", periods=1), "y": [1.0], "x": [2.0, 3.0]},
    ).to_netcdf(root / "grids.nc")
    Image.new("RGB", (2, 1), (12, 34, 56)).save(root / "plot.png")
    with rasterio.open(
        root / "model_grid.tif",
        "w",
        driver="GTiff",
        width=2,
        height=1,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(500000.0, 5200000.0, 100.0, 100.0),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(np.array([[[1.0, 2.0]]], dtype=np.float32))
    (root / "manifest.json").write_text(
        json.dumps({"generated_at": "first", "event": {"value": 1.0}}),
        encoding="utf-8",
    )
    (root / "config.yml").write_text("value: 1\nitems: [a, b]\n", encoding="utf-8")
    return {
        "schema_version": 1,
        "files": [
            {"path": "values.csv", "kind": "csv", "ignore_columns": ["generated_at"]},
            {"path": "grids.nc", "kind": "netcdf"},
            {"path": "plot.png", "kind": "image"},
            {"path": "model_grid.tif", "kind": "geotiff"},
            {"path": "manifest.json", "kind": "json", "ignore_keys": ["generated_at"]},
            {"path": "config.yml", "kind": "yaml"},
        ],
    }


def test_capture_fingerprint_compares_scientific_content(tmp_path: Path) -> None:
    module = _load_fingerprint_module()
    specification = _write_fixture(tmp_path)

    expected = module.capture_fingerprint(tmp_path, specification, provenance={"commit": "abc"})
    actual = module.capture_fingerprint(tmp_path, specification, provenance={"commit": "different"})

    assert module.compare_fingerprints(expected, actual) == []
    assert {record["kind"] for record in expected["files"]} == {
        "csv",
        "geotiff",
        "image",
        "json",
        "netcdf",
        "yaml",
    }

    (tmp_path / "values.csv").write_text(
        "date,value,generated_at\n2026-01-01,2.5,second\n",
        encoding="utf-8",
    )
    changed = module.capture_fingerprint(tmp_path, specification)
    assert module.compare_fingerprints(expected, changed) == ["artifact differs: values.csv"]


def test_ascii_grid_fingerprint_ignores_decimal_serialization(tmp_path: Path) -> None:
    module = _load_fingerprint_module()
    path = tmp_path / "grid.asc"
    path.write_text(
        "ncols 2\n"
        "nrows 1\n"
        "xllcorner 0\n"
        "yllcorner 0\n"
        "cellsize 100\n"
        "NODATA_value -9999\n"
        "1.0 2.5\n",
        encoding="utf-8",
    )
    expected = module._ascii_grid_record(path)
    path.write_text(
        "ncols 2\n"
        "nrows 1\n"
        "xllcorner 0,0\n"
        "yllcorner 0,0\n"
        "cellsize 100,0\n"
        "NODATA_value -9999,0\n"
        "1,000000 2,500000\n",
        encoding="utf-8",
    )

    assert module._ascii_grid_record(path) == expected


def test_image_fingerprint_ignores_png_metadata(tmp_path: Path) -> None:
    module = _load_fingerprint_module()
    pixels = Image.new("RGB", (2, 2), (1, 2, 3))
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    pixels.save(first)
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("generated_at", "different")
    pixels.save(second, pnginfo=metadata)

    assert first.read_bytes() != second.read_bytes()
    assert module._image_record(first) == module._image_record(second)


def test_yaml_fingerprint_uses_semantic_content(tmp_path: Path) -> None:
    module = _load_fingerprint_module()
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    first.write_text("date: 2026-01-01\nitems: [a, b]\n", encoding="utf-8")
    second.write_text("items:\n  - a\n  - b\ndate: 2026-01-01\n", encoding="utf-8")

    assert first.read_bytes() != second.read_bytes()
    assert module._yaml_record(first) == module._yaml_record(second)


def test_capture_rejects_path_escape(tmp_path: Path) -> None:
    module = _load_fingerprint_module()
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    specification = {"schema_version": 1, "files": [{"path": "../outside.txt"}]}

    try:
        module.capture_fingerprint(tmp_path, specification)
    except module.FingerprintError as exc:
        assert "escapes" in str(exc)
    else:
        raise AssertionError("path escape was accepted")

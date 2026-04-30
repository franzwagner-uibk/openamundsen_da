"""Copernicus HRWSI downloader for configured project assimilation events."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.loguru_utils import configure_cli_logger


@dataclass(frozen=True)
class ProductConfig:
    product: str
    filename_suffix: str
    output_dir: Path


@dataclass(frozen=True)
class HrwsiConfig:
    endpoint_url: str
    bucket: str
    access_key: str
    secret_key: str
    tiles: tuple[str, ...]
    snowcover: ProductConfig
    wetsnow: ProductConfig


def _require_key(cfg: dict, key: str, *, path: str) -> object:
    if key not in cfg:
        raise ValueError(f"Missing required config key: {path}.{key}")
    val = cfg[key]
    if val is None:
        raise ValueError(f"Missing required config value: {path}.{key}")
    return val


def _require_dict(cfg: dict, key: str, *, path: str) -> dict:
    val = _require_key(cfg, key, path=path)
    if not isinstance(val, dict):
        raise ValueError(f"Expected mapping for {path}.{key}")
    return val


def _resolve_secret_value(raw: object, *, path: str) -> str:
    text = str(raw).strip()
    if not text:
        raise ValueError(f"{path} must not be empty")

    # Support `${ENV_VAR}` and `env:ENV_VAR` forms.
    env_name: str | None = None
    if text.startswith("${") and text.endswith("}") and len(text) > 3:
        env_name = text[2:-1].strip()
    elif text.lower().startswith("env:"):
        env_name = text[4:].strip()

    if env_name:
        env_val = os.environ.get(env_name)
        if not env_val:
            raise ValueError(f"{path} references environment variable '{env_name}', but it is not set")
        return str(env_val).strip()

    return text


def _resolve_dir(setup_dir: Path, value: str) -> Path:
    p = Path(str(value).strip())
    return p if p.is_absolute() else (setup_dir / p)


def _parse_product_cfg(section: dict, *, path: str, setup_dir: Path) -> ProductConfig:
    product = str(_require_key(section, "product", path=path)).strip().upper()
    suffix = str(_require_key(section, "filename_suffix", path=path)).strip()
    out_dir_txt = str(_require_key(section, "output_dir", path=path)).strip()
    if not product:
        raise ValueError(f"Empty product at {path}.product")
    if not suffix:
        raise ValueError(f"Empty filename_suffix at {path}.filename_suffix")
    if not out_dir_txt:
        raise ValueError(f"Empty output_dir at {path}.output_dir")
    return ProductConfig(
        product=product,
        filename_suffix=suffix,
        output_dir=_resolve_dir(setup_dir, out_dir_txt),
    )


def load_hrwsi_config(setup_dir: Path, project_dir: Path) -> HrwsiConfig:
    project_yaml = find_project_yaml(project_dir)
    project_cfg = _read_yaml_file(project_yaml) or {}
    base_path = "copernicus_download"
    raw = _require_dict(project_cfg, "copernicus_download", path="project")

    endpoint_url = str(_require_key(raw, "endpoint_url", path=base_path)).strip()
    bucket = str(_require_key(raw, "bucket", path=base_path)).strip()
    access_key = _resolve_secret_value(_require_key(raw, "access_key", path=base_path), path=f"{base_path}.access_key")
    secret_key = _resolve_secret_value(_require_key(raw, "secret_key", path=base_path), path=f"{base_path}.secret_key")
    tiles_raw = _require_key(raw, "tiles", path=base_path)
    if not isinstance(tiles_raw, list) or not tiles_raw:
        raise ValueError("copernicus_download.tiles must be a non-empty list")
    tiles = tuple(str(t).strip().upper() for t in tiles_raw if str(t).strip())
    if not tiles:
        raise ValueError("copernicus_download.tiles must contain at least one non-empty tile id")

    snowcover_cfg = _parse_product_cfg(
        _require_dict(raw, "snowcover", path=base_path),
        path=f"{base_path}.snowcover",
        setup_dir=setup_dir,
    )
    wetsnow_cfg = _parse_product_cfg(
        _require_dict(raw, "wetsnow", path=base_path),
        path=f"{base_path}.wetsnow",
        setup_dir=setup_dir,
    )

    for field_name, value in (
        ("endpoint_url", endpoint_url),
        ("bucket", bucket),
        ("access_key", access_key),
        ("secret_key", secret_key),
    ):
        if not value:
            raise ValueError(f"copernicus_download.{field_name} must not be empty")

    return HrwsiConfig(
        endpoint_url=endpoint_url,
        bucket=bucket,
        access_key=access_key,
        secret_key=secret_key,
        tiles=tiles,
        snowcover=snowcover_cfg,
        wetsnow=wetsnow_cfg,
    )


def _list_keys(client, *, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            key = str(obj.get("Key", ""))
            if key:
                keys.append(key)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "date",
                "variable",
                "product",
                "tile",
                "source_key",
                "target_path",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def download_project_event_rasters(
    *,
    setup_dir: Path,
    project_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> Path:
    import boto3

    cfg = load_hrwsi_config(setup_dir, project_dir)
    events = load_assimilation_events(project_dir)
    if not events:
        raise ValueError(f"No assimilation events found in {project_dir}")

    client = boto3.client(
        "s3",
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        endpoint_url=cfg.endpoint_url,
    )

    manifest_rows: list[dict[str, str]] = []
    for ev in events:
        if ev.variable == "scf":
            pcfg = cfg.snowcover
        elif ev.variable in {"wet_snow", "wet_snow_line"}:
            pcfg = cfg.wetsnow
        else:
            logger.info("Skipping unsupported assimilation variable '{}'", ev.variable)
            continue

        if str(ev.product).upper() != pcfg.product:
            raise ValueError(
                f"Event product mismatch on {ev.date}: event has '{ev.product}', "
                f"but copernicus_download config for {ev.variable} requires '{pcfg.product}'."
            )

        pcfg.output_dir.mkdir(parents=True, exist_ok=True)
        y = ev.date.year
        m = ev.date.month
        d = ev.date.day
        logger.info(
            "Downloading {} {} for date {} (tiles={})",
            ev.variable,
            pcfg.product,
            ev.date.isoformat(),
            ",".join(cfg.tiles),
        )

        for tile in cfg.tiles:
            prefix = f"{pcfg.product}/{tile}/{y:04d}/{m:02d}/{d:02d}/"
            keys = _list_keys(client, bucket=cfg.bucket, prefix=prefix)
            selected = sorted(k for k in keys if k.endswith(pcfg.filename_suffix))
            if not selected:
                raise FileNotFoundError(
                    f"No files matching suffix '{pcfg.filename_suffix}' for prefix '{prefix}'"
                )

            for key in selected:
                target = pcfg.output_dir / Path(key).name
                status = "dry_run"
                if target.exists() and not overwrite:
                    status = "skipped_exists"
                else:
                    if not dry_run:
                        client.download_file(cfg.bucket, key, str(target))
                    status = "downloaded" if not dry_run else "dry_run"
                manifest_rows.append(
                    {
                        "date": ev.date.isoformat(),
                        "variable": ev.variable,
                        "product": pcfg.product,
                        "tile": tile,
                        "source_key": key,
                        "target_path": str(target),
                        "status": status,
                    }
                )
                logger.info("[{}] {}", status, target.name)

    manifest = project_dir / "hrwsi_download_manifest.csv"
    _write_manifest(manifest, manifest_rows)
    logger.info("Download manifest written: {}", manifest)
    return manifest


def cli_main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="oa-da-hrwsi-download",
        description="Download Copernicus HRWSI rasters for configured project assimilation events.",
    )
    p.add_argument("--setup-dir", required=True, type=Path, help="Setup directory")
    p.add_argument("--project-dir", required=True, type=Path, help="Project directory under setup/projects")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    p.add_argument("--dry-run", action="store_true", help="Do not download files, only list planned actions")
    p.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = p.parse_args(argv)

    configure_cli_logger(args.log_level)

    try:
        download_project_event_rasters(
            setup_dir=Path(args.setup_dir),
            project_dir=Path(args.project_dir),
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )
        return 0
    except Exception as exc:
        logger.error("HRWSI download failed: {}", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

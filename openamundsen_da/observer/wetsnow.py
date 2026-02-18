"""Generic wet-snow summarization CLI with configurable class mapping."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence

from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.methods.wet_snow.area import summarize_s1_directory
from openamundsen_da.io.paths import find_setup_yaml, find_project_yaml
from openamundsen_da.util.ts import parse_datetime_opt
from openamundsen_da.util.landcover_mask import resolve_landcover_mask
from openamundsen_da.util.roi_grid import ensure_setup_roi_vector


def _parse_ints(values: Sequence[object] | None, *, default: Sequence[int]) -> list[int]:
    out: list[int] = []
    vals = values if values is not None else default
    for v in vals:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _load_wetsnow_classes(setup_dir: Path) -> tuple[list[int], list[int], list[int]]:
    cfg = _read_yaml_file(find_setup_yaml(setup_dir)) or {}
    obs_cfg = (cfg.get("obs") or {}).get("wetsnow") or {}
    classes = obs_cfg.get("classes") or {}
    wet = _parse_ints(classes.get("wet"), default=[1, 2])
    valid = _parse_ints(classes.get("valid"), default=[1, 2, 3, 4, 255])
    exclude = _parse_ints(classes.get("exclude"), default=[5, 6])
    return wet, valid, exclude


def cli_main(argv: List[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="oa-da-wetsnow",
        description="Summarize wet-snow rasters (GeoTIFF/NetCDF) into wet_snow_summary.csv with configurable classes.",
    )
    p.add_argument("--input-dir", required=True, type=Path, help="Directory of wet-snow classification rasters (.tif/.nc)")
    p.add_argument("--roi", dest="roi", type=Path, help="ROI vector; defaults to <project>/env/roi.gpkg")
    p.add_argument("--project-label", required=True, help="Project folder name under obs/")
    p.add_argument("--setup-dir", type=Path, help="Setup directory (default: CWD)")
    p.add_argument("--output-root", type=Path, help="Override output root (default: <project>/obs/summaries)")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories for rasters")
    p.add_argument("--start-date", type=str, help="Optional ISO start date filter")
    p.add_argument("--end-date", type=str, help="Optional ISO end date filter")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing wet_snow_summary.csv")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    setup_dir = Path(args.setup_dir) if args.setup_dir else Path.cwd()
    default_root = setup_dir / "obs" / "summaries"
    output_root = Path(args.output_root) if args.output_root else default_root
    project_cfg_dir = setup_dir / "projects" / str(args.project_label)
    lc_cfg = resolve_landcover_mask(setup_dir, project_cfg_dir)

    start = parse_datetime_opt(args.start_date) if args.start_date else None
    end = parse_datetime_opt(args.end_date) if args.end_date else None
    if start is None or end is None:
        try:
            seas_yaml = find_project_yaml(setup_dir / "projects" / args.project_label)
            seas_cfg = _read_yaml_file(seas_yaml) or {}
            if start is None and seas_cfg.get("start_date"):
                start = parse_datetime_opt(str(seas_cfg["start_date"]))
            if end is None and seas_cfg.get("end_date"):
                end = parse_datetime_opt(str(seas_cfg["end_date"]))
        except Exception:
            pass

    wet, valid, exclude = _load_wetsnow_classes(setup_dir)

    try:
        roi = Path(args.roi) if args.roi else ensure_setup_roi_vector(setup_dir)
        out_csv = output_root / args.project_label / "wet_snow_summary.csv"
        summarize_s1_directory(
            setup_dir=setup_dir,
            project_dir=project_cfg_dir,
            raster_dir=Path(args.input_dir),
            aoi_path=roi,
            output_csv=out_csv,
            landcover_cfg=lc_cfg,
            overwrite=bool(args.overwrite),
            start=start,
            end=end,
            wet_values=wet,
            valid_values=valid,
            exclude_values=exclude,
            recursive=bool(args.recursive),
        )
        return 0
    except Exception as exc:
        logger.error("Wet-snow summarization failed: {}", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())



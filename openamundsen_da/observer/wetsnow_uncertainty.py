"""Generate wet-snow uncertainty companion rasters from project configuration.

This utility is intended for development/tutorial workflows where an explicit
uncertainty layer is needed per wet-snow observation raster.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio
from loguru import logger
from rasterio.warp import Resampling, reproject

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml
from openamundsen_da.util.config_validators import require_mapping
from openamundsen_da.util.landcover_mask import resolve_landcover_mask
from openamundsen_da.util.loguru_utils import configure_cli_logger


@dataclass(frozen=True)
class WetSnowClasses:
    wet: tuple[int, ...]
    valid: tuple[int, ...]
    exclude: tuple[int, ...]


@dataclass(frozen=True)
class WetSnowClassMapping:
    base_classes: tuple[int, ...]
    max_uncertainty_classes: tuple[int, ...]
    nodata_classes: tuple[int, ...]


@dataclass(frozen=True)
class PenaltyRule:
    name: str
    source: str  # wet_snow | landcover | shadow
    classes: tuple[int, ...]
    penalty: float
    enabled: bool
    input_dir: Path | None  # required for source=shadow


@dataclass(frozen=True)
class WetSnowUncertaintyConfig:
    enabled: bool
    input_dir: Path
    base_uncertainty: float
    nodata_value: float
    class_mapping: WetSnowClassMapping
    penalties: tuple[PenaltyRule, ...]


def _require_int_tuple(values: object, *, path: str, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{path} must be a list of integers")
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except Exception as exc:
            raise ValueError(f"{path} contains non-integer value: {value!r}") from exc
    uniq = tuple(sorted(set(out)))
    if not uniq and not allow_empty:
        raise ValueError(f"{path} must contain at least one integer")
    return uniq


def _resolve_path(raw: str | Path, *, base_dir: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (base_dir / p)


def _slugify_name(raw: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", raw.strip().lower()).strip("_")
    return slug or "rule"


def _normalize_rule_name(raw_name: str | None, index: int, used: set[str]) -> str:
    base = _slugify_name(raw_name) if raw_name else f"rule_{index + 1}"
    name = base
    suffix = 2
    while name in used:
        name = f"{base}_{suffix}"
        suffix += 1
    used.add(name)
    return name


def _extract_date_keys(name_or_stem: str) -> tuple[str, ...]:
    keys: set[str] = set()
    for match in re.finditer(r"(20\d{2})[_-]?(\d{2})[_-]?(\d{2})", name_or_stem):
        y, m, d = match.groups()
        keys.add(f"{y}_{m}_{d}")
        keys.add(f"{y}{m}{d}")
    return tuple(sorted(keys))


def _resolve_shadow_path(
    *,
    src_path: Path,
    shadow_by_name: dict[str, Path],
    shadow_by_date: dict[str, list[Path]],
) -> Path | None:
    stem = src_path.stem.lower()
    direct = shadow_by_name.get(stem)
    if direct is not None:
        return direct

    for repl in ("wsm", "wet", "wetsnow"):
        repl_stem = stem.replace(repl, "shadow")
        cand = shadow_by_name.get(repl_stem)
        if cand is not None:
            return cand

    for key in _extract_date_keys(src_path.stem):
        matches = shadow_by_date.get(key, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return sorted(matches)[0]
    return None


def _resolve_wetsnow_class_mapping(
    *,
    wet_unc: dict[str, object],
    obs_classes: WetSnowClasses,
) -> WetSnowClassMapping:
    path = "project.data_assimilation.uncertainty.wet_snow.class_mapping"
    class_mapping_raw = wet_unc.get("class_mapping")
    class_mapping = require_mapping(class_mapping_raw, path=path) if class_mapping_raw else {}

    for old_key in ("base_groups", "max_uncertainty_groups", "nodata_groups"):
        if old_key in class_mapping:
            raise ValueError(
                f"{path}.{old_key} is no longer supported; use '*_classes' with raw class IDs."
            )

    default_valid = set(obs_classes.valid)
    default_exclude = set(obs_classes.exclude)
    default_base = tuple(sorted(default_valid - default_exclude))
    default_max = tuple()
    default_nodata = tuple(sorted(default_exclude))

    if "base_classes" in class_mapping:
        base = _require_int_tuple(class_mapping.get("base_classes"), path=f"{path}.base_classes")
    else:
        if not default_base:
            raise ValueError(
                f"{path}: no base class mapping configured and default valid\\exclude mapping is empty"
            )
        base = default_base

    if "max_uncertainty_classes" in class_mapping:
        max_unc = _require_int_tuple(
            class_mapping.get("max_uncertainty_classes"),
            path=f"{path}.max_uncertainty_classes",
            allow_empty=True,
        )
    else:
        max_unc = default_max

    if "nodata_classes" in class_mapping:
        nodata = _require_int_tuple(
            class_mapping.get("nodata_classes"),
            path=f"{path}.nodata_classes",
            allow_empty=True,
        )
    else:
        nodata = default_nodata

    overlap_base_nodata = sorted(set(base).intersection(nodata))
    if overlap_base_nodata:
        raise ValueError(
            f"{path}: base and nodata classes overlap: {overlap_base_nodata}"
        )
    overlap_base_max = sorted(set(base).intersection(max_unc))
    if overlap_base_max:
        raise ValueError(
            f"{path}: base and max_uncertainty classes overlap: {overlap_base_max}"
        )
    overlap_max_nodata = sorted(set(max_unc).intersection(nodata))
    if overlap_max_nodata:
        raise ValueError(
            f"{path}: max_uncertainty and nodata classes overlap: {overlap_max_nodata}"
        )
    if not base and not max_unc:
        raise ValueError(f"{path}: at least one class must be mapped to base or max_uncertainty")

    return WetSnowClassMapping(
        base_classes=base,
        max_uncertainty_classes=max_unc,
        nodata_classes=nodata,
    )


def _load_project_config(project_dir: Path) -> tuple[WetSnowUncertaintyConfig, WetSnowClasses]:
    cfg = require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    obs_cfg = require_mapping(cfg.get("obs"), path="project.obs")
    wet_cfg = require_mapping(obs_cfg.get("wetsnow"), path="project.obs.wetsnow")
    obs_classes_raw = require_mapping(wet_cfg.get("classes"), path="project.obs.wetsnow.classes")
    wet_classes = WetSnowClasses(
        wet=_require_int_tuple(obs_classes_raw.get("wet"), path="project.obs.wetsnow.classes.wet"),
        valid=_require_int_tuple(obs_classes_raw.get("valid"), path="project.obs.wetsnow.classes.valid"),
        exclude=_require_int_tuple(obs_classes_raw.get("exclude"), path="project.obs.wetsnow.classes.exclude"),
    )

    da_cfg = require_mapping(cfg.get("data_assimilation"), path="project.data_assimilation")
    unc_cfg_raw = da_cfg.get("uncertainty")
    unc_cfg = require_mapping(unc_cfg_raw, path="project.data_assimilation.uncertainty") if unc_cfg_raw else {}
    wet_unc_raw = unc_cfg.get("wet_snow")
    wet_unc = require_mapping(wet_unc_raw, path="project.data_assimilation.uncertainty.wet_snow") if wet_unc_raw else {}
    wet_class_mapping = _resolve_wetsnow_class_mapping(wet_unc=wet_unc, obs_classes=wet_classes)

    setup_dir = project_dir.parent.parent
    default_input = _resolve_path(str(wet_cfg.get("dir", "obs/wetsnow")), base_dir=setup_dir)

    penalties: list[PenaltyRule] = []
    used_names: set[str] = set()
    if "penalties" not in wet_unc:
        raise ValueError(
            "Missing required configuration key: "
            "project.data_assimilation.uncertainty.wet_snow.penalties"
        )
    penalties_raw = wet_unc.get("penalties")
    if not isinstance(penalties_raw, Sequence) or isinstance(penalties_raw, (str, bytes)):
        raise ValueError("project.data_assimilation.uncertainty.wet_snow.penalties must be a list")
    for i, raw in enumerate(penalties_raw):
        rule_path = f"project.data_assimilation.uncertainty.wet_snow.penalties[{i}]"
        rule = require_mapping(raw, path=rule_path)
        source = str(rule.get("source", "")).strip().lower()
        if source not in {"wet_snow", "landcover", "shadow"}:
            raise ValueError(
                f"{rule_path}.source must be one of: "
                "wet_snow, landcover, shadow"
            )
        name = _normalize_rule_name(rule.get("name"), i, used_names)
        if "groups" in rule:
            raise ValueError(f"{rule_path}.groups is no longer supported; use .classes with raw class IDs.")
        classes = _require_int_tuple(rule.get("classes"), path=f"{rule_path}.classes")
        input_dir: Path | None = None
        if source == "shadow":
            default_shadow_input = setup_dir / "obs" / "shadow"
            input_dir = _resolve_path(
                str(rule.get("input_dir", default_shadow_input)),
                base_dir=setup_dir,
            )
        penalties.append(
            PenaltyRule(
                name=name,
                source=source,
                classes=classes,
                penalty=float(rule.get("penalty", 0.0)),
                enabled=bool(rule.get("enabled", True)),
                input_dir=input_dir,
            )
        )

    config = WetSnowUncertaintyConfig(
        enabled=bool(wet_unc.get("enabled", True)),
        input_dir=_resolve_path(str(wet_unc.get("input_dir", default_input)), base_dir=setup_dir),
        base_uncertainty=float(wet_unc.get("base_uncertainty", 15.0)),
        nodata_value=float(wet_unc.get("nodata_value", 255.0)),
        class_mapping=wet_class_mapping,
        penalties=tuple(penalties),
    )
    return config, wet_classes


def _resample_to_template(
    src_path: Path,
    template: rasterio.DatasetReader,
    *,
    dst_nodata: float,
) -> np.ndarray:
    with rasterio.open(src_path) as src:
        dst = np.full(template.shape, dst_nodata, dtype=np.float32)
        src_crs = src.crs if src.crs is not None else template.crs
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=template.transform,
            dst_crs=template.crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=dst_nodata,
        )
    return dst


def _apply_penalty_rules(
    *,
    unc_active: np.ndarray,
    active: np.ndarray,
    wet_data: np.ndarray,
    landcover_resampled: np.ndarray | None,
    shadow_by_rule: dict[str, np.ndarray],
    rules: Sequence[PenaltyRule],
) -> dict[str, float]:
    fractions: dict[str, float] = {}

    for rule in rules:
        if not rule.enabled:
            fractions[rule.name] = np.nan
            continue
        if rule.penalty == 0.0:
            fractions[rule.name] = 0.0
            continue

        mask_active = np.zeros(unc_active.shape, dtype=bool)

        if rule.source == "wet_snow":
            vals = wet_data[active]
            finite_vals = np.isfinite(vals)
            if np.any(finite_vals):
                mask_active[finite_vals] = np.isin(vals[finite_vals].astype(np.int32), rule.classes)
        elif rule.source == "landcover":
            if landcover_resampled is None:
                fractions[rule.name] = np.nan
                continue
            vals = landcover_resampled[active]
            finite_vals = np.isfinite(vals)
            if np.any(finite_vals):
                mask_active[finite_vals] = np.isin(vals[finite_vals].astype(np.int32), rule.classes)
        elif rule.source == "shadow":
            shadow = shadow_by_rule.get(rule.name)
            if shadow is None:
                fractions[rule.name] = np.nan
                continue
            vals = shadow[active]
            finite_vals = np.isfinite(vals)
            if np.any(finite_vals):
                mask_active[finite_vals] = np.isin(vals[finite_vals].astype(np.int32), rule.classes)
        else:
            fractions[rule.name] = np.nan
            continue

        frac = float(np.mean(mask_active)) if mask_active.size else np.nan
        fractions[rule.name] = frac
        if np.any(mask_active):
            unc_active[mask_active] = unc_active[mask_active] + rule.penalty

    return fractions


def _build_uncertainty(
    wet_data: np.ndarray,
    *,
    landcover_resampled: np.ndarray | None,
    shadow_by_rule: dict[str, np.ndarray],
    cfg: WetSnowUncertaintyConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    out = np.full(wet_data.shape, cfg.nodata_value, dtype=np.float32)
    finite = np.isfinite(wet_data)
    wet_int = np.zeros(wet_data.shape, dtype=np.int32)
    if np.any(finite):
        wet_int[finite] = wet_data[finite].astype(np.int32)

    nodata_mask = finite & np.isin(wet_int, cfg.class_mapping.nodata_classes)
    base_mask = finite & np.isin(wet_int, cfg.class_mapping.base_classes) & ~nodata_mask
    max_mask = finite & np.isin(wet_int, cfg.class_mapping.max_uncertainty_classes) & ~nodata_mask
    active = base_mask | max_mask

    fractions: dict[str, float] = {r.name: np.nan for r in cfg.penalties}
    if np.any(active):
        unc_active = np.zeros(int(np.count_nonzero(active)), dtype=np.float32)
        active_int = wet_int[active]
        base_active = np.isin(active_int, cfg.class_mapping.base_classes)
        max_active = np.isin(active_int, cfg.class_mapping.max_uncertainty_classes)
        if np.any(base_active):
            unc_active[base_active] = cfg.base_uncertainty
        if np.any(max_active):
            unc_active[max_active] = 100.0

        fractions = _apply_penalty_rules(
            unc_active=unc_active,
            active=active,
            wet_data=wet_data,
            landcover_resampled=landcover_resampled,
            shadow_by_rule=shadow_by_rule,
            rules=cfg.penalties,
        )
        out[active] = np.clip(unc_active, 0.0, 100.0)
    out[nodata_mask] = cfg.nodata_value
    return out, fractions


def _date_token_from_name(name: str) -> str:
    stem = Path(name).stem
    m = re.search(r"(20\d{2})[_-](\d{2})[_-](\d{2})", stem)
    if m:
        y, mon, day = m.groups()
        return f"{y}_{mon}_{day}"
    m2 = re.search(r"(20\d{2})(\d{2})(\d{2})", stem)
    if m2:
        y, mon, day = m2.groups()
        return f"{y}_{mon}_{day}"
    parts = stem.split("_")
    if len(parts) >= 4:
        return "_".join(parts[-3:])
    return stem


def generate_uncertainty_layers(*, setup_dir: Path, project_label: str, overwrite: bool = False) -> Path:
    project_dir = setup_dir / "projects" / project_label
    cfg, _ = _load_project_config(project_dir)

    if not cfg.enabled:
        logger.info("Wet-snow uncertainty generation disabled in project YAML; nothing to do.")
        return cfg.input_dir

    files_all = sorted(cfg.input_dir.glob("*.tif")) + sorted(cfg.input_dir.glob("*.tiff"))
    files = [p for p in files_all if not p.stem.lower().endswith("_uncertainty")]
    if not files:
        raise FileNotFoundError(f"No wet-snow GeoTIFF files found in {cfg.input_dir}")

    lc_cfg = resolve_landcover_mask(setup_dir, project_dir)
    if lc_cfg.path is None:
        logger.warning(
            "Land-cover mask path not resolved from project configuration; "
            "landcover penalties will be skipped."
        )
        landcover_resampled = None
    else:
        with rasterio.open(files[0]) as template:
            landcover_resampled = _resample_to_template(lc_cfg.path, template, dst_nodata=np.nan)

    shadow_indexes: dict[Path, tuple[dict[str, Path], dict[str, list[Path]]]] = {}
    for rule in cfg.penalties:
        if rule.enabled and rule.source == "shadow" and rule.input_dir is not None:
            rule_dir = rule.input_dir
            if rule_dir in shadow_indexes:
                continue
            shadow_files = sorted(rule_dir.glob("*.tif")) + sorted(rule_dir.glob("*.tiff"))
            by_name: dict[str, Path] = {}
            by_date: dict[str, list[Path]] = {}
            if not shadow_files:
                logger.warning(
                    "Shadow penalty rule '{}' enabled but no files found in {}",
                    rule.name,
                    rule_dir,
                )
            for sh in shadow_files:
                by_name[sh.stem.lower()] = sh
                for key in _extract_date_keys(sh.stem):
                    by_date.setdefault(key, []).append(sh)
            shadow_indexes[rule_dir] = (by_name, by_date)

    fraction_cols = [f"frac_{rule.name}" for rule in cfg.penalties]
    rows: list[str] = [
        "date,source,output,shadow_sources,n_valid,mean_unc,min_unc,max_unc," + ",".join(fraction_cols)
    ]

    generated = 0
    skipped = 0
    for src_path in files:
        out_name = f"{src_path.stem}_uncertainty.tif"
        out_path = src_path.parent / out_name
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        with rasterio.open(src_path) as src:
            wet_data = src.read(1)
            shadow_by_rule: dict[str, np.ndarray] = {}
            shadow_sources: list[str] = []
            for rule in cfg.penalties:
                if not (rule.enabled and rule.source == "shadow" and rule.input_dir is not None):
                    continue
                by_name, by_date = shadow_indexes.get(rule.input_dir, ({}, {}))
                shadow_path = _resolve_shadow_path(
                    src_path=src_path,
                    shadow_by_name=by_name,
                    shadow_by_date=by_date,
                )
                if shadow_path is None:
                    continue
                shadow_arr = _resample_to_template(shadow_path, src, dst_nodata=np.nan)
                shadow_by_rule[rule.name] = shadow_arr
                shadow_sources.append(shadow_path.name)

            unc, fractions = _build_uncertainty(
                wet_data=wet_data,
                landcover_resampled=landcover_resampled,
                shadow_by_rule=shadow_by_rule,
                cfg=cfg,
            )

            profile = src.profile.copy()
            profile.update(
                dtype="float32",
                nodata=cfg.nodata_value,
                compress="deflate",
                predictor=3,
                tiled=True,
                blockxsize=256,
                blockysize=256,
            )
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(unc.astype(np.float32), 1)
                dst.set_band_description(1, "uncertainty_percent")
                dst.update_tags(
                    1,
                    units="percent",
                    long_name="Wet-snow uncertainty (tutorial synthetic v1)",
                    method="constant_base_plus_penalty_rules",
                    base_uncertainty=str(cfg.base_uncertainty),
                )

        valid = unc != cfg.nodata_value
        valid_count = int(np.count_nonzero(valid))
        if valid_count > 0:
            vals = unc[valid]
            mean_unc = float(np.mean(vals))
            min_unc = float(np.min(vals))
            max_unc = float(np.max(vals))
        else:
            mean_unc = float("nan")
            min_unc = float("nan")
            max_unc = float("nan")

        frac_vals: list[str] = []
        for rule in cfg.penalties:
            v = fractions.get(rule.name, np.nan)
            frac_vals.append("nan" if not np.isfinite(v) else f"{float(v):.4f}")

        shadow_sources_joined = "|".join(sorted(set(shadow_sources)))
        rows.append(
            f"{_date_token_from_name(src_path.name)},{src_path.name},{out_name},{shadow_sources_joined},"
            f"{valid_count},{mean_unc:.4f},{min_unc:.4f},{max_unc:.4f}," + ",".join(frac_vals)
        )
        generated += 1

    (cfg.input_dir / "uncertainty_summary.csv").write_text("\n".join(rows) + "\n", encoding="ascii")
    logger.info(
        "Wet-snow uncertainty generation completed: generated={} skipped={} output={}",
        generated,
        skipped,
        cfg.input_dir,
    )
    return cfg.input_dir


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oa-da-wetsnow-uncertainty",
        description="Generate wet-snow uncertainty companion rasters from project YAML.",
    )
    parser.add_argument("--setup-dir", required=True, type=Path, help="Path to setup root")
    parser.add_argument(
        "--project-label",
        required=True,
        help="Project folder name under <setup-dir>/projects",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing uncertainty rasters",
    )
    args = parser.parse_args(argv)

    configure_cli_logger("INFO")
    setup_dir = args.setup_dir.expanduser().resolve()
    generate_uncertainty_layers(
        setup_dir=setup_dir,
        project_label=str(args.project_label),
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())

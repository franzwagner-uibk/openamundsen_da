#!/usr/bin/env python3
"""Compare assimilation sigma/ESS between two project runs.

Reads per-step weights CSV files from two completed project runs and writes:
- a compact comparison CSV
- a PNG plot with sigma and ESS time series
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class WeightsStats:
    variable: str
    date: str
    filename: str
    sigma: float
    ess: float


_WEIGHTS_RE = re.compile(r"^weights_(scf|wet_snow)_(\d{8})\.csv$")


def _iter_weight_files(project_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(project_dir.glob("steps/step_*/assim/weights_*.csv")):
        m = _WEIGHTS_RE.match(path.name)
        if not m:
            continue
        out[path.name] = path
    return out


def _compute_stats(path: Path) -> WeightsStats:
    m = _WEIGHTS_RE.match(path.name)
    if m is None:
        raise ValueError(f"Unexpected weights filename: {path.name}")
    variable = m.group(1)
    date_raw = m.group(2)
    date_iso = f"{date_raw[0:4]}-{date_raw[4:6]}-{date_raw[6:8]}"

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Weights file has no rows: {path}")

    sigma = float(rows[0]["sigma"])
    w = [float(r["weight"]) for r in rows]
    denom = sum(x * x for x in w)
    if denom <= 0.0:
        raise ValueError(f"Invalid weights (sum w^2 <= 0) in {path}")
    ess = 1.0 / denom
    return WeightsStats(variable=variable, date=date_iso, filename=path.name, sigma=sigma, ess=ess)


def _write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "variable",
        "weights_file",
        "sigma_uncertainty_layer",
        "sigma_formula",
        "delta_sigma",
        "ess_uncertainty_layer",
        "ess_formula",
        "delta_ess",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(rows: list[dict[str, object]], output_png: Path) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['date']}\n{r['variable']}" for r in rows]
    x = list(range(len(rows)))
    sigma_u = [float(r["sigma_uncertainty_layer"]) for r in rows]
    sigma_f = [float(r["sigma_formula"]) for r in rows]
    ess_u = [float(r["ess_uncertainty_layer"]) for r in rows]
    ess_f = [float(r["ess_formula"]) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    axes[0].plot(x, sigma_u, marker="o", label="uncertainty_layer")
    axes[0].plot(x, sigma_f, marker="o", label="formula")
    axes[0].set_ylabel("sigma")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, ess_u, marker="o", label="uncertainty_layer")
    axes[1].plot(x, ess_f, marker="o", label="formula")
    axes[1].set_ylabel("ESS")
    axes[1].set_xlabel("assimilation event")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="Compare sigma/ESS across two project runs.")
    p.add_argument("--project-uncertainty", type=Path, required=True)
    p.add_argument("--project-formula", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-png", type=Path, required=True)
    args = p.parse_args()

    unc_files = _iter_weight_files(args.project_uncertainty)
    form_files = _iter_weight_files(args.project_formula)

    shared = sorted(set(unc_files) & set(form_files))
    if not shared:
        raise FileNotFoundError("No shared weights_*.csv files found between both projects")

    rows: list[dict[str, object]] = []
    order: list[tuple[str, str]] = []
    tmp: dict[tuple[str, str], dict[str, object]] = {}

    for name in shared:
        u = _compute_stats(unc_files[name])
        f = _compute_stats(form_files[name])
        key = (u.date, u.variable)
        tmp[key] = {
            "date": u.date,
            "variable": u.variable,
            "weights_file": u.filename,
            "sigma_uncertainty_layer": u.sigma,
            "sigma_formula": f.sigma,
            "delta_sigma": u.sigma - f.sigma,
            "ess_uncertainty_layer": u.ess,
            "ess_formula": f.ess,
            "delta_ess": u.ess - f.ess,
        }
        order.append(key)

    for key in sorted(set(order)):
        rows.append(tmp[key])

    _write_csv(rows, args.output_csv)
    _write_plot(rows, args.output_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

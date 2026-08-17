from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_cleanup_benchmark_selects_a_worker_and_removes_exact_trial_trees(
    tmp_path: Path,
) -> None:
    repo = Path(__file__).resolve().parents[2]
    scratch = tmp_path / "scratch"
    result = tmp_path / "benchmark.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)

    subprocess.run(
        [
            sys.executable,
            str(repo / "scripts/benchmark_runtime_cleanup.py"),
            "--scratch-root",
            str(scratch),
            "--workers",
            "1,2",
            "--sample-files",
            "100",
            "--sample-bytes",
            "100000",
            "--sample-units",
            "10",
            "--result-json",
            str(result),
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["contract"] == "runtime-cleanup-benchmark-v1"
    assert payload["selected_workers"] in {1, 2}
    assert [row["deleted_files"] for row in payload["sample_results"]] == [100, 100]
    assert not scratch.exists()

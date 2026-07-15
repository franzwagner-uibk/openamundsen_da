from __future__ import annotations

from pathlib import Path

import pytest

from openamundsen_da.util.yaml_utils import read_yaml_mapping


def test_read_yaml_mapping_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yml"
    path.write_text("start_date: 2022-01-01\nstart_date: 2022-02-01\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="duplicate key"):
        read_yaml_mapping(path)

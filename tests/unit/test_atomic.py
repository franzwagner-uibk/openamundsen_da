from pathlib import Path

from openamundsen_da.util import atomic


def test_durable_replace_fsyncs_file_then_replacement_then_parent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "accepted.bin"
    temp = tmp_path / ".accepted.tmp"
    target.write_bytes(b"old")
    temp.write_bytes(b"new")
    calls: list[str] = []
    real_replace = atomic.os.replace

    monkeypatch.setattr(atomic, "fsync_file", lambda path: calls.append(f"file:{Path(path).name}"))
    monkeypatch.setattr(atomic, "fsync_directory", lambda path: calls.append(f"dir:{Path(path).name}"))

    def replace(source, destination):
        calls.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(atomic.os, "replace", replace)
    atomic.durable_replace(temp, target)

    assert calls == ["file:.accepted.tmp", "replace", f"dir:{tmp_path.name}"]
    assert target.read_bytes() == b"new"

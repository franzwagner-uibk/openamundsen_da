from __future__ import annotations

import json
from pathlib import Path

import pytest

from openamundsen_da import cli
from openamundsen_da.exceptions import ProjectRunError


@pytest.mark.parametrize(
    "arguments",
    [
        ["observations", "snow-cover", "/tmp/project", "--json"],
        ["observations", "wet-snow", "/tmp/project", "--json"],
        ["prepare", "/tmp/project", "--json"],
        ["run", "/tmp/project", "--max-workers", "4", "--json"],
        ["render", "/tmp/project", "--json"],
        ["clean", "/tmp/project", "--apply", "--json"],
        ["subdomains", "prepare", "/tmp/project", "--json"],
        ["subdomains", "run", "/tmp/project", "--json"],
        ["subdomains", "merge", "/tmp/project", "--json"],
        ["subdomains", "render", "/tmp/project", "--json"],
        ["subdomains", "model", "prepare", "/tmp/setup", "--json"],
        ["subdomains", "model", "run", "/tmp/setup", "--json"],
        ["subdomains", "model", "merge", "/tmp/setup", "--json"],
    ],
)
def test_supported_command_tree_parses(arguments: list[str]) -> None:
    parsed = cli.build_parser().parse_args(arguments)

    assert parsed.json is True


def test_json_success_envelope_is_stable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "_dispatch",
        lambda _args: {"status": "preview", "path": Path("/tmp/project")},
    )

    exit_code = cli.main(["clean", "/tmp/project", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload == {
        "ok": True,
        "command": "clean",
        "result": {"path": "/tmp/project", "status": "preview"},
    }


def test_json_error_envelope_is_stable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(_args):
        raise ProjectRunError("failed safely")

    monkeypatch.setattr(cli, "_dispatch", fail)

    exit_code = cli.main(["run", "/tmp/project", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload == {
        "ok": False,
        "command": "run",
        "error": {"message": "failed safely", "type": "ProjectRunError"},
    }


def test_clean_defaults_to_preview() -> None:
    parsed = cli.build_parser().parse_args(["clean", "/tmp/project"])

    assert parsed.apply is False

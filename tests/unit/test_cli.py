from __future__ import annotations

import argparse
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


def test_every_public_cli_argument_has_meaningful_help() -> None:
    pending = [cli.build_parser()]
    while pending:
        parser = pending.pop()
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                pending.extend(action.choices.values())
                continue
            if action.dest == "help":
                continue
            assert isinstance(action.help, str)
            assert action.help.strip()


@pytest.mark.parametrize(
    ("arguments", "expected_help"),
    [
        (["run", "--help"], "Limit concurrent open-loop and ensemble propagations"),
        (["clean", "--help"], "without this flag nothing is removed"),
        (["subdomains", "prepare", "--help"], "env/subdomains.gpkg"),
        (["subdomains", "run", "--help"], "inside each subdomain project"),
        (["subdomains", "merge", "--help"], "uncovered edge pixels"),
        (["subdomains", "model", "prepare", "--help"], "Plain openAMUNDSEN setup directory"),
    ],
)
def test_leaf_help_explains_user_visible_behavior(
    arguments: list[str],
    expected_help: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.build_parser().parse_args(arguments)

    assert exc_info.value.code == 0
    normalized_help = " ".join(capsys.readouterr().out.split())
    assert expected_help in normalized_help


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "PROJECT_DIR", "--max-workers", "24"],
        ["subdomains", "prepare", "PROJECT_DIR", "--regions", "PATH"],
        [
            "subdomains",
            "run",
            "PROJECT_DIR",
            "--max-workers",
            "8",
            "--inner-max-workers",
            "4",
        ],
        ["subdomains", "merge", "PROJECT_DIR"],
        ["subdomains", "render", "PROJECT_DIR", "--max-workers", "8"],
        ["subdomains", "model", "prepare", "SETUP_DIR", "--regions", "PATH"],
        ["subdomains", "model", "run", "SETUP_DIR", "--max-workers", "8"],
        ["subdomains", "model", "merge", "SETUP_DIR"],
    ],
)
def test_curated_cli_guide_examples_parse(arguments: list[str]) -> None:
    cli.build_parser().parse_args(arguments)

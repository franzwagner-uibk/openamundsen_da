#!/usr/bin/env python3
"""Validate the published documentation contract without network access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
ASSET_MANIFEST = REPO_ROOT / "tests" / "baselines" / "rofental_es30_manuscript_assets.json"
TUTORIAL_ASSET_ROOT = DOCS_ROOT / "assets" / "images" / "tutorial" / "rofental_2022_2023_es30"
CONFIG_ARCHITECTURE_FIGURE = DOCS_ROOT / "assets" / "images" / "diagrams" / "setup-project-configuration.png"
CONFIG_ARCHITECTURE_FIGURE_SHA256 = "fd2e413b6aaafa2ee2c779456e48cb0ee6e23f28ba66ef31795905cfdf2b13bc"
REQUIRED_TOP_LEVEL_TITLES = (
    "Home",
    "Installation",
    "Input Data",
    "Configuration",
    "Running",
    "Output Data",
    "Example Data",
    "How to Use",
    "Advanced",
    "Reference",
)
REMOVED_PUBLISHED_PATHS = (
    "Tutorial/03-workflow.md",
    "Tutorial/04-framework.md",
    "guides/index.md",
    "guides/cli.md",
    "guides/experiments/index.md",
    "project-structure.md",
    "workflow.md",
    "reference/repo-code-review-2026-04-17.md",
    "reference/repo-code-review-v1-2026-06-15.md",
)
JEKYLL_LINK = re.compile(r"\{%\s*link\s+([^%]+?)\s*%\}")
SITE_ASSET = re.compile(r"\{\{\s*site\.baseurl\s*\}\}/([^\s)?]+)")


def _published_markdown() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(DOCS_ROOT.rglob("*.md"))
        if path.name != "README.md"
        and "_site" not in path.parts
        and ".jekyll-cache" not in path.parts
        and "tmp" not in path.parts
    )


def _front_matter(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        return {}
    try:
        end = lines.index("---", 1)
    except ValueError:
        return {}
    values: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line or line[:1].isspace():
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_front_matter(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    records = {path: _front_matter(path) for path in paths}
    title_paths: dict[str, list[Path]] = {}
    for path, front_matter in records.items():
        if not front_matter:
            errors.append(f"missing or malformed front matter: {path.relative_to(REPO_ROOT)}")
            continue
        title = front_matter.get("title")
        if not title:
            errors.append(f"missing front-matter title: {path.relative_to(REPO_ROOT)}")
            continue
        title_paths.setdefault(title, []).append(path)

    for path, front_matter in records.items():
        parent = front_matter.get("parent")
        if parent and parent not in title_paths:
            errors.append(f"unknown parent {parent!r}: {path.relative_to(REPO_ROOT)}")

    top_level = {
        front_matter.get("title")
        for front_matter in records.values()
        if front_matter and "parent" not in front_matter and front_matter.get("nav_exclude") != "true"
    }
    for title in REQUIRED_TOP_LEVEL_TITLES:
        if title not in top_level:
            errors.append(f"missing required top-level navigation page: {title}")
    return errors


def _validate_links(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for target in JEKYLL_LINK.findall(text):
            resolved = DOCS_ROOT / target.strip()
            if not resolved.is_file():
                errors.append(f"broken Jekyll link in {path.relative_to(REPO_ROOT)}: {target.strip()}")
        for target in SITE_ASSET.findall(text):
            resolved = DOCS_ROOT / target.split("?", 1)[0]
            if not resolved.is_file():
                errors.append(f"missing site asset in {path.relative_to(REPO_ROOT)}: {target}")
    return errors


def _validate_removed_and_stale(paths: Iterable[Path]) -> list[str]:
    errors = [
        f"stale published page still exists: docs/{relative}"
        for relative in REMOVED_PUBLISHED_PATHS
        if (DOCS_ROOT / relative).exists()
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if re.search(r"\boa-da-[a-z0-9-]+", text):
            errors.append(f"legacy installed command appears in {path.relative_to(REPO_ROOT)}")
        if re.search(r"\bv1\.0\b", text, flags=re.IGNORECASE):
            errors.append(f"premature stable-version claim appears in {path.relative_to(REPO_ROOT)}")
    return errors


def _validate_tutorial_assets() -> list[str]:
    manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    records = manifest.get("tutorial_assets")
    if not isinstance(records, list) or not records:
        return [f"missing tutorial asset records: {ASSET_MANIFEST.relative_to(REPO_ROOT)}"]
    errors: list[str] = []
    selected: set[str] = set()
    tutorial_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((DOCS_ROOT / "Tutorial").glob("*.md"))
    )
    for record in records:
        destination = str(record["destination"])
        selected.add(destination)
        path = TUTORIAL_ASSET_ROOT / destination
        if not path.is_file():
            errors.append(f"missing selected tutorial asset: {path.relative_to(REPO_ROOT)}")
        elif _sha256(path) != record["file_sha256"]:
            errors.append(f"selected tutorial asset hash differs: {path.relative_to(REPO_ROOT)}")
        if f"rofental_2022_2023_es30/{destination}" not in tutorial_text:
            errors.append(f"selected tutorial asset is not referenced: {path.relative_to(REPO_ROOT)}")
    present = {path.name for path in TUTORIAL_ASSET_ROOT.glob("*.png")}
    if present != selected:
        errors.append(
            "tutorial reference asset directory differs from the manifest: "
            f"extra={sorted(present - selected)}, missing={sorted(selected - present)}"
        )
    return errors


def _validate_reviewed_documentation_contracts() -> list[str]:
    errors: list[str] = []
    if not CONFIG_ARCHITECTURE_FIGURE.is_file():
        errors.append(f"missing configuration architecture figure: {CONFIG_ARCHITECTURE_FIGURE.relative_to(REPO_ROOT)}")
    elif _sha256(CONFIG_ARCHITECTURE_FIGURE) != CONFIG_ARCHITECTURE_FIGURE_SHA256:
        errors.append(f"configuration architecture figure hash differs: {CONFIG_ARCHITECTURE_FIGURE.relative_to(REPO_ROOT)}")

    figure_reference = "assets/images/diagrams/setup-project-configuration.png"
    for relative in ("guides/observations.md", "guides/configuration.md"):
        path = DOCS_ROOT / relative
        if figure_reference not in path.read_text(encoding="utf-8"):
            errors.append(f"configuration architecture figure is not referenced: docs/{relative}")

    tutorial_home = DOCS_ROOT / "Tutorial" / "index.md"
    if _front_matter(tutorial_home).get("permalink") != "/tutorial/":
        errors.append("Tutorial landing page must use canonical permalink /tutorial/")
    redirect = DOCS_ROOT / "tutorial-uppercase-redirect.md"
    redirect_text = redirect.read_text(encoding="utf-8") if redirect.is_file() else ""
    if _front_matter(redirect).get("permalink") != "/Tutorial/" or "url={{ '/tutorial/'" not in redirect_text:
        errors.append("uppercase /Tutorial/ compatibility redirect is missing or invalid")

    tutorial_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((DOCS_ROOT / "Tutorial").glob("*.md"))
    )
    if "--cpus 8" in tutorial_text or re.search(r"--max-workers\s+(?:4|8)(?:\s|$)", tutorial_text):
        errors.append("tutorial still contains a fixed CPU or worker count")
    for placeholder in ("<CPU_COUNT>", "<MAX_WORKERS>"):
        if placeholder not in tutorial_text:
            errors.append(f"tutorial is missing required hardware placeholder: {placeholder}")

    marked_blocks = "\n".join(
        re.findall(
            r"\*\*🟢 Run command:\*\*\s*```bash\n(.*?)```",
            tutorial_text,
            flags=re.DOTALL,
        )
    )
    for required in (
        "docker run hello-world",
        "docker pull {{ site.data.release.image }}",
        "cp -a /workspace/examples/rofental",
        "bash --noprofile --norc",
        "openamundsen-da --version",
        "echo \"$PROJECT_DIR\"",
        "observations snow-cover",
        "observations wet-snow",
        "openamundsen-da prepare",
        "find \"$PROJECT_DIR/steps\"",
        "openamundsen-da run",
    ):
        if required not in marked_blocks:
            errors.append(f"required tutorial command lacks the run marker: {required}")
    for optional in ("sudo chown", "python - <<'PY'", "openamundsen-da render", "openamundsen-da clean"):
        if optional in marked_blocks:
            errors.append(f"optional tutorial command must not use the required-command marker: {optional}")

    release_text = (DOCS_ROOT / "release.md").read_text(encoding="utf-8")
    for forbidden in (
        "## Release gates",
        "## Repository setup",
        "## Release procedure",
        "pypi.org/project/openamundsen-da",
        "`edge`",
    ):
        if forbidden in release_text:
            errors.append(f"public release page contains maintainer-only or unavailable material: {forbidden}")
    return errors


def validate_docs() -> tuple[str, ...]:
    """Return all documentation contract violations."""
    paths = _published_markdown()
    errors = [
        *_validate_front_matter(paths),
        *_validate_links(paths),
        *_validate_removed_and_stale(paths),
        *_validate_tutorial_assets(),
        *_validate_reviewed_documentation_contracts(),
    ]
    return tuple(errors)


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        print("validate_docs.py does not accept arguments", file=sys.stderr)
        return 2
    errors = validate_docs()
    if errors:
        print("Documentation validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Documentation validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

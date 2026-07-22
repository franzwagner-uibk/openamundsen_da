from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "release" / "validate_release.py"
ROOT = SCRIPT.parents[2]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
DOCS_WORKFLOW = ROOT / ".github" / "workflows" / "deploy-docs.yml"
CLOUDFLARE_WORKFLOW = ROOT / ".github" / "workflows" / "deploy-cloudflare.yml"
SPEC = importlib.util.spec_from_file_location("validate_release", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
validate_release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate_release)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v0.9.0rc1", ("0.9.0rc1", True)),
        ("v0.9.0rc12", ("0.9.0rc12", True)),
        ("v0.9.0", ("0.9.0", False)),
    ],
)
def test_release_from_tag(tag: str, expected: tuple[str, bool]) -> None:
    assert validate_release.release_from_tag(tag) == expected


@pytest.mark.parametrize(
    "tag",
    ["0.9.0", "v0.9", "v0.9.0rc0", "v0.9.0-beta1", "v1.0.0.dev1"],
)
def test_release_from_tag_rejects_unsupported_forms(tag: str) -> None:
    with pytest.raises(ValueError, match="Unsupported release tag"):
        validate_release.release_from_tag(tag)


def test_fiona_trivy_exception_requires_non_vulnerable_gdal_pin() -> None:
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    ignore = (ROOT / ".trivyignore").read_text(encoding="utf-8")

    assert re.search(r"^\s*-\s+fiona=1\.9\.6\s*$", environment, re.MULTILINE)
    assert re.search(r"^\s*-\s+gdal=3\.8\.5\s*$", environment, re.MULTILINE)
    assert "GHSA-q5fm-55c2-v6j9 exp:2027-07-15" in ignore


def test_release_workflow_prepares_metadata_directory_before_sbom() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    prepare_step = "      - name: Prepare release metadata directory"
    sbom_step = "      - name: Generate SPDX package SBOM"

    prepare_index = workflow.index(prepare_step)
    sbom_index = workflow.index(sbom_step)

    assert prepare_index < sbom_index
    assert "run: mkdir -p release-metadata" in workflow[prepare_index:sbom_index]


def test_release_workflow_checksums_match_flat_github_release_assets() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    package_start = workflow.index("  package:")
    package_end = workflow.index("\n  portable:", package_start)
    package_job = workflow[package_start:package_end]

    assert "cd dist && sha256sum *.whl *.tar.gz" in package_job
    assert "cd release-metadata && sha256sum *.spdx.json" in package_job
    assert "sha256sum dist/*.whl dist/*.tar.gz" not in package_job
    assert "      - name: Verify flat release bundle checksums" in package_job
    assert 'release-metadata/SHA256SUMS "${verify_dir}/"' in package_job
    assert '(cd "${verify_dir}" && sha256sum -c SHA256SUMS)' in package_job


def test_release_workflow_stages_downloaded_wheel_before_p8_smoke() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    trusted_start = workflow.index("  trusted-integration:")
    trusted_end = workflow.index("\n  publish-python:", trusted_start)
    trusted_job = workflow[trusted_start:trusted_end]
    stage_step = "      - name: Stage exact release wheel"
    smoke_step = "      - name: Validate installed wheel interface"

    stage_index = trusted_job.index(stage_step)
    smoke_index = trusted_job.index(smoke_step)

    assert stage_index < smoke_index
    assert "cp release/dist/openamundsen_da-*.whl dist/" in trusted_job[stage_index:smoke_index]


def test_release_workflow_verifies_published_manifest_digest_from_raw_bytes() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    promotion_start = workflow.index("  publish-container:")
    promotion_end = workflow.index("\n  github-release:", promotion_start)
    promotion_job = workflow[promotion_start:promotion_end]

    assert 'imagetools inspect "${IMAGE}:${VERSION}" --raw' in promotion_job
    assert "| sha256sum" in promotion_job
    assert '[[ "${published_digest}" != "${EXPECTED_DIGEST}" ]]' in promotion_job
    assert 'grep -F "Digest: ${EXPECTED_DIGEST}"' not in promotion_job


def test_release_image_contains_both_examples_without_agent_guidance() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "COPY examples/rofental /workspace/examples/rofental" in dockerfile
    assert "COPY examples/subdomains /workspace/examples/subdomains" in dockerfile
    assert "!examples/rofental/**" in dockerignore
    assert "!examples/subdomains/**" in dockerignore
    assert "test -f /workspace/examples/rofental/rofental.yml" in workflow
    assert "test -f /workspace/examples/subdomains/subdomains.yml" in workflow
    assert "find /workspace -name AGENTS.md" in workflow
    source_agent_files = tuple(
        path
        for path in ROOT.rglob("AGENTS.md")
        if not {".git", "_site", ".jekyll-cache"}.intersection(path.parts)
    )
    assert not source_agent_files


def test_ci_docs_only_scope_is_narrow_and_full_ci_is_the_fallback() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    classifier = (ROOT / "scripts" / "ci" / "classify_changes.py").read_text(
        encoding="utf-8"
    )

    assert '"tests/baselines/rofental_es30_tutorial_assets.json"' in classifier
    assert 'PurePosixPath("CHANGELOG.md")' in classifier
    assert 'path.parts[0] == "docs"' in classifier
    assert "rofental_es30_manuscript_assets.json" not in classifier
    assert "python scripts/ci/classify_changes.py --force-full" in workflow
    assert "git diff --name-only --no-renames" in workflow
    assert "if: needs.change-scope.outputs.docs_only != 'true'" in workflow
    assert "name: CI gate" in workflow
    assert "test \"${TRUSTED_RESULT}\" = success" in workflow


def test_docs_deployment_tracks_only_the_dedicated_tutorial_asset_manifest() -> None:
    workflow = DOCS_WORKFLOW.read_text(encoding="utf-8")

    assert "tests/baselines/rofental_es30_tutorial_assets.json" in workflow
    assert "tests/baselines/rofental_es30_manuscript_assets.json" not in workflow


def test_docs_deployment_uses_github_pages_and_keeps_cloudflare_gated() -> None:
    pages = DOCS_WORKFLOW.read_text(encoding="utf-8")
    cloudflare = CLOUDFLARE_WORKFLOW.read_text(encoding="utf-8")

    assert "name: Deploy Docs to GitHub Pages" in pages
    assert "  push:\n    branches:\n      - main" in pages
    assert "actions/configure-pages@" in pages
    assert "actions/upload-pages-artifact@" in pages
    assert "actions/deploy-pages@" in pages
    assert "name: github-pages" in pages
    assert "cloudflare/wrangler-action@" not in pages

    assert "name: Deploy Docs to Cloudflare Pages (gated fallback)" in cloudflare
    assert "  push:\n    branches:\n      - main" in cloudflare
    assert "  workflow_dispatch:" in cloudflare
    assert (
        "if: github.event_name == 'workflow_dispatch' || "
        "vars.CLOUDFLARE_AUTO_DEPLOY == 'true'" in cloudflare
    )
    assert "cloudflare/wrangler-action@" in cloudflare
    assert "pages deploy docs/_site --project-name=openamundsen-da" in cloudflare


def test_container_publication_uses_hyphenated_organization_namespace() -> None:
    image = "ghcr.io/openamundsen/openamundsen-da"

    assert f"IMAGE: {image}" in CI_WORKFLOW.read_text(encoding="utf-8")
    assert f"IMAGE: {image}" in RELEASE_WORKFLOW.read_text(encoding="utf-8")
    assert f"${{IMAGE:-{image}:latest}}" in (ROOT / "compose.yml").read_text(
        encoding="utf-8"
    )
    assert (
        'org.opencontainers.image.source="https://github.com/openamundsen/openamundsen-da"'
        in (ROOT / "Dockerfile").read_text(encoding="utf-8")
    )

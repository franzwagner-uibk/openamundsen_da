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
CITATION = ROOT / "CITATION.cff"
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


def test_citation_metadata_is_validated_in_ci_and_release() -> None:
    citation = CITATION.read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    distribution_validator = (
        ROOT / "scripts" / "ci" / "validate_distribution.py"
    ).read_text(encoding="utf-8")

    assert 'version: "0.9.2"' in citation
    assert 'doi: "10.5281/zenodo.21519389"' in citation
    assert "preferred-citation:" not in citation
    assert not (ROOT / ".zenodo.json").exists()
    assert "include CITATION.cff" in manifest
    assert '"CITATION.cff"' in distribution_validator
    for workflow_path in (CI_WORKFLOW, RELEASE_WORKFLOW):
        workflow = workflow_path.read_text(encoding="utf-8")
        assert "cffconvert==2.0.0" in workflow
        assert "cffconvert --validate" in workflow


def test_release_workflow_stages_downloaded_wheel_before_p8_smoke() -> None:
    workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    trusted_start = workflow.index("  trusted-integration:")
    trusted_end = workflow.index("\n  native-integration:", trusted_start)
    trusted_job = workflow[trusted_start:trusted_end]
    stage_step = "      - name: Stage exact release wheel"
    smoke_step = "      - name: Validate installed wheel interface"

    stage_index = trusted_job.index(stage_step)
    smoke_index = trusted_job.index(smoke_step)

    assert stage_index < smoke_index
    assert "cp release/dist/openamundsen_da-*.whl dist/" in trusted_job[stage_index:smoke_index]


def test_native_pip_rofental_is_a_ci_and_release_gate() -> None:
    ci_workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    release_workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "ci" / "run_native_integration_tests.sh").read_text(
        encoding="utf-8"
    )
    integration = (ROOT / "scripts" / "ci" / "run_integration_tests.sh").read_text(
        encoding="utf-8"
    )
    api_driver = (ROOT / "scripts" / "ci" / "run_project_api.py").read_text(
        encoding="utf-8"
    )
    constraints = (ROOT / "constraints" / "native-ci-py312.txt").read_text(
        encoding="utf-8"
    )
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    ci_start = ci_workflow.index("  native-integration:")
    ci_end = ci_workflow.index("\n  publish-edge:", ci_start)
    ci_job = ci_workflow[ci_start:ci_end]
    release_start = release_workflow.index("  native-integration:")
    release_end = release_workflow.index("\n  publish-python:", release_start)
    release_job = release_workflow[release_start:release_end]

    for job in (ci_job, release_job):
        assert "runs-on: [self-hosted, linux, x64, oa-da]" in job
        assert 'python-version: "3.12"' in job
        assert "bash scripts/ci/run_native_integration_tests.sh" in job
        assert "OA_DA_TEST_SETUP_RESOLUTION=500" in job
        assert "OA_DA_TEST_ENSEMBLE_SIZE=2" in job
        assert "OA_DA_TEST_MAX_WORKERS=8" in job
        assert "constraints/native-ci-py312.txt" in job

    assert "needs: [change-scope, package, trusted-integration, native-integration]" in ci_workflow
    assert "test \"${NATIVE_RESULT}\" = success" in ci_workflow
    assert "native_dependency_mode:" in ci_workflow
    assert "inputs.native_dependency_mode || 'locked'" in ci_job
    assert "needs: [package, trusted-integration, native-integration]" in release_workflow
    assert "OA_DA_NATIVE_DEPENDENCY_MODE=locked" in release_job
    assert "matplotlib>=3.10" in runner
    assert "pip check" in runner
    assert "pip freeze" in runner
    assert "OA_DA_NATIVE_DEPENDENCY_MODE:-locked" in runner
    assert "OA_DA_NATIVE_PIP_VERSION:-26.1.2" in runner
    assert '"pip==${PIP_VERSION}"' in runner
    assert "validate_installed_wheel.py" in runner
    assert "OA_DA_TEST_RUNTIME=native" in runner
    assert "OA_DA_TEST_PROJECT_DRIVER=api" in runner
    assert "native)" in integration
    assert 'PROJECT_DRIVER="${OA_DA_TEST_PROJECT_DRIVER:-cli}"' in integration
    assert "scripts/ci/run_project_api.py" in integration
    assert "from openamundsen_da import prepare_project, run_project" in api_driver
    assert 'if __name__ == "__main__":' in api_driver
    assert 'env -u PYTHONHOME -u PYTHONPATH PYTHONNOUSERSITE=1 "$@"' in integration
    assert "matplotlib==3.10.9" in constraints
    assert "openamundsen==1.2.1" in constraints
    assert "openamundsen-da" not in constraints
    assert "prune constraints" in manifest


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
    assert "test \"${NATIVE_RESULT}\" = success" in workflow


def test_docs_deployment_tracks_only_the_dedicated_tutorial_asset_manifest() -> None:
    workflow = DOCS_WORKFLOW.read_text(encoding="utf-8")

    assert "tests/baselines/rofental_es30_tutorial_assets.json" in workflow
    assert "tests/baselines/rofental_es30_manuscript_assets.json" not in workflow


def test_docs_deployment_uses_github_pages_and_keeps_cloudflare_manual() -> None:
    pages = DOCS_WORKFLOW.read_text(encoding="utf-8")
    cloudflare = CLOUDFLARE_WORKFLOW.read_text(encoding="utf-8")

    assert "name: Deploy Docs to GitHub Pages" in pages
    assert "  push:\n    branches:\n      - main" in pages
    assert "actions/configure-pages@" in pages
    assert "actions/upload-pages-artifact@" in pages
    assert "actions/deploy-pages@" in pages
    assert "name: github-pages" in pages
    assert "cloudflare/wrangler-action@" not in pages

    assert "name: Deploy Docs to Cloudflare Pages (manual fallback)" in cloudflare
    assert "  workflow_dispatch:" in cloudflare
    assert "  push:" not in cloudflare
    assert "CLOUDFLARE_AUTO_DEPLOY" not in cloudflare
    assert "cloudflare/wrangler-action@" in cloudflare
    assert "pages deploy docs/_site --project-name=openamundsen-da" in cloudflare


def test_container_publication_and_public_usage_target_organization() -> None:
    publication_image = "ghcr.io/openamundsen/openamundsen-da"
    public_image = f"{publication_image}:0.9.2"
    public_package_page = (
        "https://github.com/openamundsen/openamundsen-da/"
        "pkgs/container/openamundsen-da"
    )
    compose = (ROOT / "compose.yml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_release = (ROOT / "docs" / "_data" / "release.yml").read_text(
        encoding="utf-8"
    )
    docs_index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    docs_distribution = (ROOT / "docs" / "release.md").read_text(
        encoding="utf-8"
    )

    assert f"IMAGE: {publication_image}" in CI_WORKFLOW.read_text(encoding="utf-8")
    assert f"IMAGE: {publication_image}" in RELEASE_WORKFLOW.read_text(
        encoding="utf-8"
    )
    assert f"${{IMAGE:-{public_image}}}" in compose
    assert f"docker pull {public_image}" in readme
    assert f'image: "{public_image}"' in docs_release
    assert public_package_page in docs_index
    assert public_package_page in docs_distribution
    assert "ghcr.io/franzwagner-uibk/openamundsen_da" not in compose
    assert "ghcr.io/franzwagner-uibk/openamundsen_da" not in readme
    assert "ghcr.io/franzwagner-uibk/openamundsen_da" not in docs_release
    assert (
        'org.opencontainers.image.source="https://github.com/openamundsen/openamundsen-da"'
        in (ROOT / "Dockerfile").read_text(encoding="utf-8")
    )

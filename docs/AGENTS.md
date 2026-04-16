# Docs Agent Notes

Inherit the repo-root `AGENTS.md`. This file adds documentation-specific rules.

- Treat `docs/*.md`, `docs/Tutorial/`, `docs/guides/`, `docs/reference/`, and config files such as `_config.yml` as source.
- Treat `docs/_site/`, `docs/.jekyll-cache/`, `docs/tmp/`, and `bundle.log` files as generated or local artifacts, not editable source.
- Keep documentation aligned with actual CLI names, config ownership, example structure, and CI behavior.
- Keep scientific framing honest: document the current ROI-scale PF workflow, current observation support, and current limitations explicitly instead of implying future localization or multivariate capabilities already exist.
- Prefer updating the narrowest relevant page set rather than restating the same guidance in many pages.
- If workflow or command examples change, verify they still match `README.md` and the CI wrapper scripts.

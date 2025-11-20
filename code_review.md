# Code review summary

This review follows the expectations in `CODE_REVIEW_GUIDELINES.md` with a focus on cleanliness, reuse of helpers, and respecting the documented configuration pathways.

## 1. `assimilate_scf` references helpers that are never imported
- `_read_likelihood_from_project` calls `find_project_yaml` and `_read_yaml_file` but the module only imports `list_member_dirs` and `default_results_dir`. At runtime this raises `NameError` as soon as the function executes, preventing any assimilation run from starting.
- The fix is simply to import the helpers that are already provided in `io.paths` and `core.env` (per the "Consistency" and "Reuse existing helpers" guidance).

Impacted code: `openamundsen_da/methods/pf/assimilate_scf.py` lines 41-74. 【F:openamundsen_da/methods/pf/assimilate_scf.py†L41-L74】

## 2. Rejuvenation hard-codes `project.yml` instead of using the shared path helpers
- `_read_rejuvenation_params` does `Path(project_dir) / "project.yml"`, which breaks as soon as the caller passes a non-default layout (e.g., project YAML named differently or located elsewhere). The repository already exposes `find_project_yaml`, and the guidelines explicitly call out consolidating IO/path logic.
- This should be refactored to reuse `openamundsen_da.io.paths.find_project_yaml`, mirroring the approach used throughout `prior_forcing` and other PF modules. Doing so keeps the module aligned with the overall structure and avoids brittle assumptions about the file system layout.

Impacted code: `openamundsen_da/methods/pf/rejuvenate.py` lines 70-90. 【F:openamundsen_da/methods/pf/rejuvenate.py†L70-L90】

## 3. `run_member` ignores the configurable restart/dump flags
- The launcher CLI exposes `--restart-from-state` and `--dump-state`, and `launch_members` forwards those flags into `run_member`. However, `run_member` immediately overrides the behavior based purely on the step name (always cold start for `step_00*`, always warm + dump for later steps) and never inspects the `restart_from_state` or `dump_state` arguments. As a result the CLI switches have no effect, which violates the "Consistency" and "Configuration" sections of the guidelines.
- Either honor the explicit arguments (taking precedence over project defaults) or remove the unused parameters to avoid a misleading public API. Ideally, the function should read the restart policy in one place (project.yml + CLI override) and pass the resulting booleans into the warm-start/dump logic.

Impacted code: `openamundsen_da/core/runner.py` lines 193-287. 【F:openamundsen_da/core/runner.py†L193-L287】

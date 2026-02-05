# Tester access to private GHCR image

## Owner steps (grant access)
1. **Repo access** (either way is fine):
   - Add the tester as a collaborator on the GitHub repo: `Settings → Collaborators`.
   - Or, on the GHCR package page, open **Package settings → Manage access** and grant the tester **Read**. Ensure the package either inherits repo permissions or lists the tester explicitly.

## Tester steps (pull and run)
1. Create a GitHub PAT with **read:packages** scope  
   GitHub → Settings → Developer settings → Personal access tokens (classic) → New token → check **read:packages** → generate → copy the token.
2. Log in to GHCR with your PAT (no shared creds):
   ```bash
   echo "$PAT" | docker login ghcr.io -u <your-github-username> --password-stdin
   ```
3. Pull the image:
   ```bash
   docker pull ghcr.io/franzwagner-uibk/openamundsen_da
   ```
4. Run the Rofental quickstart (no clone):
   ```bash
   mkdir -p openamundsen-da
   docker run --rm -v "$(pwd)/openamundsen-da:/data" \
     ghcr.io/franzwagner-uibk/openamundsen_da \
     bash -lc "cp -a /workspace/examples/rofental /data/rofental && \
               python -m openamundsen_da.pipeline.season_skeleton \
                 --project-dir /data/rofental \
                 --season-dir /data/rofental/propagation/season_2022_2023 \
                 --log-level INFO && \
               python -m openamundsen_da.pipeline.season \
                 --project-dir /data/rofental \
                 --season-dir /data/rofental/propagation/season_2022_2023 \
                 --max-workers 8 \
                 --perf-monitor \
                 --overwrite \
                 --log-level INFO"
   ```
   Outputs/logs end up under `openamundsen-da/rofental/propagation/season_2022_2023` on the host.

Notes:
- If the image becomes public, steps 1–2 aren’t needed; `docker pull ghcr.io/franzwagner-uibk/openamundsen_da` works unauthenticated.
- Owners should not share their PAT; testers use their own.

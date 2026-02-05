# Tester access to private GHCR image

## Owner steps
- Add the tester as a collaborator on the GitHub repo **or** grant them access on the GHCR package page.
- Ensure the GHCR package inherits repo permissions or explicitly grants the tester read access.

## Tester steps
1. Create a GitHub PAT with `read:packages` scope.
2. Log in to GHCR using their own PAT (no shared credentials):
   ```bash
   echo "$PAT" | docker login ghcr.io -u <github-username> --password-stdin
   ```
3. Pull and run:
   ```bash
   docker pull ghcr.io/franzwagner-uibk/openamundsen_da
   ```
   Then follow the quickstart in the docs to run the Rofental example.

Notes:
- If the package is public, steps 1–2 are not needed.
- Do not share your PAT; testers use their own.

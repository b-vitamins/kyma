# AGENTS

## Remotectl

- Machine definitions live in a root `.env` file that must stay out of git.
- Keep machine names, login details, queue names, container tags, and host
  topology out of committed defaults.
- Prefer `AUTH=key` with `SSH_KEY` configured. Keep `PASSWORD` populated only as
  an operator fallback while public-key access is being repaired or rolled out.
- Primary commands:
  - `./scripts/remote-list`
  - `./scripts/remote-print-config --machine <name>`
  - `./scripts/remote-shell --machine <name>`
  - `./scripts/remote-rsync --machine <name>`
  - `./scripts/remote-smoke --machine <name>`
- The scripts use a repo-local `.remote-known-hosts` file for non-interactive
  access. Any manual SSH aliases belong in the user's `~/.ssh/config`.
- Keep this tooling lean and generic. The goal is reliable operator control,
  not a machine-specific orchestration layer.

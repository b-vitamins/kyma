# kyma

Minimal repository basics with optional remote helper tooling.

## Local Setup

Bootstrap a local virtual environment:

```bash
bash scripts/bootstrap-venv.sh
```

Run the local quality gate:

```bash
make pre-commit
```

## Remote Helpers

The remote helpers are optional operator tooling under `scripts/`.

Copy the example config and fill in your machine details:

```bash
cp .env.example .env
```

Primary commands:

- `./scripts/remote-list`
- `./scripts/remote-print-config --machine <name>`
- `./scripts/remote-shell --machine <name>`
- `./scripts/remote-rsync --machine <name>`
- `./scripts/remote-smoke --machine <name>`

See `scripts/README.md` for details.

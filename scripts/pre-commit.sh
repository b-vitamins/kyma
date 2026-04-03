#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
venv_bin="${repo_root}/.venv/bin"

if [[ ! -x "${venv_bin}/python" ]]; then
  echo "Expected a local .venv at ${repo_root}/.venv. Run scripts/bootstrap-venv.sh first." >&2
  exit 1
fi

for tool in pyright ruff pytest; do
  if [[ ! -x "${venv_bin}/${tool}" ]]; then
    echo "Missing ${tool} in ${venv_bin}. Run scripts/bootstrap-venv.sh first." >&2
    exit 1
  fi
done

"${venv_bin}/pyright"
"${venv_bin}/ruff" check .
"${venv_bin}/ruff" format --check .
"${venv_bin}/pytest"

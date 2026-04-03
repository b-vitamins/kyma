#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
venv_dir="${repo_root}/.venv"
python_bin="${venv_dir}/bin/python"
bootstrap_python="${BOOTSTRAP_PYTHON:-python3}"

cd "${repo_root}"

command -v "${bootstrap_python}" >/dev/null 2>&1 || {
  echo "Missing bootstrap interpreter: ${bootstrap_python}" >&2
  exit 1
}

"${bootstrap_python}" -m venv --clear "${venv_dir}"
"${python_bin}" -m pip install --upgrade pip setuptools wheel
"${python_bin}" -m pip install --upgrade \
  "pytest>=9" \
  "ruff>=0.9.5" \
  "pyright>=1.1.403"

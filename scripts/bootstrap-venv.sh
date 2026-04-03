#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
venv_dir="${repo_root}/.venv"
python_bin="${venv_dir}/bin/python"
bootstrap_python="${BOOTSTRAP_PYTHON:-python3}"

patch_elf_interpreter() {
  local binary_path="$1"
  local loader_path

  if [[ ! -x "${binary_path}" ]]; then
    return 0
  fi
  if ! command -v patchelf >/dev/null 2>&1; then
    return 0
  fi
  loader_path="$(
    readelf -l "$(readlink -f "${python_bin}")" 2>/dev/null \
      | sed -n 's/.*Requesting program interpreter: \(.*\)]/\1/p'
  )"
  if [[ -z "${loader_path}" ]]; then
    return 0
  fi

  if file "${binary_path}" | grep -q 'interpreter /lib64/ld-linux-x86-64.so.2'; then
    patchelf --set-interpreter "${loader_path}" "${binary_path}"
  fi
}

patch_elf_rpath() {
  local binary_path="$1"
  local runpath

  if [[ ! -x "${binary_path}" ]]; then
    return 0
  fi
  if ! command -v patchelf >/dev/null 2>&1; then
    return 0
  fi
  runpath="$(
    readelf -d "$(readlink -f "${python_bin}")" 2>/dev/null \
      | sed -n 's/.*Library runpath: \[\(.*\)\]/\1/p'
  )"
  if [[ -z "${runpath}" ]]; then
    return 0
  fi

  patchelf --set-rpath "${runpath}" "${binary_path}"
}

cd "${repo_root}"

command -v "${bootstrap_python}" >/dev/null 2>&1 || {
  echo "Missing bootstrap interpreter: ${bootstrap_python}" >&2
  exit 1
}

"${bootstrap_python}" -m venv --clear "${venv_dir}"
"${python_bin}" -m pip install --upgrade pip setuptools wheel
"${python_bin}" -m pip install --upgrade -e '.[all]'
patch_elf_interpreter "${venv_dir}/bin/ruff"
patch_elf_rpath "${venv_dir}/bin/ruff"

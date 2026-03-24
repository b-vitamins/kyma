#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
venv_dir="${repo_root}/.venv"
python_bin="${venv_dir}/bin/python"
vendor_dir="${repo_root}/artifacts/vendor"

cd "${repo_root}"

run_local_pip() {
  "${python_bin}" - "${venv_site_packages}" "$@" <<'PY'
import sys

site_packages = sys.argv[1]
pip_args = sys.argv[2:]
sys.path.insert(0, site_packages)

from pip._internal.cli.main import main

raise SystemExit(main(pip_args))
PY
}

use_system_torch=0
if python3 - <<'PY'
import sys

try:
    import torch
except ModuleNotFoundError:
    sys.exit(1)

sys.exit(0 if torch.__version__.startswith("2.10.0") else 1)
PY
then
  use_system_torch=1
fi

ariautils_source="${KYMA_ARIAUTILS_SOURCE:-git+https://github.com/EleutherAI/aria-utils.git}"
if [[ -d "${vendor_dir}/ariautils-src" && -z "${KYMA_ARIAUTILS_SOURCE:-}" ]]; then
  ariautils_source="${vendor_dir}/ariautils-src"
fi
slinoss_source="${KYMA_SLINOSS_WHEEL:-https://github.com/b-vitamins/slinoss/releases/download/v0.1.1/slinoss-0.1.1-py3-none-any.whl}"

python3 -m venv --clear "${venv_dir}"
venv_site_packages="$("${python_bin}" - <<'PY'
import sysconfig

print(sysconfig.get_path("purelib"))
PY
)"
if [[ "${use_system_torch}" -eq 1 ]]; then
  system_site_packages="$(
    python3 - <<'PY'
from collections import OrderedDict
from pathlib import Path
import site

import torch

paths = OrderedDict()
paths[str(Path(torch.__file__).resolve().parents[1])] = None
for site_path in site.getsitepackages():
    paths[str(Path(site_path).resolve())] = None

for path in paths:
    print(path)
PY
  )"
  printf '%s\n' "${system_site_packages}" > "${venv_site_packages}/kyma-system-site.pth"
fi
run_local_pip install --upgrade pip setuptools wheel

if [[ "${use_system_torch}" -eq 1 ]]; then
  echo "Reusing compatible system torch for local .venv bootstrap."
  run_local_pip install --upgrade \
    "numpy>=1.26" \
    "tqdm>=4.66" \
    "jsonlines>=4.0" \
    "safetensors>=0.4.5" \
    "mido>=1.3.3"
  run_local_pip install --no-deps \
    "${ariautils_source}" \
    "${slinoss_source}"
  run_local_pip install --upgrade --force-reinstall \
    "pytest>=9" \
    "ruff>=0.9.5" \
    "pyright>=1.1.403"
  if ! "${venv_dir}/bin/ruff" --version >/dev/null 2>&1; then
    system_ruff="$(command -v ruff || true)"
    if [[ -n "${system_ruff}" ]]; then
      printf '%s\n' \
        '#!/usr/bin/env bash' \
        'set -euo pipefail' \
        "exec \"${system_ruff}\" \"\$@\"" > "${venv_dir}/bin/ruff"
      chmod +x "${venv_dir}/bin/ruff"
    fi
  fi
  run_local_pip install --no-deps .
else
  echo "No compatible system torch detected. Installing the full local dependency stack."
  run_local_pip install ".[dev]"
fi

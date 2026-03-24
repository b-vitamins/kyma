# Install

Kyma consumes published `slinoss` release wheels rather than a source checkout.
That keeps the dependency contract aligned with the upstream package release
pipeline.

## CPU / Reference Install

For the default repo-local install:

```bash
scripts/bootstrap-venv.sh
```

The bootstrap script recreates `.venv`, upgrades the local packaging tooling,
and installs Kyma plus the developer extras as a non-editable local virtualenv
environment. On workstations that already expose the matching `torch==2.10.0`
runtime in the system Python, the script reuses that runtime and installs the
rest of Kyma's dependencies locally to avoid re-downloading multi-gigabyte CUDA
wheels. On Linux `x86_64` with Python 3.11 and `nvidia-smi` available, the
bootstrap script now prefers the published `slinoss` CUDA wheel and also
installs the matching local `nvidia-cutlass-dsl` and `cuda-python` runtime
dependencies. If no compatible system torch is present, it falls back to the
full local dependency install and then reapplies the resolved `ariautils` /
`slinoss` wheel sources.

If GitHub is unavailable but you have a local non-editable source checkout or
wheel, the bootstrap script also accepts:

```bash
KYMA_ARIAUTILS_SOURCE=/abs/path/to/ariautils-src \
KYMA_SLINOSS_WHEEL=/abs/path/to/slinoss.whl \
scripts/bootstrap-venv.sh
```

It will also automatically prefer an ignored local source cache at
`artifacts/vendor/ariautils-src/` when present. Setting
`KYMA_SLINOSS_WHEEL` also overrides the bootstrap script's default CUDA-wheel
preference on supported GPU workstations.

## CUDA Install

If you want the CuTe backend and the compiled causal-conv extension, install the
matching CUDA wheel directly, mirroring the upstream `slinoss` release guidance.
For Linux `x86_64` on Python 3.11:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  "torch==2.10.0" \
  "slinoss[cuda] @ https://github.com/b-vitamins/slinoss/releases/download/v0.1.1/slinoss-0.1.1-cp311-cp311-linux_x86_64.whl" \
  "ariautils @ git+https://github.com/EleutherAI/aria-utils.git" \
  "numpy>=1.26" \
  "tqdm>=4.66" \
  "jsonlines>=4.0" \
  "safetensors>=0.4.5" \
  "pytest>=9" \
  "ruff>=0.9.5" \
  "pyright>=1.1.403"
python -m pip install .
```

That keeps all dependencies isolated to the local virtualenv without relying on
editable installs.

## Notes

- Replace `v0.1.1` / `0.1.1` with a newer `slinoss` release when Kyma upgrades.
- Pick the wheel whose Python and platform tags match your environment.
- Run local source-tree commands through the venv interpreter, for example:

```bash
.venv/bin/python -m kyma.cli list-configs model
```

- The strict local quality gate remains:

```bash
make pre-commit
```

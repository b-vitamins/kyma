# Install

Kyma consumes published `slinoss` release wheels rather than a source checkout.
That keeps the dependency contract aligned with the upstream package release
pipeline.

## CPU / Reference Install

For a Python 3.11 environment with the universal `slinoss` wheel:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This follows the universal wheel contract used in
[pyproject.toml](/home/b/projects/kyma/pyproject.toml).

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
python -m pip install -e . --no-deps
```

That keeps Kyma editable while preserving the release-wheel install path for the
backend package.

## Notes

- Replace `v0.1.1` / `0.1.1` with a newer `slinoss` release when Kyma upgrades.
- Pick the wheel whose Python and platform tags match your environment.
- The strict local quality gate remains:

```bash
make pre-commit
```

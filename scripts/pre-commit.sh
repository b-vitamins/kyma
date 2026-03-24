#!/usr/bin/env bash
set -euo pipefail

pyright
ruff check .
ruff format --check .
pytest

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
kyma_exp_dir="${repo_root}/experiments/ada"
aria_root="${ARIA_BASELINE_ROOT:-/data/home/ayand/aria-baseline}"

kill_pattern() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "$pattern" || true)"
  if [[ -n "${pids}" ]]; then
    kill ${pids} || true
    sleep 2
    pids="$(pgrep -f "$pattern" || true)"
    if [[ -n "${pids}" ]]; then
      kill -9 ${pids} || true
    fi
  fi
}

collect_run_ids() {
  local root="$1"
  if [[ -d "${root}" ]]; then
    find "${root}" -type d -path '*/wandb/run-*' -printf '%f\n' 2>/dev/null \
      | sed -E 's/^run-[0-9_]+-//' \
      | sort -u
  fi
}

delete_wandb_runs() {
  local ids
  ids="$( { collect_run_ids "${kyma_exp_dir}"; collect_run_ids "${aria_root}"; } | sort -u )"
  if [[ -z "${ids}" ]]; then
    return
  fi
  RUN_IDS="${ids}" \
  WANDB_ENTITY="${WANDB_ENTITY:-incado1010-iisc}" \
  WANDB_PROJECT="${WANDB_PROJECT:-kyma}" \
  "${repo_root}/.venv/bin/python" - <<'PY'
import os

ids = [line.strip() for line in os.environ.get("RUN_IDS", "").splitlines() if line.strip()]
if not ids:
    raise SystemExit(0)

try:
    import wandb
except Exception:
    raise SystemExit(0)

api = wandb.Api()
entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
for run_id in ids:
    try:
        api.run(f"{entity}/{project}/{run_id}").delete(delete_artifacts=True)
    except Exception:
        continue
PY
}

delete_wandb_runs

kill_pattern "${repo_root}.*kyma.training.pretrain"
kill_pattern "${aria_root}.*aria/training/train.py"

find "${kyma_exp_dir}" -mindepth 1 -maxdepth 1 \
  \( -type d \( -name 'runs' -o -name 'kyma-*' -o -name 'diag-*' \) -o \
     -type f \( -name '*.launcher.log' -o -name 'bench-*.log' -o -name 'midi-*.log' -o -name 'pack-*.log' \) \) \
  -exec rm -rf {} +
mkdir -p "${kyma_exp_dir}/runs"

if [[ -d "${aria_root}" ]]; then
  rm -rf "${aria_root}/experiments" "${aria_root}/wandb"
  mkdir -p "${aria_root}/experiments"
  find "${aria_root}" -maxdepth 1 -type f -name '*.launcher.log' -delete
fi

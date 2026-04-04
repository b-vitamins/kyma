#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
aria_root="${ARIA_BASELINE_ROOT:-/data/home/ayand/aria-baseline}"
train_pack="${TRAIN_PACK_DIR:-/data/home/ayand/datasets/kyma-pretrain-pruned/train}"
val_pack="${VAL_PACK_DIR:-/data/home/ayand/datasets/kyma-pretrain-pruned/val}"
aria_train_dir="${ARIA_TRAIN_DIR:-/data/home/ayand/datasets/aria-pretrain-pruned/train}"
aria_val_dir="${ARIA_VAL_DIR:-/data/home/ayand/datasets/aria-pretrain-pruned/val}"
run_stamp="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
group="${WANDB_RUN_GROUP:-aria-kyma-fullpass-${run_stamp}}"
eval_every="${EVAL_EVERY:-250}"
save_every="${SAVE_EVERY:-250}"
batch_size="${BATCH_SIZE:-8}"
workers="${WORKERS:-2}"

kyma_run="${repo_root}/experiments/ada/runs/kyma-base-fullpass-${run_stamp}"
aria_run="${aria_root}/experiments/aria-medium-fullpass-${run_stamp}"

steps="$("${repo_root}/.venv/bin/python" - <<'PY'
import json, math
from pathlib import Path

manifest = json.loads(
    Path("/data/home/ayand/datasets/kyma-pretrain-pruned/train/manifest.json").read_text()
)
print(math.ceil(manifest["sequence_count"] / 8))
PY
)"

patch_aria_train() {
  ARIA_TRAIN_PATH="${aria_root}/aria/training/train.py" python - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["ARIA_TRAIN_PATH"])
text = path.read_text()

train_old = """                        wandb.log(
                            {
                                \"train/loss\": loss.item(),
                                \"train/trailing_loss\": trailing_loss,
                                \"train/average_loss\": avg_train_loss,
                                \"train/lr\": optimizer.param_groups[-1][\"lr\"],
                                \"train/epoch\": _epoch,
                                \"train/step\": step,
                            },
                            step=step,
                        )
"""
train_new = """                        wandb.log(
                            {
                                \"train/loss\": loss.item(),
                                \"train/avg_loss\": avg_train_loss,
                            },
                            step=step,
                        )
"""
val_old = """            wandb.log(
                {
                    \"val/loss\": avg_val_loss,
                    \"val/epoch\": _epoch,
                },
                step=((_epoch + 1) * len(train_dataloader)),
            )
"""
val_new = """            wandb.log(
                {
                    \"val/loss\": avg_val_loss,
                },
                step=((_epoch + 1) * len(train_dataloader)),
            )
"""
if train_old in text:
    text = text.replace(train_old, train_new)
if val_old in text:
    text = text.replace(val_old, val_new)
path.write_text(text)
PY
}

mkdir -p "$(dirname "${kyma_run}")" "$(dirname "${aria_run}")"
patch_aria_train

common_env=(
  WANDB_PROJECT="${WANDB_PROJECT:-kyma}"
  WANDB_ENTITY="${WANDB_ENTITY:-incado1010-iisc}"
  WANDB_RUN_GROUP="${group}"
  WANDB_MODE="${WANDB_MODE:-online}"
)

env "${common_env[@]}" \
  KYMA_WANDB=1 \
  KYMA_WANDB_STEP_INTERVAL=1 \
  WANDB_NAME="kyma-base-fullpass-${run_stamp}" \
  WANDB_TAGS="baseline,kyma,fullpass,ada" \
  TORCHINDUCTOR_COMPILE_THREADS=16 \
  OMP_NUM_THREADS=16 \
  MKL_NUM_THREADS=16 \
  CUDA_VISIBLE_DEVICES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup "${repo_root}/.venv/bin/python" -m kyma.training.pretrain train \
    kyma-base \
    --train_data "${train_pack}" \
    --val_data "${val_pack}" \
    --max_steps "${steps}" \
    --eval_every "${eval_every}" \
    --save_every "${save_every}" \
    --bs "${batch_size}" \
    --workers "${workers}" \
    --pdir "${kyma_run}" \
    --lr 1e-3 \
    --compile_backend inductor \
    > "${kyma_run}.launcher.log" 2>&1 &
echo $! > "${kyma_run}.pid"

env "${common_env[@]}" \
  WANDB_NAME="aria-medium-fullpass-${run_stamp}" \
  WANDB_GROUP="${group}" \
  WANDB_TAGS="baseline,aria,fullpass,ada" \
  CUDA_VISIBLE_DEVICES=0 \
  nohup "${aria_root}/.venv/bin/python" "${aria_root}/aria/training/train.py" train \
    medium \
    --train_data "${aria_train_dir}" \
    --val_data "${aria_val_dir}" \
    --epochs 1 \
    --max_steps "${steps}" \
    --spc "${save_every}" \
    --bs "${batch_size}" \
    --workers "${workers}" \
    --pdir "${aria_run}" \
    --no_compile \
    > "${aria_root}/aria-medium-fullpass.launcher.log" 2>&1 &
echo $! > "${aria_run}.pid"

echo "steps=${steps}"
echo "kyma_run=${kyma_run}"
echo "aria_run=${aria_run}"

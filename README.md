# kyma

`kyma` is a symbolic-music modeling codebase built around a SLinOSS state-space
backbone and trained against the Aria MIDI data contract.

The project thesis is narrow by design:

- long-form stateful training and generation
- real-time interactive continuation
- rhythm-aware and time-aware symbolic modeling

Aria is the behavioral reference point for dataset construction, tokenization,
training modes, inference surfaces, and evaluation entrypoints. The model
implementation is intentionally different.

## Layout

- `kyma/` contains the library code.
- `config/` contains dataset and model presets.
- `demo/` contains realtime and calibration tooling.
- `example-prompts/` contains prompt MIDIs for manual generation checks.
- `scripts/` contains local tooling and optional remote helpers.
- `tests/` contains compatibility, unit, and integration coverage.

## Setup

Create the local virtual environment and install the repo in editable mode:

```bash
bash scripts/bootstrap-venv.sh
```

Run the full quality gate:

```bash
make pre-commit
```

## Workflows

Kyma preserves the main Aria-style command surfaces on the PyTorch/CUDA path:

```bash
kyma generate --backend torch_cuda ...
kyma conditioned-generate --backend torch_cuda ...
kyma midi-dataset ...
kyma pack-dataset ...
```

The realtime continuation path lives in `demo/demotorch.py`.

Language-model preset:

- `kyma-base`

## Notes

- `kyma` consumes `slinoss` through its release wheel, as recommended by the
  upstream `slinoss` README.
- `kyma` intentionally targets the PyTorch/CUDA path only. It does not ship an
  MLX backend.
- Pretraining data is packed once into reusable shard sets with a `manifest.json`
  plus `shard-*.jsonl` files, then trained by step count against those stable
  shards.
- `midi-dataset` and `pack-dataset` accept optional `--workers` overrides when
  you need to cap preprocessing concurrency on shared machines.
- Training jobs automatically pick up `WANDB_*` and `KYMA_WANDB*` keys from the
  repo-local `.env` when present. On production machines, prefer `~/.netrc`
  for auth and keep the repo-local `.env` limited to non-secret defaults.
- The remote helpers are optional and remain generic operator tooling. See
  `scripts/README.md` for their usage contract.

# RTX 3060 Pilot

This branch-local workflow prepares a bounded Kyma pretraining pilot for a
single `RTX 3060` with `12 GB` VRAM. It is intentionally separate from master
because the configs and launch surface are pilot-specific.

## Goals

- exercise the full Kyma state-carry training path on real Aria-MIDI data
- cap the actual pilot launch at `24h`
- produce a go/no-go signal without pretending this is full-scale pretraining

## Defaults

- model config: [model.json](/home/b/projects/kyma/experiments/rtx3060/model.json)
- training config: [training.json](/home/b/projects/kyma/experiments/rtx3060/training.json)
- pilot cache path: `artifacts/data/rtx3060-pilot/aria-midi-pruned-r0-16000.jsonl`
- output dir: `artifacts/runs/rtx3060-pilot/`

## Prepare The Cache

Assuming the Aria-MIDI `pruned` subset is already downloaded and extracted:

```bash
./.venv/bin/python scripts/rtx3060_pilot.py build-cache --overwrite
```

That builds a shuffled, bounded piece cache for the pilot instead of trying to
materialize the full dataset into the in-memory pilot path.

## Inspect The Plan

```bash
./.venv/bin/python scripts/rtx3060_pilot.py plan --write-summary
```

The plan command resolves:

- the model and training configs
- tokenizer pad id
- deterministic train/validation split
- window counts and loss-token counts
- effective tokens per optimizer step
- output paths

## Launch Command

Do not run this until the launch is explicitly approved:

```bash
timeout 24h ./.venv/bin/python scripts/rtx3060_pilot.py train
```

## First Adjustment Ladder

If the eventual launch hits VRAM pressure, reduce in this order:

1. `chunk_size_tokens` in [model.json](/home/b/projects/kyma/experiments/rtx3060/model.json)
2. `n_layers` in [model.json](/home/b/projects/kyma/experiments/rtx3060/model.json)
3. the bounded pilot cache size via `--max-pieces`

Do not reduce `grad_accum_steps` first if the issue is memory; that changes the
effective batch more than the per-step memory footprint.

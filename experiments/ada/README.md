# Ada Pretraining

This directory contains Ada-specific preparation and launch scaffolding for the
parameter-matched Aria-vs-Kyma baseline runs.

## Scope

- Target model: `kyma-base`
- Control model: Aria `medium`
- Target machine: `ada`
- Dataset source: `loubb/aria-midi`
- Dataset subset: `pruned`

The Aria-MIDI dataset card explicitly recommends the `pruned` subset for
foundation-model pretraining, and notes that Aria itself was pretrained on it.

## Current Operating Assumptions

- Use a quarter of Ada's CPU threads for dataset prep by default so pack-time
  tokenization stays fast without saturating the machine or pounding the disk.
- One full Aria-style packed pass at `bs=8` is `69,697` optimizer steps on the
  current pruned-train shard set.

The exact optimizer-step counts should be derived from the packed train shards,
not guessed in advance.

## Tooling

The main entrypoint is [`prepare.py`](/home/b/projects/kyma/experiments/ada/prepare.py).

Typical flow on Ada:

```bash
.venv/bin/python experiments/ada/prepare.py extract
.venv/bin/python experiments/ada/prepare.py midi
.venv/bin/python experiments/ada/prepare.py pack
.venv/bin/python experiments/ada/prepare.py plan
.venv/bin/python experiments/ada/prepare.py bench --model kyma-s --gpu 1
.venv/bin/python experiments/ada/prepare.py bench --model kyma-m --gpu 1
```

`pack` builds one reusable train shard set and one reusable val shard set,
backed by `manifest.json` plus `shard-*.jsonl` files. `plan` then reports the
exact train tokens per pass and the corresponding Kyma-base target-step count.

## Notes

- `pack` delegates to Kyma's reusable shard builder and will replace any
  existing shard directory it targets.
- `bench` uses synthetic token batches with the model's real sequence length so
  memory results are relevant to production pretraining.
- On Ada, launch long pack jobs through `nice` and `ionice` so the box stays
  responsive while the shard build runs.
- `fetch` and `extract` are idempotent in the sense that they can be rerun, but
  they do not delete partially prepared data automatically.
- [`clear-runtime.sh`](/home/b/projects/kyma/experiments/ada/clear-runtime.sh)
  clears stale Kyma/Aria runtime state on Ada before a fresh baseline launch.
- [`run-fullpass-match.sh`](/home/b/projects/kyma/experiments/ada/run-fullpass-match.sh)
  launches the matched Aria `medium` and Kyma `base` runs for one full
  Aria-style packed pass.
- That launcher defaults Kyma evaluation to every `1000` steps because a full
  Ada validation sweep costs about twelve minutes of wall clock.

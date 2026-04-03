# Ada Pretraining

This directory contains Ada-specific preparation and launch scaffolding for the
first Kyma pretraining runs.

## Scope

- Target models: `kyma-s`, `kyma-m`
- Target machine: `ada`
- Dataset source: `loubb/aria-midi`
- Dataset subset: `pruned`

The Aria-MIDI dataset card explicitly recommends the `pruned` subset for
foundation-model pretraining, and notes that Aria itself was pretrained on it.

## Current Operating Assumptions

- Use GPU `1` by default. GPU `0` is currently occupied by unrelated work.
- Treat `torch.compile` as experimental on Ada. The default production setting
  for now is `--compile_backend no`.
- Use Chinchilla-style token budgets of roughly `20x` model parameters:
  - `kyma-s`: `1,652,359,040` target tokens
  - `kyma-m`: `4,823,131,040` target tokens

The exact epoch counts should be derived from the packed train shards, not
guessed in advance.

## Tooling

The main entrypoint is [`prepare.py`](/home/b/projects/kyma/experiments/ada/prepare.py).

Typical flow on Ada:

```bash
.venv/bin/python experiments/ada/prepare.py extract
.venv/bin/python experiments/ada/prepare.py midi
.venv/bin/python experiments/ada/prepare.py pack --train_epochs 1
.venv/bin/python experiments/ada/prepare.py plan
.venv/bin/python experiments/ada/prepare.py bench --model kyma-s --gpu 1
.venv/bin/python experiments/ada/prepare.py bench --model kyma-m --gpu 1
```

After the one-epoch packing pass, `plan` reports the exact tokens-per-epoch and
recommended epoch counts for `kyma-s` and `kyma-m`. Re-run `pack` with the final
`--train_epochs` once those counts are known.

## Notes

- `pack` delegates to Kyma's dataset builder and will replace any existing shard
  directory it targets.
- `bench` uses synthetic token batches with the model's real sequence length so
  memory results are relevant to production pretraining.
- `fetch` and `extract` are idempotent in the sense that they can be rerun, but
  they do not delete partially prepared data automatically.

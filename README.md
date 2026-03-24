# Kyma

Kyma is a pretrained autoregressive symbolic-music model built on the
[`SLinOSS`](https://github.com/b-vitamins/slinoss) state-space architecture.
The project is intentionally not a straight Aria port. Kyma is designed around
three differentiators:

1. Long-form stateful training and generation across contiguous musical context
2. Real-time interactive continuation with compact recurrent state
3. Rhythm-aware modeling that treats musical time as a first-class signal

The repository is being built in milestone order. The live milestone ledger is
tracked in [`TODO.md`](./TODO.md), and the running release notes live in
[`CHANGELOG.md`](./CHANGELOG.md).

Kyma consumes the published `slinoss` release artifacts rather than a source
checkout so the dependency contract stays aligned with the package release
pipeline.

Installation guidance, including the CUDA wheel path, is documented in
[`docs/install.md`](./docs/install.md).
The default developer workflow uses a repo-local `.venv` and non-editable
installs.

Workflow documentation is split by surface:

- [`docs/data-prep.md`](./docs/data-prep.md)
- [`docs/pretraining.md`](./docs/pretraining.md)
- [`docs/evaluation.md`](./docs/evaluation.md)
- [`docs/inference.md`](./docs/inference.md)

The Aria-MIDI local-cache workflow is documented in
[`docs/data-prep.md`](./docs/data-prep.md). Downloaded archives and derived
artifacts live under ignored paths such as `artifacts/data/aria-midi/`.

## Development

The local quality gate is intentionally strict:

```bash
make pre-commit
```

That runs:

- `pyright`
- `ruff check .`
- `ruff format --check .`
- `pytest`

Developer instructions and project constraints are documented in
[`AGENTS.md`](./AGENTS.md).

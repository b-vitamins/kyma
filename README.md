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

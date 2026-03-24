# AGENTS

This file is the standing operating contract for work in `kyma`.

## Mission

Build `kyma` into a technically rigorous symbolic-music research codebase with a
clear methodological edge over Aria in three areas:

1. Long-form stateful training and generation
2. Real-time interactive continuation
3. Rhythm-aware and time-aware symbolic modeling

## Non-negotiables

- Keep the history readable. Prefer small milestone commits over sprawling
  mixed commits.
- Use conventional commit messages.
- Update `CHANGELOG.md` for every milestone commit.
- Keep `TODO.md` current. Mark completed milestones explicitly.
- Do not add features that blur the project thesis. If a change does not serve
  one of the three differentiators, treat it as suspect.

## Quality Gate

Before any milestone commit, run:

```bash
make pre-commit
```

That must run the full gate:

1. `pyright`
2. `ruff check .`
3. `ruff format --check .`
4. `pytest`

If the gate fails, fix the problem before committing.

## Architecture Principles

- Favor explicit modules and narrow public surfaces over implicit magic.
- Keep runtime dependencies modest and justified.
- Use Aria as a reference point, not as a structure to copy blindly.
- Preserve a clear separation between:
  - data and tokenization
  - model definition
  - training loops
  - inference and streaming
  - evaluation
- Build for stateful execution from the start. Do not hard-code a
  transformer-style fixed-window worldview into the design.

## Documentation Rules

- Every milestone should leave the repo easier to understand than before.
- Public modules should have module docstrings.
- Prefer short, precise comments over noisy commentary.
- Keep the README high-level and current.

## Commit Conventions

Use conventional commits such as:

- `chore: establish repository standards and tooling`
- `feat(model): add SLinOSS-backed language model skeleton`
- `feat(eval): add long-horizon evaluation scaffold`
- `test(data): cover tokenizer adapter edge cases`
- `docs: document streaming evaluation protocol`

## Milestone Workflow

1. Pick the next unchecked item in `TODO.md`.
2. Implement the milestone cleanly.
3. Run `make pre-commit`.
4. Update `CHANGELOG.md`.
5. Mark the milestone as done in `TODO.md`.
6. Commit with a conventional commit message.

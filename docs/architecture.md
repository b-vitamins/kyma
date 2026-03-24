# Architecture

Kyma is organized around five explicit surfaces:

- `kyma.data`: tokenization and dataset adapters
- `kyma.model`: typed configuration and model definitions
- `kyma.training`: training loops and checkpointing
- `kyma.eval`: evaluation protocol definitions and runners
- `kyma.cli`: lightweight repo utilities and future user-facing commands

The project keeps Aria compatibility where it improves comparability, but it
does not inherit Aria's evaluation scope or transformer-centric assumptions.

## Differentiators

Kyma is obligated to remain aligned with three technical claims:

1. Long-form stateful modeling across contiguous pieces
2. Real-time interactive continuation with compact recurrent state
3. Rhythm-aware symbolic modeling with explicit time features

Those claims are encoded in both the model configuration surface and the
evaluation protocol surface so they stay visible in the codebase.

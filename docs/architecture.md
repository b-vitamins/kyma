# Architecture

Kyma is organized around five explicit surfaces:

- `kyma.data`: tokenization and dataset adapters
- `kyma.model`: typed configuration and model definitions
- `kyma.training`: training loops and checkpointing
- `kyma.eval`: evaluation protocol definitions and runners
- `kyma.cli`: lightweight repo utilities and future user-facing commands

The project keeps Aria compatibility where it improves comparability, but it
does not inherit Aria's evaluation scope or transformer-centric assumptions.

## Data Contract

The data layer works at the piece level first. Canonical encoded pieces carry:

- the original token sequence
- the encoded token ids
- a dense per-token timing feature tensor
- a per-feature validity mask

That keeps time-aware modeling explicit and lets later state-carry windowing
operate on contiguous pieces without losing the timing side channel.

The state-carry training surface slices pieces into contiguous windows with:

- explicit piece-start and piece-end markers
- carry and detach flags for recurrent-state management
- burn-in-aware loss masks
- padded time-feature tensors that stay aligned with token ids

## Differentiators

Kyma is obligated to remain aligned with three technical claims:

1. Long-form stateful modeling across contiguous pieces
2. Real-time interactive continuation with compact recurrent state
3. Rhythm-aware symbolic modeling with explicit time features

## Model Contract

The initial language model surface is a decoder-only autoregressive stack with:

- SLinOSS-compatible recurrent mixer blocks
- explicit recurrent state objects for every layer
- a dedicated time-conditioning path that projects structured timing features
  into the model width
- optional learned positional embeddings without making them the default view
  of sequence structure

That keeps the long-context and streaming story visible in the model API rather
than deferring it to later inference glue.

Those claims are encoded in both the model configuration surface and the
evaluation protocol surface so they stay visible in the codebase.

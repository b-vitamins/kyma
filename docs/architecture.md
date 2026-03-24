# Architecture

Kyma is organized around six explicit surfaces:

- `kyma.data`: tokenization and dataset adapters
- `kyma.model`: typed configuration and model definitions
- `kyma.inference`: stateful decode sessions and sampling
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

## Training Contract

The pretraining surface is built around contiguous piece streams rather than
fully shuffled fixed windows. The first training loop therefore assumes:

- stream-aligned batching that keeps active batch rows attached to a piece until
  that piece runs out of windows
- recurrent-state resets only when a slot switches pieces or becomes inactive
- truncated backpropagation through time boundaries that detach gradients
  without discarding recurrent state
- a checkpoint bundle that preserves the model weights, typed model/training
  configs, optimizer state, scheduler state, and scalar training progress

That contract keeps long-form carry behavior explicit at the training level,
which is necessary for later long-horizon and streaming evaluation.

## Inference Contract

The inference surface is session-based rather than prompt-plus-helper-only. A
decode session therefore always represents:

- recurrent state after consuming a known prefix
- logits for the next token after that prefix
- a stable boundary between prompt prefill and token-by-token advance

That keeps the realtime and future-kernel story straightforward. Faster decode
backends can replace the internals of prompt prefill and single-step advance
without changing the public sampling API or the semantics of a running session.

## Evaluation Contract

The evaluation surface is layered on purpose:

- short-context parity acts as the baseline track and keeps continuation
  comparisons grounded in prompt-conditioned generation
- later long-horizon, streaming, and rhythm tracks are expected to measure the
  actual project differentiators directly
- prompt slicing is driven by real-time token features rather than raw token
  counts so baseline evaluations already respect musical time
- long-horizon evaluation is expected to report loss by continuation horizon and
  under explicit recurrent-state reset intervals, so state carry is measured
  rather than assumed

That keeps the baseline comparable without letting it dominate the overall
methodology.

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

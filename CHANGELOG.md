# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project uses conventional
commit messages for milestone commits.

## [Unreleased]

### Added

- Initial project charter and milestone plan
- Repository standards, milestone ledger, changelog discipline, and developer operating instructions
- Packaging, linting, typing, testing, and pre-commit gate configuration
- A local git-hook installer and a repository contract test
- The initial `kyma` package layout with explicit data, model, training, evaluation, and CLI module boundaries
- Typed model and evaluation protocol configuration surfaces with packaged JSON defaults
- Basic CLI utilities for listing and printing packaged configurations
- Architecture documentation and configuration-focused tests
- Piece-level data adapters for Aria-compatible tokenization, canonical encoded piece records, and tempo-map-aware token timing features
- Focused tests for tempo maps, timing-feature extraction, and MIDI-to-piece tokenization
- A stateful Kyma language model surface with SLinOSS-compatible mixer blocks, structured time conditioning, and recurrent decode state
- Model tests covering time-feature validation, state handling, learned-position limits, and step-vs-forward consistency
- Contiguous piece windowing, typed state-carry training windows, and a flat dataset surface for TBPTT-style pretraining
- Tests covering burn-in masking, detach boundaries, truncation, and window collation
- A dedicated install guide covering the reference wheel path and the CUDA wheel path for `slinoss`
- Typed pretraining configs, a packaged small-model pretraining preset, recurrent-state-aware batch scheduling, and the first end-to-end training loop
- Checkpoint bundle helpers for model, optimizer, scheduler, and scalar training-state persistence
- Training tests covering state-carry batching, checkpoint round-trips, evaluation, and a tiny end-to-end pretraining run
- A dedicated `kyma.inference` surface with decode sessions, batched autoregressive generation, and typed sampling controls
- Inference tests covering session boundaries, greedy and filtered sampling, and stop-token-aware rectangular generation

### Changed

- Expanded the README to reflect the actual project thesis
- Scoped data-directory ignore rules to repo-root artifacts so package modules remain trackable
- Strengthened the model config schema with feedforward width, absolute-time conditioning, and validation hooks
- Switched the `slinoss` dependency to the published release wheel contract and aligned the PyTorch pin with that release surface
- Expanded the packaged-config surface and CLI to cover training presets in addition to model and evaluation configs

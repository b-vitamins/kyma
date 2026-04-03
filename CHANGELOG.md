# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Established the Kyma repository contract, local `.venv` workflow, packaging,
  linting, typing, and test gates.
- Added the flat `kyma/` package with Aria-compatible config loading, dataset
  builders, tokenization adapters, SLinOSS-backed language models, and recurrent
  inference helpers.
- Added pretraining, classifier finetuning, contrastive embedding finetuning,
  linear-probe evaluation, optional MERT and M3 evaluation hooks, and a
  PyTorch realtime continuation demo.
- Added compatibility, unit, integration, and repository-contract tests for the
  initial milestone series.
- Added first-class language-model preset names: `kyma-s`, `kyma-m`, and
  `kyma-l`.
- Added env-driven W&B observability hooks for Kyma training runs, with
  repo-local `.env` defaults and `~/.netrc` auth support.
- Added optional `torch.compile` controls to the pretraining entrypoint through
  Accelerate's TorchDynamo plugin surface.
- Stopped ignoring the entire `experiments/` tree so experiment code and docs
  can be version controlled while runtime byproducts remain ignored.
- Switched LM pretraining to a flattened token-loss path that preserves the
  objective while avoiding the broken BF16 CUDA `nll_loss2d_backward` route at
  large batch shapes.
- Fixed pretraining shard header serialization so dataset packing works with
  the slotted schema serialization on real runs.
- Replaced the epoch-indexed pretraining pack format with reusable packed shard
  manifests and switched LM pretraining control to step-based runs over stable
  shard sets.
- Added explicit worker controls to MIDI hydration and reusable shard packing so
  preprocessing concurrency can be capped cleanly on shared machines.

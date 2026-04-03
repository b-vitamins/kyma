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

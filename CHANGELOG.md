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
- Added a dedicated pretraining continuation mode that starts a new phase from
  an Accelerate checkpoint with fresh LR scheduling, ancestry metadata, and
  explicit W&B continuation tags.
- Collapsed the LM preset surface to a single `kyma-base` family, aligned the
  Kyma default LM/training surface with Aria-style init and loss behavior,
  replaced learned positions with RoPE, and fixed packed-window augmentation so
  tempo transforms only touch complete BOS/EOS-bounded sequences.
- Fixed the RoPE checkpoint wrapper so gradient-checkpointed Kyma blocks pass
  rotary state correctly during full-pass compiled training.
- Reallocated the canonical `kyma-base` preset family toward recurrent
  long-context capacity with an explicit `1024/16/14`, `ff_mult=2`,
  `d_state=256` backbone, raised the default SLinOSS mixer `r_min` to `0.8`,
  and made LM pretraining request BF16 explicitly through Accelerate.
- Added Ada-specific experiment scaffolding for dataset prep, token-budget
  planning, and pretraining calibration.
- Added Ada launch scripts for clearing stale runtime state and starting the
  matched full-pass Aria/Kyma baseline pair with aligned W&B metrics.
- Hardened the Ada full-pass launch scripts so stale Kyma processes are
  actually cleared, project directories are created lazily, and the Aria
  launcher uses the patched baseline clone's current CLI flags.
- Reset the matched Ada full-pass Kyma baseline to Aria's learning rate so the
  paired restart stays stable while preserving the compiled Kyma path.
- Relaxed the Ada matched full-pass Kyma launcher default from validating every
  `250` steps to every `1000` steps so long validation sweeps do not dominate
  wall clock.
- Aligned the Ada calibration bench with the flattened token-loss path so large
  BF16 probes measure the real training route without tripping the broken 3D CE
  backward kernel.
- Reworked Ada experiment prep to target reusable shard manifests, step-based
  run planning, and quarter-core pack concurrency on shared hardware.
- Updated the pinned SLinOSS release wheel to `v0.4.1`.
- Defaulted Kyma pretraining CLI launches to validate every `1000` steps so
  long-run evaluation stays a modest share of wall clock unless explicitly
  overridden.

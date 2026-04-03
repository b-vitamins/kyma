# TODO

- [x] Establish repository standards, dependency wiring, and the Kyma package layout.
- [x] Implement the Aria-compatible data pipeline and the SLinOSS-backed core models.
- [x] Implement inference, training, evaluation, and realtime demo workflows.
- [x] Add compatibility, unit, and integration tests and close the first milestone series.
- [x] Add optional W&B observability for training runs and secure remote defaults for production.
- [x] Add optional torch-compile controls to pretraining so long runs can be tuned per machine.
- [x] Track experiment scaffolding in git while keeping runtime outputs ignored.
- [x] Fix the pretraining LM token loss to avoid the unstable BF16 CUDA 3D CE backward path.
- [x] Fix pretraining shard header serialization so packed-dataset builds complete on real runs.
- [x] Replace epoch-based pretraining packs with reusable shard manifests and step-based training control.
- [x] Add explicit worker controls for MIDI hydration and shard packing on shared machines.
- [x] Add a dedicated pretraining continuation mode with fresh-schedule phase restarts from saved checkpoints.
- [x] Align the default Kyma LM surface with the Aria reference, adopt RoPE, and make packed-window augmentation sequence-safe.
- [x] Fix the RoPE checkpoint path so compiled full-pass Kyma runs work with the Aria-aligned block interface.
- [x] Reallocate the canonical Kyma base family toward recurrent long-context capacity, raise the default mixer `r_min` to `0.8`, and make pretraining request BF16 explicitly.
- [x] Add Ada-specific dataset and calibration scaffolding for the first `kyma-s` and `kyma-m` runs.

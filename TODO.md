# TODO

- [x] M01 Establish repository standards, tooling, changelog discipline, and the full pre-commit gate
- [x] M02 Create the package skeleton and project-wide configuration surface
- [x] M03 Add tokenizer and MIDI dataset adapters that preserve Aria comparability without inheriting Aria's layout wholesale
- [x] M04 Add the core SLinOSS-backed autoregressive language model surface for symbolic music
- [x] M05 Add state-carry pretraining dataset support for contiguous piece training and truncated backpropagation through time
- [x] M06 Add the main pretraining loop and checkpoint format for Kyma language-model training
- [x] M07 Add offline autoregressive sampling and a stateful decode loop designed to adopt future fast CuTe AR kernels
- [x] M08 Add short-context parity evaluation against Aria-style prompt continuation
- [ ] M09 Add long-horizon evaluation for contiguous context carry, horizon-conditioned loss, and long-form continuation analysis
- [ ] M10 Add streaming systems evaluation for latency, throughput, and memory growth over session length
- [ ] M11 Add rhythm-aware evaluation for onset, duration, tempo, and beat-phase consistency
- [ ] M12 Add documentation and example workflows for data preparation, pretraining, evaluation, and inference

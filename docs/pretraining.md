# Pretraining

Kyma’s current pretraining path is API-driven. The expected inputs are:

- a `KymaModelConfig`
- a `KymaPretrainConfig`
- a `KymaStateCarryDataset`

## Minimal Setup

```python
from pathlib import Path

from kyma.config import load_model_config, load_training_config
from kyma.data import KymaStateCarryDataset, KymaWindowSpec
from kyma.model import KymaAutoregressiveLM, KymaModelConfig
from kyma.training import KymaPretrainConfig, train_language_model

model_config = KymaModelConfig.from_dict(load_model_config("kyma-small"))
pretrain_config = KymaPretrainConfig.from_dict(
    load_training_config("kyma-small-pretrain")
)

window_spec = KymaWindowSpec.from_long_context_config(model_config.long_context)
dataset = KymaStateCarryDataset.from_pieces(
    pieces,
    window_spec=window_spec,
    pad_token_id=0,  # replace with your tokenizer's pad id
)

model = KymaAutoregressiveLM(model_config)
train_state = train_language_model(
    model,
    dataset,
    model_config=model_config,
    pretrain_config=pretrain_config,
    checkpoint_dir=Path("artifacts/checkpoints"),
)
print(train_state)
```

## Checkpoints

Checkpoints contain:

- model weights
- typed model config
- typed pretraining config
- optimizer state
- scheduler state
- scalar training progress

Load them with:

```python
from kyma.training import load_pretrain_checkpoint

bundle = load_pretrain_checkpoint("artifacts/checkpoints/latest.pt", model=model)
print(bundle.train_state.global_step)
```

## Notes

- The pretraining loop assumes state-carry windows, not shuffled fixed windows.
- Gradient detaches happen at the TBPTT boundaries specified by the window
  contract.
- The packaged model config currently defaults to the SLinOSS reference scan
  backend for state-carry execution on CUDA. That keeps training and evaluation
  compatible with today's stateful backend surface until stateful CuTe scan
  execution lands upstream.
- `max_steps` counts optimizer updates. Use `grad_accum_steps` to increase the
  effective batch size without forcing the microbatch to fit directly in GPU
  memory.
- `precision` can be set to `fp32`, `fp16`, or `bf16`. Mixed precision is only
  supported on CUDA devices.
- The current packaged training preset is intentionally a starting point, not a
  claim that the hyperparameters are final.

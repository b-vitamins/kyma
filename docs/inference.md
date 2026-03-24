# Inference

Kyma’s inference API is session-based. That makes the boundary between prompt
prefill and token-by-token advance explicit, which is important for future fast
decode backends.

## One-Shot Generation

```python
from kyma.inference import KymaSamplingConfig, generate

sampling_config = KymaSamplingConfig(
    max_new_tokens=256,
    temperature=0.95,
    min_p=0.03,
)
result = generate(
    model,
    prompt_ids,
    sampling_config=sampling_config,
    prompt_time_features=prompt_time_features,
    prompt_time_feature_mask=prompt_time_feature_mask,
    continuation_time_features=continuation_time_features,
    continuation_time_feature_mask=continuation_time_feature_mask,
)
print(result.generated_ids.shape)
```

`continuation_time_features` are optional. When provided, they should align with
the generated tokens that will be fed back into the model during decoding.

## Session-Based Streaming

```python
from kyma.inference import (
    KymaSamplingConfig,
    advance_decode_session,
    prefill_decode_session,
    sample_next_token,
)

session = prefill_decode_session(
    model,
    prompt_ids,
    time_features=prompt_time_features,
    time_feature_mask=prompt_time_feature_mask,
)
sampling_config = KymaSamplingConfig(max_new_tokens=1, temperature=0.0)

next_token = sample_next_token(session.next_logits, sampling_config)
session = advance_decode_session(
    model,
    session,
    next_token,
    time_features=step_time_features,
    time_feature_mask=step_time_feature_mask,
)
```

That same contract is what the streaming evaluation track benchmarks.

## Notes

- `temperature=0.0` gives greedy decoding.
- `top_p` and `min_p` are mutually exclusive in the current sampler.
- Decode sessions expose the next-token logits and the recurrent state
  separately so downstream systems can decide how to sample or schedule tokens.

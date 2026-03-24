# Evaluation

Kyma’s evaluation surface is split into four tracks:

- short-context parity
- long-horizon carry analysis
- streaming systems benchmarks
- rhythm-aware token metrics

Load the packaged protocol with:

```python
from kyma.config import load_eval_config
from kyma.eval import KymaEvalProtocol

protocol = KymaEvalProtocol.from_dict(load_eval_config("default"))
```

## Short-Context Parity

```python
from kyma.eval import evaluate_short_context_parity

report = evaluate_short_context_parity(
    model,
    pieces,
    spec=protocol.short_context_parity,
    device="cuda",
)
```

This track keeps prompt-conditioned continuation comparable to the baseline
setup while still using Kyma’s time-aware piece representation.

## Long-Horizon Carry Analysis

```python
from kyma.eval import evaluate_long_horizon

report = evaluate_long_horizon(
    model,
    pieces,
    spec=protocol.long_horizon,
    device="cuda",
)
```

This runner measures:

- continuation NLL over long real-time windows
- horizon-conditioned loss curves
- reset-interval ablations on recurrent state carry

## Streaming Systems Benchmarks

```python
from kyma.eval import evaluate_streaming

report = evaluate_streaming(
    model,
    pieces,
    spec=protocol.streaming,
    device="cuda",
)
```

This track separates:

- prompt-prefill latency
- steady token-step throughput
- decode-session memory footprint

## Rhythm-Aware Metrics

```python
from kyma.eval import evaluate_rhythm

report = evaluate_rhythm(
    model,
    pieces,
    spec=protocol.rhythm,
    device="cuda",
)
```

This runner currently reports:

- onset-token NLL
- duration-token NLL
- tempo-change consistency
- beat-phase-conditioned accuracy

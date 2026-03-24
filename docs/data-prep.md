# Data Preparation

Kyma’s data layer works on tokenized pieces with explicit time features. The
current workflow is Python-first rather than CLI-first.

## From Aria-Style JSONL

If you already have an Aria-style JSONL dataset of `MidiDict` records:

```python
from pathlib import Path

from kyma.data import get_abs_tokenizer, iter_mididict_jsonl, tokenize_midi_record

tokenizer = get_abs_tokenizer()
pieces = []
for midi_dict in iter_mididict_jsonl(Path("data/train.jsonl")):
    piece = tokenize_midi_record(midi_dict, tokenizer=tokenizer)
    pieces.append(piece)
```

Each `KymaTokenizedPiece` contains:

- token strings
- encoded token ids
- dense per-token time features
- a validity mask for those features

## From An Existing Token Sequence

If you already have an Aria-compatible token sequence:

```python
from kyma.data import make_tokenized_piece

piece = make_tokenized_piece(
    piece_id="example-piece",
    tokens=tokens,
    tokenizer=tokenizer,
    metadata={"split": "train"},
)
```

## Preparing Stateful Training Windows

State-carry training uses contiguous windows built from full pieces:

```python
from kyma.data import KymaStateCarryDataset, KymaWindowSpec

window_spec = KymaWindowSpec(
    chunk_size_tokens=1024,
    burn_in_tokens=128,
    tbptt_window_tokens=1024,
    max_piece_tokens=32768,
)
dataset = KymaStateCarryDataset.from_pieces(
    pieces,
    window_spec=window_spec,
    pad_token_id=0,  # replace with your tokenizer's pad id
)
```

Use `KymaWindowSpec.from_long_context_config(model_config.long_context)` if you
want the windowing contract to come directly from the model config.

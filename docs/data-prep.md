# Data Preparation

Kyma’s data layer works on tokenized pieces with explicit time features. The
current workflow is Python-first rather than CLI-first.

## Downloading Aria-MIDI Locally

Kyma now includes a downloader for the public Aria-MIDI release hosted at
`loubb/aria-midi` on Hugging Face. The default subset is `pruned`, which is the
recommended starting point for foundation-model pretraining.

The default local cache root is:

```text
artifacts/data/aria-midi/
```

That path is already ignored by git, so downloaded archives, manifests, and
future derived caches do not enter version control.

Dry-run the resolved plan:

```bash
python3 -m kyma.cli download-aria-midi --dry-run
```

Trigger the actual download after reviewing the upstream dataset card,
disclaimer, and `CC-BY-NC-SA 4.0` license:

```bash
python3 -m kyma.cli download-aria-midi --subset pruned --accept-license
```

The downloader stores:

- the subset tarball
- the upstream dataset README
- the upstream disclaimer
- the relevant preprocess JSON when one exists
- a local `manifest.json`

The manifest records the upstream archive size. If a download is interrupted or
otherwise truncated, Kyma will refuse to extract it and will ask you to rerun
the download with `--overwrite`.

## Extracting The Archive

The downloaded archive can be extracted into the ignored cache tree with:

```bash
python3 -m kyma.cli extract-aria-midi --subset pruned
```

By default that unpacks into:

```text
artifacts/data/aria-midi/pruned/extracted/aria-midi-v1-pruned-ext/
```

## Building A Piece Cache

Kyma can build a tokenized piece cache directly from the extracted Aria-MIDI
tree:

```bash
python3 -m kyma.cli build-aria-midi-piece-cache --subset pruned
```

Useful options for smoke runs or bounded experiments:

```bash
python3 -m kyma.cli build-aria-midi-piece-cache \
  --subset pruned \
  --limit 1024 \
  --shuffle \
  --random-seed 0 \
  --overwrite
```

The default cache path is:

```text
artifacts/data/aria-midi/pruned/piece-cache.jsonl
```

Load that cache with:

```python
from kyma.data import load_piece_cache

pieces = load_piece_cache("artifacts/data/aria-midi/pruned/piece-cache.jsonl")
```

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

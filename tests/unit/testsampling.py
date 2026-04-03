from __future__ import annotations

import torch

from kyma.inference.sampling import sampleminp, sampletopp, updateseqids


class TinyTokenizer:
    pad_tok = "<pad>"
    eos_tok = "<eos>"
    dim_tok = ("dim",)
    tok_to_id = {pad_tok: 0, eos_tok: 1, dim_tok: 2}
    id_to_tok = {
        0: pad_tok,
        1: eos_tok,
        2: dim_tok,
        3: ("note",),
        4: ("dur",),
    }


def test_sampling_helpers_return_one_id_per_batch() -> None:
    probs = torch.tensor([[0.1, 0.7, 0.2], [0.4, 0.4, 0.2]])
    assert sampletopp(probs, 0.95).shape == (2, 1)
    assert sampleminp(probs, 0.1).shape == (2, 1)


def test_updateseqids_pads_after_eos() -> None:
    tokenizer = TinyTokenizer()
    seq = torch.zeros((2, 4), dtype=torch.long)
    nexttokenids = torch.tensor([1, 3], dtype=torch.long)
    dimtokinserted = [False, False]
    eostokseen = [False, True]

    updateseqids(
        seq=seq,
        index=2,
        nexttokenids=nexttokenids,
        dimtokinserted=dimtokinserted,
        eostokseen=eostokseen,
        maxlen=4,
        forceend=False,
        tokenizer=tokenizer,  # type: ignore[arg-type]
    )

    assert seq[0, 2].item() == 1
    assert seq[1, 2].item() == 0

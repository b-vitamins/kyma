"""Embedding-head models and embedding extraction helpers."""

from __future__ import annotations

import copy

import torch
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer
from ariautils.tokenizer._base import Token
from torch import nn

from kyma.model.backbone import KymaBackbone
from kyma.model.config import ModelConfig

MAX_EMBEDDING_SEQ_LEN = 2048


class KymaEmbeddingModel(nn.Module):
    """Sequence model with an embedding head."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.emb_size is None:
            raise ValueError("ModelConfig.emb_size must be set before construction.")
        self.backbone = KymaBackbone(config)
        self.embhead = nn.Linear(config.d_model, config.emb_size, bias=False)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        return self.embhead(self.backbone(src))


def _validatemidiforembedding(mididict: MidiDict) -> None:
    present = {
        mididict.program_to_instrument[msg["data"]] for msg in mididict.instrument_msgs
    }
    if present != {"piano"}:
        raise ValueError("Only piano MIDIs are supported for embeddings.")
    if not mididict.note_msgs:
        raise ValueError("Cannot embed an empty MIDI.")


def _getchunks(mididict: MidiDict, notesperchunk: int) -> list[MidiDict]:
    chunks: list[MidiDict] = []
    for index in range(0, len(mididict.note_msgs), notesperchunk):
        notechunk = mididict.note_msgs[index : index + notesperchunk]
        if not notechunk:
            break
        chunk = copy.deepcopy(mididict)
        chunk.note_msgs = notechunk
        chunk.metadata = {}
        chunks.append(chunk)
    return chunks


@torch.inference_mode()
def getembeddingfromseq(
    model: KymaEmbeddingModel,
    seq: list[Token],
    *,
    device: str = "cuda",
) -> torch.Tensor:
    tokenizer = AbsTokenizer()
    if len(seq) > MAX_EMBEDDING_SEQ_LEN:
        raise ValueError(
            f"Sequence lengths above {MAX_EMBEDDING_SEQ_LEN} are not supported."
        )
    _validatemidiforembedding(tokenizer.detokenize(seq))
    eospos = seq.index(tokenizer.eos_tok)
    seqenc = torch.tensor(tokenizer.encode(seq), device=device).view(1, -1)
    model.eval()
    return model(seqenc)[0, eospos]


def getglobalembeddingfrommidi(
    model: KymaEmbeddingModel,
    *,
    mididict: MidiDict | None = None,
    midipath: str | None = None,
    notesperchunk: int = 300,
    device: str = "cuda",
) -> torch.Tensor:
    if mididict is None and midipath is None:
        raise ValueError("Either mididict or midipath must be provided.")
    if mididict is None:
        if midipath is None:
            raise ValueError("midipath must be provided when mididict is absent.")
        mididict = MidiDict.from_midi(mid_path=midipath)

    tokenizer = AbsTokenizer()
    _validatemidiforembedding(mididict)
    seqs = [
        tokenizer.tokenize(chunk, add_dim_tok=False)[:MAX_EMBEDDING_SEQ_LEN]
        for chunk in _getchunks(mididict, notesperchunk)
    ]
    for seq in seqs:
        if seq[-1] != tokenizer.eos_tok:
            seq[-1] = tokenizer.eos_tok
    embs = [getembeddingfromseq(model, seq, device=device) for seq in seqs]
    return torch.mean(torch.stack(embs), dim=0)

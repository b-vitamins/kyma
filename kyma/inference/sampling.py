"""Sampling helpers for Kyma generation."""

from __future__ import annotations

import warnings
from contextlib import nullcontext

import torch
from ariautils.tokenizer import AbsTokenizer, Tokenizer
from tqdm import tqdm

from kyma.inference.decode import decodeone
from kyma.inference.prefill import prefill
from kyma.model.languagemodel import KymaLM

DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)


def sampleminp(probs: torch.Tensor, pbase: float) -> torch.Tensor:
    pmax, _ = torch.max(probs, dim=-1, keepdim=True)
    scaled = pbase * pmax
    mask = probs >= scaled
    masked = probs.clone()
    masked[~mask] = 0.0
    masked.div_(masked.sum(dim=-1, keepdim=True))
    return torch.multinomial(masked, num_samples=1)


def sampletopp(probs: torch.Tensor, topp: float) -> torch.Tensor:
    probssort, probsidx = torch.sort(probs, dim=-1, descending=True)
    probssum = torch.cumsum(probssort, dim=-1)
    mask = probssum - probssort > topp
    probssort[mask] = 0.0
    probssort.div_(probssort.sum(dim=-1, keepdim=True))
    nexttoken = torch.multinomial(probssort, num_samples=1)
    return torch.gather(probsidx, -1, nexttoken)


def _autocastfor(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", dtype=DTYPE)


def _preparemodel(model: KymaLM):
    if any(parameter.is_cuda for parameter in model.parameters()):
        model.eval()
        return model, next(model.parameters()).device
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, next(model.parameters()).device


def updateseqids(
    *,
    seq: torch.Tensor,
    index: int,
    nexttokenids: torch.Tensor,
    dimtokinserted: list[bool],
    eostokseen: list[bool],
    maxlen: int,
    forceend: bool,
    tokenizer: Tokenizer,
) -> None:
    for batchindex in range(seq.shape[0]):
        if eostokseen[batchindex]:
            nexttokenids[batchindex] = tokenizer.tok_to_id[tokenizer.pad_tok]
        elif (
            forceend
            and index >= maxlen - 130
            and not dimtokinserted[batchindex]
            and tokenizer.id_to_tok[int(nexttokenids[batchindex].item())][0]
            not in ("dur", "onset")
        ):
            nexttokenids[batchindex] = tokenizer.tok_to_id[tokenizer.dim_tok]

        if (
            int(nexttokenids[batchindex].item())
            == tokenizer.tok_to_id[tokenizer.dim_tok]
        ):
            dimtokinserted[batchindex] = True
        elif (
            int(nexttokenids[batchindex].item())
            == tokenizer.tok_to_id[tokenizer.eos_tok]
        ):
            eostokseen[batchindex] = True

    seq[:, index] = nexttokenids


def _sampletokens(
    logits: torch.Tensor,
    *,
    temp: float,
    topp: float | None,
    minp: float | None,
) -> torch.Tensor:
    if temp > 0.0:
        probs = torch.softmax(logits / temp, dim=-1)
        if minp is not None:
            return sampleminp(probs, minp).flatten()
        if topp is None:
            raise ValueError("topp must be provided when minp is not set.")
        return sampletopp(probs, topp).flatten()
    return torch.argmax(logits, dim=-1).flatten()


def samplebatch(
    *,
    model: KymaLM,
    tokenizer: Tokenizer,
    prompt: list,
    numvariations: int,
    maxnewtokens: int,
    temp: float,
    forceend: bool = False,
    topp: float | None = None,
    minp: float | None = None,
    compile: bool = False,
) -> list[list]:
    if topp is None and minp is None:
        raise ValueError("Either topp or minp must be provided.")
    if forceend and maxnewtokens <= 130:
        raise ValueError("maxnewtokens must exceed 130 when forceend is enabled.")
    if compile:
        warnings.warn("torch.compile is not yet used in Kyma sampling.", stacklevel=2)

    model, device = _preparemodel(model)
    promptlen = len(prompt)
    totallen = promptlen + maxnewtokens
    dimtokinserted = [False for _ in range(numvariations)]
    eostokseen = [False for _ in range(numvariations)]
    seq = torch.stack(
        [
            torch.tensor(
                tokenizer.encode(prompt + [tokenizer.pad_tok] * (totallen - promptlen))
            )
            for _ in range(numvariations)
        ]
    ).to(device)
    state = model.initstate(
        numvariations,
        device=device,
        dtype=next(model.parameters()).dtype,
    )

    with _autocastfor(device):
        logits, state = prefill(model, seq[:, :promptlen], state=state)
        current = logits[:, -1]
        for index in tqdm(
            range(promptlen, totallen), total=totallen - promptlen, leave=False
        ):
            nexttokenids = _sampletokens(current, temp=temp, topp=topp, minp=minp)
            updateseqids(
                seq=seq,
                index=index,
                nexttokenids=nexttokenids,
                dimtokinserted=dimtokinserted,
                eostokseen=eostokseen,
                maxlen=totallen,
                forceend=forceend,
                tokenizer=tokenizer,
            )
            if all(eostokseen):
                break
            current, state = decodeone(model, seq[:, index], state=state)

    decoded = [tokenizer.decode(entry) for entry in seq.tolist()]
    return [
        decodedentry[: decodedentry.index(tokenizer.eos_tok) + 1]
        if tokenizer.eos_tok in decodedentry
        else decodedentry
        for decodedentry in decoded
    ]


def samplebatchcfg(
    *,
    model: KymaLM,
    tokenizer: AbsTokenizer,
    prompt: list,
    numvariations: int,
    maxnewtokens: int,
    cfggamma: float,
    embedding: list[float],
    temp: float,
    forceend: bool = False,
    topp: float | None = None,
    minp: float | None = None,
    compile: bool = False,
) -> list[list]:
    if model.embeddingadapter is None:
        raise ValueError("Conditioned generation requires a model with emb_size.")
    if topp is None and minp is None:
        raise ValueError("Either topp or minp must be provided.")
    if forceend and maxnewtokens <= 130:
        raise ValueError("maxnewtokens must exceed 130 when forceend is enabled.")
    if compile:
        warnings.warn(
            "torch.compile is not yet used in Kyma CFG sampling.", stacklevel=2
        )

    model, device = _preparemodel(model)
    promptlen = len(prompt)
    totallen = promptlen + maxnewtokens
    dimtokinserted = [False for _ in range(numvariations)]
    eostokseen = [False for _ in range(numvariations)]
    seq = torch.stack(
        [
            torch.tensor(
                tokenizer.encode(prompt + [tokenizer.pad_tok] * (totallen - promptlen))
            )
            for _ in range(numvariations)
        ]
    ).to(device)
    condstate = model.initstate(
        numvariations,
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    uncondstate = model.initstate(
        numvariations,
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    uncondstate.position = 1
    condembedding = torch.tensor(
        [embedding for _ in range(numvariations)],
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    condstate = model.primecondition(condembedding, condstate)

    warmupsteps = min(250, maxnewtokens)
    currstep = 0
    with _autocastfor(device):
        condlogits, condstate = prefill(model, seq[:, :promptlen], state=condstate)
        uncondlogits, uncondstate = prefill(
            model,
            seq[:, :promptlen],
            state=uncondstate,
        )
        condcurrent = condlogits[:, -1]
        uncondcurrent = uncondlogits[:, -1]
        for index in tqdm(
            range(promptlen, totallen), total=totallen - promptlen, leave=False
        ):
            currstep += 1
            gamma = min(cfggamma, (currstep / warmupsteps) * cfggamma)
            logitscfg = gamma * condcurrent + (1 - gamma) * uncondcurrent
            logitscfg[:, tokenizer.tok_to_id[tokenizer.dim_tok]] = float("-inf")
            nexttokenids = _sampletokens(logitscfg, temp=temp, topp=topp, minp=minp)
            updateseqids(
                seq=seq,
                index=index,
                nexttokenids=nexttokenids,
                dimtokinserted=dimtokinserted,
                eostokseen=eostokseen,
                maxlen=totallen,
                forceend=forceend,
                tokenizer=tokenizer,
            )
            if all(eostokseen):
                break
            condcurrent, condstate = decodeone(model, seq[:, index], state=condstate)
            uncondcurrent, uncondstate = decodeone(
                model,
                seq[:, index],
                state=uncondstate,
            )

    decoded = [tokenizer.decode(entry) for entry in seq.tolist()]
    return [
        decodedentry[: decodedentry.index(tokenizer.eos_tok) + 1]
        if tokenizer.eos_tok in decodedentry
        else decodedentry
        for decodedentry in decoded
    ]

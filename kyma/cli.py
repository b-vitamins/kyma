"""Command-line entrypoint for Kyma."""

from __future__ import annotations

import argparse
import json
import os
import sys

from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

from kyma.compat.ariacontracts import (
    DEFAULT_EMBEDDING_MODEL_PRESET,
    DEFAULT_MODEL_PRESET,
)
from kyma.compat.checkpointio import loadstate
from kyma.config.loaders import loadmodelschema
from kyma.data.mididataset import MidiDataset
from kyma.data.packeddataset import PackedDataset
from kyma.data.tokenization import gettokenizer
from kyma.inference.prompting import getinferenceprompt
from kyma.inference.sampling import samplebatch, samplebatchcfg
from kyma.model import KymaEmbeddingModel, KymaLM, getglobalembeddingfrommidi


def _parsegenerateargs():
    parser = argparse.ArgumentParser(prog="kyma generate")
    parser.add_argument("--backend", choices=["torch_cuda"], default="torch_cuda")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--prompt_midi_path", required=True)
    parser.add_argument("--prompt_duration", type=int, default=15)
    parser.add_argument("--variations", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.98)
    parser.add_argument("--min_p", type=float, default=0.035)
    parser.add_argument("--top_p", type=float, required=False)
    parser.add_argument("--end", action="store_true")
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save_dir", type=str, default=".")
    return parser.parse_args(sys.argv[2:])


def _parseconditionedgenerateargs():
    parser = argparse.ArgumentParser(prog="kyma conditioned-generate")
    parser.add_argument("--backend", choices=["torch_cuda"], default="torch_cuda")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--prompt_midi_path", required=True)
    parser.add_argument("--prompt_duration", type=int, default=15)
    parser.add_argument("--embedding_model_checkpoint_path", required=True)
    parser.add_argument("--embedding_midi_path", required=True)
    parser.add_argument("--variations", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.98)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.035)
    parser.add_argument("--top_p", type=float, required=False)
    parser.add_argument("--end", action="store_true")
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save_dir", type=str, default=".")
    return parser.parse_args(sys.argv[2:])


def _parsemididatasetargs():
    parser = argparse.ArgumentParser(prog="kyma midi-dataset")
    parser.add_argument("dir")
    parser.add_argument("save_path")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--split", type=float, required=False)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--metadata", nargs=2, metavar=("KEY", "VALUE"), action="append"
    )
    return parser.parse_args(sys.argv[2:])


def _parsepackdatasetargs():
    parser = argparse.ArgumentParser(prog="kyma pack-dataset")
    parser.add_argument("--load_path", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--tokenizer_name", choices=["abs", "rel"], required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--shard_tokens", type=int, default=33_554_432)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--sep_sequences", action="store_true")
    parser.add_argument("--embedding_dataset_path", required=False)
    return parser.parse_args(sys.argv[2:])


def _getprompt(midipath: str, *, promptduration: int):
    return getinferenceprompt(
        mididict=MidiDict.from_midi(mid_path=midipath),
        tokenizer=AbsTokenizer(),
        promptlenms=1000 * promptduration,
    )


def _loadembeddingmodel(checkpointpath: str) -> KymaEmbeddingModel:
    tokenizer = AbsTokenizer()
    config = loadmodelschema(DEFAULT_EMBEDDING_MODEL_PRESET)
    config.setvocabsize(tokenizer.vocab_size)
    model = KymaEmbeddingModel(config)
    model.load_state_dict(loadstate(checkpointpath), strict=True)
    return model


def _loadinferencemodel(
    checkpointpath: str, configname: str, *, strict: bool
) -> KymaLM:
    tokenizer = AbsTokenizer()
    config = loadmodelschema(configname)
    config.setvocabsize(tokenizer.vocab_size)
    model = KymaLM(config)
    model.load_state_dict(loadstate(checkpointpath), strict=strict)
    return model


def _getembedding(
    embeddingmodelcheckpointpath: str, embeddingmidipath: str
) -> list[float]:
    model = _loadembeddingmodel(embeddingmodelcheckpointpath).cpu()
    embedding = getglobalembeddingfrommidi(
        model=model,
        midipath=embeddingmidipath,
        device="cpu",
    )
    return embedding.tolist()


def generate(args) -> None:
    if not os.path.isdir(args.save_dir):
        raise FileNotFoundError(f"Save directory not found: {args.save_dir}")

    tokenizer = AbsTokenizer()
    prompt = _getprompt(args.prompt_midi_path, promptduration=args.prompt_duration)
    model = _loadinferencemodel(
        args.checkpoint_path, DEFAULT_MODEL_PRESET, strict=False
    )
    maxnewtokens = min(model.max_seq_len - len(prompt), args.length)
    results = samplebatch(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        numvariations=args.variations,
        maxnewtokens=maxnewtokens,
        temp=args.temp,
        forceend=args.end,
        topp=args.top_p,
        minp=args.min_p,
        compile=args.compile,
    )
    for index, tokenizedseq in enumerate(results, start=1):
        tokenizer.detokenize(tokenizedseq).to_midi().save(
            os.path.join(args.save_dir, f"res_{index}.mid")
        )


def conditionedgenerate(args) -> None:
    if not os.path.isdir(args.save_dir):
        raise FileNotFoundError(f"Save directory not found: {args.save_dir}")

    tokenizer = AbsTokenizer()
    prompt = _getprompt(args.prompt_midi_path, promptduration=args.prompt_duration)
    embedding = _getembedding(
        args.embedding_model_checkpoint_path,
        args.embedding_midi_path,
    )
    model = _loadinferencemodel(
        args.checkpoint_path, DEFAULT_EMBEDDING_MODEL_PRESET, strict=True
    )
    maxnewtokens = min(model.max_seq_len - len(prompt) - 1, args.length)
    results = samplebatchcfg(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        numvariations=args.variations,
        maxnewtokens=maxnewtokens,
        cfggamma=args.cfg,
        embedding=embedding,
        temp=args.temp,
        forceend=args.end,
        topp=args.top_p,
        minp=args.min_p,
        compile=args.compile,
    )
    for index, tokenizedseq in enumerate(results, start=1):
        tokenizer.detokenize(tokenizedseq).to_midi().save(
            os.path.join(args.save_dir, f"res_{index}.mid")
        )


def buildmididataset(args) -> None:
    metadata = {key: value for key, value in args.metadata} if args.metadata else {}
    MidiDataset.buildtofile(
        dir=args.dir,
        savepath=args.save_path,
        recur=args.recursive,
        overwrite=True,
        manualmetadata=metadata,
        shuffle=args.shuffle,
        workers=args.workers,
    )
    if args.split is not None:
        if not 0.0 < args.split < 1.0:
            raise ValueError("split must be in the open interval (0, 1).")
        MidiDataset.splitfromfile(
            args.save_path,
            trainvalratio=args.split,
            repeatable=True,
        )


def buildpackeddataset(args) -> None:
    tokenizer = gettokenizer(args.tokenizer_name)
    if args.embedding_dataset_path is not None:
        with open(args.embedding_dataset_path, encoding="utf-8") as handle:
            fileembeddings = {
                entry["metadata"]["abs_load_path"]: entry["emb"]
                for entry in map(json.loads, handle)
            }
    else:
        fileembeddings = None

    PackedDataset.build(
        tokenizer=tokenizer,
        savedir=args.save_dir,
        max_seq_len=args.seq_len,
        shard_tokens=args.shard_tokens,
        mididatasetpath=args.load_path,
        separatesequences=args.sep_sequences,
        fileembeddings=fileembeddings,
        workers=args.workers,
    )


def main() -> None:
    parser = argparse.ArgumentParser(usage="kyma <command> [<args>]")
    parser.add_argument(
        "command",
        choices=(
            "generate",
            "conditioned-generate",
            "midi-dataset",
            "pack-dataset",
        ),
    )
    args = parser.parse_args(sys.argv[1:2])
    if args.command == "generate":
        generate(_parsegenerateargs())
    elif args.command == "conditioned-generate":
        conditionedgenerate(_parseconditionedgenerateargs())
    elif args.command == "midi-dataset":
        buildmididataset(_parsemididatasetargs())
    elif args.command == "pack-dataset":
        buildpackeddataset(_parsepackdatasetargs())


if __name__ == "__main__":
    main()

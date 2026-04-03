"""Ada-specific orchestration for Kyma pretraining prep."""

from __future__ import annotations

import argparse
import json
import math
import os
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ariautils.tokenizer import AbsTokenizer

from kyma.config.loaders import loadmodelschema
from kyma.data.mididataset import MidiDataset
from kyma.data.packeddataset import PackedDataset
from kyma.data.tokenization import gettokenizer
from kyma.model import KymaLM

CHINCHILLA_TOKENS_PER_PARAM = 20
DEFAULT_SPLIT = 0.995
DEFAULT_GRAD_ACC_STEPS = 1
ADA_MICROBATCHES = {
    "kyma-s": 18,
    "kyma-m": 12,
}
ADA_TOKENS_PER_SECOND = {
    "kyma-s": 46_008,
    "kyma-m": 21_672,
}


@dataclass(frozen=True, slots=True)
class AdaPaths:
    """Filesystem defaults for Ada dataset prep."""

    dataroot: Path
    runroot: Path
    snapshotdir: Path
    extractdir: Path
    midijsonl: Path
    packroot: Path

    @property
    def trainjsonl(self) -> Path:
        return self.midijsonl.with_name(f"{self.midijsonl.stem}_train.jsonl")

    @property
    def valjsonl(self) -> Path:
        return self.midijsonl.with_name(f"{self.midijsonl.stem}_val.jsonl")

    @property
    def trainpack(self) -> Path:
        return self.packroot / "train"

    @property
    def valpack(self) -> Path:
        return self.packroot / "val"


def defaultpaths() -> AdaPaths:
    dataroot = Path(os.environ.get("KYMA_ADA_DATA_ROOT", "/data/home/ayand/datasets"))
    runroot = Path(
        os.environ.get("KYMA_ADA_RUN_ROOT", "/data/home/ayand/kyma/experiments/ada")
    )
    snapshotdir = dataroot / "aria-midi"
    extractdir = dataroot / "aria-midi-pruned"
    midijsonl = dataroot / "aria-midi-pruned.jsonl"
    packroot = dataroot / "kyma-pretrain-pruned"
    return AdaPaths(
        dataroot=dataroot,
        runroot=runroot,
        snapshotdir=snapshotdir,
        extractdir=extractdir,
        midijsonl=midijsonl,
        packroot=packroot,
    )


def _archivepath(paths: AdaPaths, subset: str) -> Path:
    return paths.snapshotdir / f"aria-midi-v1-{subset}-ext.tar.gz"


def _quarterworkers() -> int:
    return max(1, (os.cpu_count() or 1) // 4)


def _paramcount(modelname: str) -> int:
    config = loadmodelschema(modelname)
    config.setvocabsize(AbsTokenizer().vocab_size)
    model = KymaLM(config)
    return sum(param.numel() for param in model.parameters())


def _targettokens(modelname: str) -> int:
    return _paramcount(modelname) * CHINCHILLA_TOKENS_PER_PARAM


def _flatceloss(
    lossfn: torch.nn.CrossEntropyLoss,
    logits: torch.Tensor,
    tgt: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits shape (batch, seq, vocab), got {tuple(logits.shape)}."
        )
    if tgt.ndim != 2:
        raise ValueError(f"Expected tgt shape (batch, seq), got {tuple(tgt.shape)}.")
    if tuple(logits.shape[:2]) != tuple(tgt.shape):
        raise ValueError(
            "Expected logits batch/seq dims to match tgt shape, got "
            f"{tuple(logits.shape[:2])} and {tuple(tgt.shape)}."
        )
    return lossfn(logits.reshape(-1, int(logits.shape[-1])), tgt.reshape(-1))


def fetch(args) -> None:
    from huggingface_hub import snapshot_download

    paths = defaultpaths()
    paths.snapshotdir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="loubb/aria-midi",
        repo_type="dataset",
        local_dir=str(paths.snapshotdir),
    )


def extract(args) -> None:
    paths = defaultpaths()
    archivepath = _archivepath(paths, args.subset)
    if not archivepath.is_file():
        raise FileNotFoundError(f"Missing archive: {archivepath}")
    paths.extractdir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archivepath, mode="r:gz") as archive:
        archive.extractall(paths.extractdir)


def buildmidi(args) -> None:
    paths = defaultpaths()
    if not paths.extractdir.is_dir():
        raise FileNotFoundError(f"Extracted dataset not found: {paths.extractdir}")
    MidiDataset.buildtofile(
        dir=str(paths.extractdir),
        savepath=str(paths.midijsonl),
        recur=True,
        overwrite=True,
        shuffle=True,
        workers=args.workers,
    )
    MidiDataset.splitfromfile(
        paths.midijsonl,
        trainvalratio=args.split,
        repeatable=True,
        overwrite=True,
    )


def pack(args) -> None:
    paths = defaultpaths()
    tokenizer = gettokenizer(args.tokenizer)
    if not paths.trainjsonl.is_file():
        raise FileNotFoundError(f"Train JSONL not found: {paths.trainjsonl}")
    if not paths.valjsonl.is_file():
        raise FileNotFoundError(f"Val JSONL not found: {paths.valjsonl}")

    PackedDataset.build(
        tokenizer=tokenizer,
        savedir=str(paths.trainpack),
        max_seq_len=args.seq_len,
        shard_tokens=args.shard_tokens,
        mididatasetpath=str(paths.trainjsonl),
        separatesequences=args.sep_sequences,
        workers=args.workers,
    )
    PackedDataset.build(
        tokenizer=tokenizer,
        savedir=str(paths.valpack),
        max_seq_len=args.seq_len,
        shard_tokens=args.shard_tokens,
        mididatasetpath=str(paths.valjsonl),
        separatesequences=args.sep_sequences,
        workers=args.workers,
    )


def plan(args) -> None:
    paths = defaultpaths()
    summary: dict[str, Any] = {
        "models": {},
    }
    plannedseqlen = args.seq_len
    if PackedDataset.manifestpath(paths.trainpack).is_file():
        trainmanifest = PackedDataset.loadmanifest(paths.trainpack)
        valmanifest = (
            PackedDataset.loadmanifest(paths.valpack)
            if PackedDataset.manifestpath(paths.valpack).is_file()
            else None
        )
        plannedseqlen = trainmanifest.max_seq_len
        inputtokensperpass = trainmanifest.sequence_count * trainmanifest.max_seq_len
        summary["dataset"] = {
            "train_shards": len(trainmanifest.shards),
            "val_shards": len(valmanifest.shards) if valmanifest is not None else None,
            "max_seq_len": trainmanifest.max_seq_len,
            "shard_token_capacity": trainmanifest.shard_token_capacity,
            "train_sequence_count": trainmanifest.sequence_count,
            "train_input_tokens_per_pass": inputtokensperpass,
            "train_loss_tokens_per_pass": trainmanifest.loss_token_count,
            "val_sequence_count": (
                valmanifest.sequence_count if valmanifest is not None else None
            ),
            "val_loss_tokens_per_pass": (
                valmanifest.loss_token_count if valmanifest is not None else None
            ),
        }
    else:
        inputtokensperpass = None

    for modelname in args.models:
        params = _paramcount(modelname)
        targettokens = params * args.tokens_per_param
        modelsummary: dict[str, Any] = {
            "params": params,
            "target_tokens": targettokens,
        }
        microbatch = ADA_MICROBATCHES.get(modelname)
        if microbatch is not None:
            tokensperstep = args.gpus * args.grad_acc_steps * plannedseqlen * microbatch
            modelsummary.update(
                {
                    "microbatch": microbatch,
                    "grad_acc_steps": args.grad_acc_steps,
                    "tokens_per_step": tokensperstep,
                    "recommended_steps": math.ceil(targettokens / tokensperstep),
                }
            )
            tokensthroughput = ADA_TOKENS_PER_SECOND.get(modelname)
            if tokensthroughput is not None:
                effective = args.gpus * tokensthroughput
                modelsummary["estimated_hours"] = round(
                    targettokens / effective / 3600,
                    2,
                )
        if inputtokensperpass is not None:
            modelsummary["recommended_passes"] = round(
                targettokens / inputtokensperpass,
                2,
            )
        summary["models"][modelname] = modelsummary

    print(json.dumps(summary, indent=2))


def bench(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Ada benchmarking.")

    torch.cuda.set_device(args.gpu)
    if args.matmul_precision is not None:
        torch.set_float32_matmul_precision(args.matmul_precision)

    config = loadmodelschema(args.model)
    config.setvocabsize(AbsTokenizer().vocab_size)
    if config.vocab_size is None:
        raise RuntimeError("Expected vocab_size to be populated before benchmarking.")
    vocabsize = config.vocab_size
    lossfn = torch.nn.CrossEntropyLoss(ignore_index=0)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    for microbatch in args.microbatches:
        result: dict[str, Any] = {
            "model": args.model,
            "gpu": args.gpu,
            "seq_len": config.max_seq_len,
            "microbatch": microbatch,
            "dtype": args.dtype,
            "compile_backend": args.compile_backend,
        }
        try:
            model = KymaLM(config).cuda().train()
            target = model
            if args.compile_backend != "no":
                target = torch.compile(model, backend=args.compile_backend)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            times: list[float] = []
            torch.cuda.reset_peak_memory_stats()
            for step in range(args.warmup + args.steps):
                src = torch.randint(
                    0,
                    vocabsize,
                    (microbatch, config.max_seq_len),
                    device="cuda",
                )
                tgt = src.roll(-1, dims=1)
                tgt[:, -1] = 0

                started = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=dtype):
                    logits = target(src)
                    loss = _flatceloss(lossfn, logits, tgt)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if step >= args.warmup:
                    times.append(elapsed)

            peak = torch.cuda.max_memory_allocated() / (1024**3)
            result.update(
                {
                    "ok": True,
                    "peak_mem_gb": round(peak, 3),
                    "step_time_s": round(sum(times) / len(times), 3),
                    "tokens_per_step": microbatch * config.max_seq_len,
                }
            )
        except torch.OutOfMemoryError as exc:
            result.update({"ok": False, "error": "oom", "detail": str(exc)})
        except Exception as exc:  # pragma: no cover - operational path
            result.update({"ok": False, "error": repr(exc)})
        finally:
            torch.cuda.empty_cache()

        print(json.dumps(result), flush=True)


def buildparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python experiments/ada/prepare.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("fetch")

    extractparser = subparsers.add_parser("extract")
    extractparser.add_argument(
        "--subset",
        choices=("full", "pruned", "deduped", "unique"),
        default="pruned",
    )

    midiparser = subparsers.add_parser("midi")
    midiparser.add_argument("--split", type=float, default=DEFAULT_SPLIT)
    midiparser.add_argument("--workers", type=int, default=_quarterworkers())

    packparser = subparsers.add_parser("pack")
    packparser.add_argument("--tokenizer", choices=("abs", "rel"), default="abs")
    packparser.add_argument("--seq_len", type=int, default=8192)
    packparser.add_argument("--shard_tokens", type=int, default=33_554_432)
    packparser.add_argument("--workers", type=int, default=_quarterworkers())
    packparser.add_argument("--sep_sequences", action="store_true")

    planparser = subparsers.add_parser("plan")
    planparser.add_argument(
        "--models",
        nargs="+",
        default=["kyma-s", "kyma-m"],
    )
    planparser.add_argument(
        "--tokens_per_param",
        type=int,
        default=CHINCHILLA_TOKENS_PER_PARAM,
    )
    planparser.add_argument("--seq_len", type=int, default=8192)
    planparser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=DEFAULT_GRAD_ACC_STEPS,
    )
    planparser.add_argument("--gpus", type=int, default=1)

    benchparser = subparsers.add_parser("bench")
    benchparser.add_argument("--model", choices=("kyma-s", "kyma-m"), required=True)
    benchparser.add_argument("--gpu", type=int, default=0)
    benchparser.add_argument("--compile_backend", default="no")
    benchparser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    benchparser.add_argument("--warmup", type=int, default=1)
    benchparser.add_argument("--steps", type=int, default=2)
    benchparser.add_argument("--matmul_precision", choices=("high", "medium"))
    benchparser.add_argument(
        "--microbatches",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
    )

    return parser


def main() -> None:
    parser = buildparser()
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args)
    elif args.command == "extract":
        extract(args)
    elif args.command == "midi":
        buildmidi(args)
    elif args.command == "pack":
        pack(args)
    elif args.command == "plan":
        plan(args)
    else:
        bench(args)


if __name__ == "__main__":
    main()

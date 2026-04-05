"""Step-based language-model pretraining entrypoints."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import accelerate
import torch
from accelerate.logging import get_logger
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from kyma.compat.checkpointio import (
    convertaccelerate,
    loadacceleratemodelstate,
    loadstate,
)
from kyma.config.loaders import loadmodelschema
from kyma.config.schemas import PackedDatasetManifest, ProjectPaths
from kyma.data import PackedDataset, gettokenizer
from kyma.data.transforms import buildpackedaugmentations
from kyma.model import KymaLM
from kyma.training.dynamo import CompileConfig, addcompileargs
from kyma.training.engine import LossTracker, gatheredloss, lrstring
from kyma.training.optim import buildadamw, buildlinearscheduler
from kyma.training.project import createprojectlogger, createprojectpaths
from kyma.utils.validation import ensuredir
from kyma.utils.wandb import WandbRun, createwandbrun, defaultwandbname

STATE_FILENAME = "pretrain_state.json"
CONTINUATION_FILENAME = "continuation.json"
SAMPLER_SEED = 42
MIXED_PRECISION_CHOICES = ("no", "fp16", "bf16")
DEFAULT_EVAL_EVERY = 1000


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Resume metadata for the step-based pretraining loop."""

    step: int
    tokens_seen: int
    pass_index: int
    batches_processed_in_pass: int


@dataclass(frozen=True, slots=True)
class ContinuationState:
    """Metadata describing a new pretraining phase bootstrapped from a checkpoint."""

    checkpoint_dir: Path
    source_step: int
    source_tokens_seen: int


def loadmanifestpair(
    traindatapaths: list[str], valdatapath: str
) -> tuple[PackedDatasetManifest, PackedDatasetManifest]:
    trainmanifest = PackedDataset.loadmanifest(traindatapaths[0])
    valmanifest = PackedDataset.loadmanifest(valdatapath)
    if trainmanifest.tokenizer_name != valmanifest.tokenizer_name:
        raise ValueError("Training and validation datasets use different tokenizers.")
    return trainmanifest, valmanifest


def gettokenizername(traindatapaths: list[str], valdatapath: str) -> str:
    trainmanifest, _ = loadmanifestpair(traindatapaths, valdatapath)
    return trainmanifest.tokenizer_name


def getdatasets(
    *,
    traindatadirs: list[str],
    valdatadir: str,
    tokenizer,
    useembeddings: bool,
    applyaug: bool = True,
) -> tuple[PackedDataset, PackedDataset]:
    traindataset = PackedDataset(traindatadirs, tokenizer)
    valdataset = PackedDataset(valdatadir, tokenizer)
    if applyaug:
        traindataset.settransform(buildpackedaugmentations(tokenizer))

    if useembeddings:
        _, _, _, trainemb = traindataset[0]
        _, _, _, valemb = valdataset[0]
        if trainemb.numel() == 0 or valemb.numel() == 0:
            raise ValueError(
                "Embedding-conditioned training requires embedding-aware shards."
            )
    return traindataset, valdataset


def builddataloaders(
    *,
    accelerator: accelerate.Accelerator,
    traindataset: PackedDataset,
    valdataset: PackedDataset,
    batchsize: int,
    numworkers: int,
) -> tuple[DataLoader, DataLoader]:
    pinmemory = torch.cuda.is_available()
    persistentworkers = numworkers > 0
    trainsampler = DistributedSampler(
        traindataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        seed=SAMPLER_SEED,
        drop_last=False,
    )
    valsampler = DistributedSampler(
        valdataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,
        seed=SAMPLER_SEED,
        drop_last=False,
    )
    trainloader = DataLoader(
        traindataset,
        batch_size=batchsize,
        sampler=trainsampler,
        num_workers=numworkers,
        pin_memory=pinmemory,
        persistent_workers=persistentworkers,
    )
    valloader = DataLoader(
        valdataset,
        batch_size=batchsize,
        sampler=valsampler,
        num_workers=numworkers,
        pin_memory=pinmemory,
        persistent_workers=persistentworkers,
    )
    return trainloader, valloader


def buildoptim(
    *,
    model: nn.Module,
    maxsteps: int,
    lr: float = 3e-4,
    endratio: float = 0.1,
    warmupsteps: int = 200,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = buildadamw(model, lr=lr)
    scheduler = buildlinearscheduler(
        optimizer,
        totalsteps=maxsteps,
        warmupsteps=warmupsteps,
        endratio=endratio,
    )
    return optimizer, scheduler


def tokenlossmap(
    lossfn: nn.CrossEntropyLoss,
    logits: torch.Tensor,
    tgt: torch.Tensor,
) -> torch.Tensor:
    """Return per-token LM loss through the Aria-style 3D CE path."""

    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits shape (batch, T, vocab), got {tuple(logits.shape)}."
        )
    if tgt.ndim != 2:
        raise ValueError(f"Expected tgt shape (batch, T), got {tuple(tgt.shape)}.")
    if tuple(logits.shape[:2]) != tuple(tgt.shape):
        raise ValueError(
            "Expected logits batch/time dims to match tgt shape, got "
            f"{tuple(logits.shape[:2])} and {tuple(tgt.shape)}."
        )

    return lossfn(logits.transpose(1, 2), tgt)


def _gatheredint(accelerator: accelerate.Accelerator, value: int | torch.Tensor) -> int:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor([value], device=accelerator.device, dtype=torch.int64)
    else:
        value = value.to(device=accelerator.device, dtype=torch.int64).reshape(1)
    return int(accelerator.gather(value).sum().item())


def _setpass(loader: DataLoader, passindex: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(passindex)


def _movebatch(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def _statepath(checkpointdir: Path) -> Path:
    return checkpointdir / STATE_FILENAME


def _continuationpath(projectpaths: ProjectPaths) -> Path:
    return projectpaths.root / CONTINUATION_FILENAME


def _normalizeresume(
    *, passindex: int, batchesprocessed: int, batchesperpass: int
) -> tuple[int, int]:
    if batchesprocessed >= batchesperpass:
        return passindex + 1, 0
    return passindex, batchesprocessed


def saveresumestate(
    accelerator: accelerate.Accelerator,
    projectpaths: ProjectPaths,
    *,
    step: int,
    tokensseen: int,
    passindex: int,
    batchesprocessedinpass: int,
    batchesperpass: int,
) -> None:
    if not accelerator.is_main_process:
        return
    resumepass, resumebatch = _normalizeresume(
        passindex=passindex,
        batchesprocessed=batchesprocessedinpass,
        batchesperpass=batchesperpass,
    )
    checkpointdir = projectpaths.checkpoints / f"step{step}"
    accelerator.save_state(str(checkpointdir))
    payload = ResumeState(
        step=step,
        tokens_seen=tokensseen,
        pass_index=resumepass,
        batches_processed_in_pass=resumebatch,
    )
    with _statepath(checkpointdir).open("w", encoding="utf-8") as handle:
        json.dump(asdict(payload), handle, indent=2)


def loadresumestate(checkpointdir: str | Path) -> ResumeState:
    statepath = _statepath(Path(checkpointdir))
    if not statepath.is_file():
        raise FileNotFoundError(f"Missing pretraining resume metadata: {statepath}")
    with statepath.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return ResumeState(
        step=int(payload["step"]),
        tokens_seen=int(payload["tokens_seen"]),
        pass_index=int(payload["pass_index"]),
        batches_processed_in_pass=int(payload["batches_processed_in_pass"]),
    )


def loadcontinuationstate(checkpointdir: str | Path) -> ContinuationState:
    checkpointpath = Path(checkpointdir).resolve()
    resumestate = loadresumestate(checkpointpath)
    return ContinuationState(
        checkpoint_dir=checkpointpath,
        source_step=resumestate.step,
        source_tokens_seen=resumestate.tokens_seen,
    )


def savecontinuationstate(
    projectpaths: ProjectPaths, continuation: ContinuationState
) -> None:
    payload = {
        "checkpoint_dir": str(continuation.checkpoint_dir),
        "source_step": continuation.source_step,
        "source_tokens_seen": continuation.source_tokens_seen,
    }
    with _continuationpath(projectpaths).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def resolveprojectpaths(
    projectdir: str | None, *, checkpointdir: str | None = None
) -> ProjectPaths:
    if checkpointdir is None:
        return createprojectpaths(projectdir)

    if projectdir is None:
        root = Path(checkpointdir).resolve().parents[1]
    else:
        root = Path(projectdir).resolve()
        root.mkdir(parents=True, exist_ok=True)

    checkpoints = root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    return ProjectPaths(
        root=root,
        checkpoints=checkpoints,
        logs=(root / "logs.txt").resolve(),
        metrics=(root / "metrics").resolve(),
    )


def _buildmodel(modelname: str, tokenizername: str, *, useembeddings: bool) -> KymaLM:
    tokenizer = gettokenizer(tokenizername)
    config = loadmodelschema(modelname)
    config.setvocabsize(tokenizer.vocab_size)
    if useembeddings and config.emb_size is None:
        raise ValueError(
            f"Model preset {modelname!r} does not define emb_size for conditioning."
        )
    return KymaLM(config)


def _opencsv(
    filestack: ExitStack,
    path: Path,
    header: list[str],
    *,
    append: bool,
) -> tuple[object, object]:
    mode = "a" if append and path.exists() else "w"
    handle = filestack.enter_context(path.open(mode, newline="", encoding="utf-8"))
    writer = csv.writer(handle)
    if mode == "w":
        writer.writerow(header)
    return handle, writer


def _runtrain(
    *,
    accelerator: accelerate.Accelerator,
    model: KymaLM,
    trainloader: DataLoader,
    valloader: DataLoader,
    useembeddings: bool,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    projectpaths: ProjectPaths,
    wandbrun: WandbRun,
    maxsteps: int,
    evalevery: int,
    saveevery: int | None,
    resumestate: ResumeState | None,
) -> None:
    logger = get_logger(__name__)
    losstracker = LossTracker(trailingwindow=200)
    padid = trainloader.dataset.tokenizer.pad_id
    lossfn = nn.CrossEntropyLoss(ignore_index=padid, reduction="none")
    filestack = ExitStack()

    appendmode = resumestate is not None
    if accelerator.is_main_process:
        losscsv, losswriter = _opencsv(
            filestack,
            projectpaths.root / "loss.csv",
            ["pass", "batch", "step", "tokens_seen", "loss"],
            append=appendmode,
        )
        evalcsv, evalwriter = _opencsv(
            filestack,
            projectpaths.root / "eval.csv",
            ["step", "tokens_seen", "avg_train_loss", "avg_val_loss"],
            append=appendmode,
        )
    else:
        losscsv = None
        losswriter = None
        evalcsv = None
        evalwriter = None

    optimizer.zero_grad(set_to_none=True)
    globalstep = 0 if resumestate is None else resumestate.step
    tokensseen = 0 if resumestate is None else resumestate.tokens_seen
    passindex = 0 if resumestate is None else resumestate.pass_index
    startbatch = 0 if resumestate is None else resumestate.batches_processed_in_pass
    lastsavedstep = globalstep if resumestate is None else 0
    lastevalstep = globalstep if resumestate is None else 0
    lasttrainavg = 0.0

    if globalstep >= maxsteps:
        raise ValueError(
            "Checkpoint already reached or exceeded the requested max_steps."
        )

    def evaluateloop(step: int, trainavg: float, tokens: int) -> float:
        model.eval()
        totalnumerator = torch.zeros(1, device=accelerator.device)
        totaltokens = torch.zeros(1, device=accelerator.device, dtype=torch.int64)
        iterator = valloader
        if accelerator.is_main_process:
            iterator = tqdm(valloader, desc=f"Val step {step}", leave=False)
        with torch.no_grad():
            for batch in iterator:
                src, tgt, mask, emb = _movebatch(batch, accelerator.device)
                usecond = useembeddings and emb.numel() > 0 and random.random() > 0.5
                logits = model(src=src, emb=emb) if usecond else model(src)
                if usecond:
                    tgt = tgt[:, :-1]
                    mask = mask[:, :-1]

                lossmap = tokenlossmap(lossfn, logits, tgt)
                totalnumerator += (lossmap * mask).sum()
                totaltokens += mask.sum().to(dtype=torch.int64).reshape(1)

        numerator = accelerator.gather(totalnumerator).sum()
        tokencount = accelerator.gather(totaltokens).sum()
        avgloss = (
            0.0
            if int(tokencount.item()) == 0
            else float((numerator / tokencount).item())
        )
        logger.info(
            "STEP %s: validation average_loss=%.4f tokens_seen=%s",
            step,
            avgloss,
            tokens,
        )
        if accelerator.is_main_process and evalwriter is not None:
            evalwriter.writerow([step, tokens, trainavg, avgloss])
            if evalcsv is not None:
                evalcsv.flush()
        wandbrun.log(
            {
                "val/loss": avgloss,
                "train/avg_loss": trainavg,
            },
            step=step,
            force=True,
        )
        model.train()
        return avgloss

    batchesperpass = len(trainloader)
    while globalstep < maxsteps:
        _setpass(trainloader, passindex)
        iterator: DataLoader | object = trainloader
        if startbatch > 0:
            iterator = accelerator.skip_first_batches(trainloader, startbatch)
        if accelerator.is_main_process:
            iterator = tqdm(
                iterator,
                total=len(trainloader),
                initial=startbatch,
                desc=f"Pass {passindex}",
                leave=False,
            )

        batchinpass = startbatch
        pendinglosses: list[float] = []
        pendingtokens = 0
        model.train()
        for batch in iterator:
            batchinpass += 1
            src, tgt, mask, emb = _movebatch(batch, accelerator.device)
            usecond = useembeddings and emb.numel() > 0 and random.random() > 0.5
            with accelerator.accumulate(model):
                logits = model(src=src, emb=emb) if usecond else model(src)
                if usecond:
                    tgt = tgt[:, :-1]
                    mask = mask[:, :-1]

                loss = tokenlossmap(lossfn, logits, tgt)
                if mask.sum() == 0:
                    loss = (loss * 0).sum()
                else:
                    loss = (loss * mask).sum() / mask.sum()

                pendinglosses.append(gatheredloss(accelerator, loss))
                pendingtokens += _gatheredint(accelerator, mask.sum())
                accelerator.backward(loss)

                if not accelerator.sync_gradients:
                    continue

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()

                globalstep += 1
                tokensseen += pendingtokens
                steploss = sum(pendinglosses) / len(pendinglosses)
                trailing, average = losstracker.update(steploss)
                lasttrainavg = average
                logger.info(
                    (
                        "STEP %s/%s: lr=%s loss=%.4f trailing_loss=%.4f "
                        "average_loss=%.4f tokens_seen=%s pass=%s batch=%s"
                    ),
                    globalstep,
                    maxsteps,
                    lrstring(optimizer, scheduler),
                    steploss,
                    trailing,
                    average,
                    tokensseen,
                    passindex,
                    batchinpass,
                )
                if accelerator.is_main_process and losswriter is not None:
                    losswriter.writerow(
                        [passindex, batchinpass, globalstep, tokensseen, steploss]
                    )
                    if losscsv is not None:
                        losscsv.flush()

                wandbrun.log(
                    {
                        "train/loss": steploss,
                        "train/avg_loss": average,
                    },
                    step=globalstep,
                )
                pendinglosses.clear()
                pendingtokens = 0

                if saveevery is not None and globalstep % saveevery == 0:
                    saveresumestate(
                        accelerator,
                        projectpaths,
                        step=globalstep,
                        tokensseen=tokensseen,
                        passindex=passindex,
                        batchesprocessedinpass=batchinpass,
                        batchesperpass=batchesperpass,
                    )
                    lastsavedstep = globalstep

                if globalstep % evalevery == 0:
                    evaluateloop(globalstep, lasttrainavg, tokensseen)
                    lastevalstep = globalstep

                if globalstep >= maxsteps:
                    break

        if globalstep >= maxsteps:
            break
        passindex += 1
        startbatch = 0

    if lastevalstep != globalstep:
        evaluateloop(globalstep, lasttrainavg, tokensseen)
    if lastsavedstep != globalstep:
        saveresumestate(
            accelerator,
            projectpaths,
            step=globalstep,
            tokensseen=tokensseen,
            passindex=passindex,
            batchesprocessedinpass=batchinpass,
            batchesperpass=batchesperpass,
        )

    logging.shutdown()
    filestack.close()


def _runjob(
    *,
    modelname: str,
    traindatapaths: list[str],
    valdatapath: str,
    useembeddings: bool,
    numworkers: int,
    batchsize: int,
    gradaccsteps: int,
    maxsteps: int,
    evalevery: int | None,
    saveevery: int | None,
    lr: float,
    warmupsteps: int,
    endratio: float,
    mixedprecision: str,
    compileconfig: CompileConfig | None,
    checkpointpath: str | None,
    checkpointdir: str | None,
    continuation: ContinuationState | None,
    projectdir: str | None,
) -> None:
    if maxsteps <= 0 or batchsize <= 0 or gradaccsteps <= 0 or numworkers < 0:
        raise ValueError("Invalid training configuration.")
    if mixedprecision not in MIXED_PRECISION_CHOICES:
        raise ValueError(
            "mixedprecision must be one of "
            f"{', '.join(MIXED_PRECISION_CHOICES)}. Got {mixedprecision!r}."
        )
    if checkpointdir is not None and continuation is not None:
        raise ValueError(
            "Use either checkpointdir for resume or continuation, not both."
        )
    if checkpointpath is not None and continuation is not None:
        raise ValueError(
            "Continuation bootstraps from checkpointdir state, not cp_path."
        )
    for path in traindatapaths:
        ensuredir(path, label="training dataset directory")
    ensuredir(valdatapath, label="validation dataset directory")

    tokenizername = gettokenizername(traindatapaths, valdatapath)
    compileconfig = compileconfig or CompileConfig()
    accelerator = accelerate.Accelerator(
        project_dir=projectdir,
        gradient_accumulation_steps=gradaccsteps,
        mixed_precision=mixedprecision,
        dynamo_plugin=compileconfig.createplugin(),
    )
    resumestate = loadresumestate(checkpointdir) if checkpointdir is not None else None
    projectpaths = (
        resolveprojectpaths(projectdir, checkpointdir=checkpointdir)
        if accelerator.is_main_process
        else None
    )
    if projectpaths is not None:
        logger = createprojectlogger(projectpaths, name=__name__)
        logger.info("Compile config: %s", compileconfig.asdict())
        logger.info("Accelerate mixed precision: %s", mixedprecision)
        if continuation is not None:
            savecontinuationstate(projectpaths, continuation)

    model = _buildmodel(modelname, tokenizername, useembeddings=useembeddings)
    if continuation is not None:
        model.load_state_dict(
            loadacceleratemodelstate(continuation.checkpoint_dir),
            strict=False,
        )
    elif checkpointpath is not None:
        model.load_state_dict(loadstate(checkpointpath), strict=False)

    tokenizer = gettokenizer(tokenizername)
    traindataset, valdataset = getdatasets(
        traindatadirs=traindatapaths,
        valdatadir=valdatapath,
        tokenizer=tokenizer,
        useembeddings=useembeddings,
        applyaug=True,
    )
    if traindataset.max_seq_len != model.max_seq_len:
        raise ValueError("Training dataset max_seq_len does not match the model.")
    if valdataset.max_seq_len != model.max_seq_len:
        raise ValueError("Validation dataset max_seq_len does not match the model.")

    trainloader, valloader = builddataloaders(
        accelerator=accelerator,
        traindataset=traindataset,
        valdataset=valdataset,
        batchsize=batchsize,
        numworkers=numworkers,
    )
    stepsperpass = max(1, math.ceil(len(trainloader) / gradaccsteps))
    evalevery = stepsperpass if evalevery is None else evalevery
    optimizer, scheduler = buildoptim(
        model=model,
        maxsteps=maxsteps,
        lr=lr,
        endratio=endratio,
        warmupsteps=warmupsteps,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if checkpointdir is not None:
        accelerator.load_state(checkpointdir)

    projectpaths_for_run = projectpaths or ProjectPaths(
        root=Path(projectdir or "experiments"),
        checkpoints=Path(projectdir or "experiments") / "checkpoints",
        logs=Path(projectdir or "experiments") / "logs.txt",
        metrics=Path(projectdir or "experiments") / "metrics",
    )
    jobtype = (
        "pretrain-resume"
        if checkpointdir is not None
        else "pretrain-continue"
        if continuation is not None
        else "pretrain"
    )
    runprefix = (
        "pretrain-resume"
        if checkpointdir is not None
        else "pretrain-continue"
        if continuation is not None
        else "pretrain"
    )
    runtags = [
        "pretrain",
        *(["resume"] if checkpointdir is not None else []),
        *(["continue"] if continuation is not None else []),
        modelname,
    ]
    wandbrun = (
        createwandbrun(
            projectpaths=projectpaths_for_run,
            jobtype=jobtype,
            name=defaultwandbname(projectpaths_for_run, prefix=runprefix),
            group=modelname,
            tags=runtags,
            runconfig={
                "model_name": modelname,
                "train_data": traindatapaths,
                "val_data": valdatapath,
                "use_embeddings": useembeddings,
                "num_workers": numworkers,
                "batch_size_per_process": batchsize,
                "grad_acc_steps": gradaccsteps,
                "mixed_precision": mixedprecision,
                "max_steps": maxsteps,
                "steps_per_pass": stepsperpass,
                "eval_every": evalevery,
                "save_every": saveevery,
                "lr": lr,
                "warmup_steps": warmupsteps,
                "end_ratio": endratio,
                "train_sequence_count": len(traindataset),
                "train_loss_tokens_per_pass": traindataset.loss_token_count,
                "resume_from": checkpointdir,
                "continue_from": (
                    str(continuation.checkpoint_dir)
                    if continuation is not None
                    else None
                ),
                "continue_from_step": (
                    continuation.source_step if continuation is not None else None
                ),
                "continue_from_tokens_seen": (
                    continuation.source_tokens_seen
                    if continuation is not None
                    else None
                ),
                **compileconfig.asdict(),
                **asdict(model.config),
            },
        )
        if projectpaths is not None
        else WandbRun(run=None)
    )

    try:
        _runtrain(
            accelerator=accelerator,
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            useembeddings=useembeddings,
            optimizer=optimizer,
            scheduler=scheduler,
            projectpaths=projectpaths_for_run,
            wandbrun=wandbrun,
            maxsteps=maxsteps,
            evalevery=evalevery,
            saveevery=saveevery,
            resumestate=resumestate,
        )
    finally:
        traindataset.close()
        valdataset.close()
        wandbrun.finish()


def train(
    *,
    modelname: str,
    traindatapaths: list[str],
    valdatapath: str,
    useembeddings: bool,
    numworkers: int,
    batchsize: int,
    gradaccsteps: int,
    maxsteps: int,
    evalevery: int | None,
    saveevery: int | None,
    lr: float,
    warmupsteps: int,
    endratio: float,
    mixedprecision: str = "bf16",
    compileconfig: CompileConfig | None = None,
    checkpointpath: str | None = None,
    projectdir: str | None = None,
) -> None:
    _runjob(
        modelname=modelname,
        traindatapaths=traindatapaths,
        valdatapath=valdatapath,
        useembeddings=useembeddings,
        numworkers=numworkers,
        batchsize=batchsize,
        gradaccsteps=gradaccsteps,
        maxsteps=maxsteps,
        evalevery=evalevery,
        saveevery=saveevery,
        lr=lr,
        warmupsteps=warmupsteps,
        endratio=endratio,
        mixedprecision=mixedprecision,
        compileconfig=compileconfig,
        checkpointpath=checkpointpath,
        checkpointdir=None,
        continuation=None,
        projectdir=projectdir,
    )


def resumetrain(
    *,
    modelname: str,
    traindatapaths: list[str],
    valdatapath: str,
    useembeddings: bool,
    numworkers: int,
    batchsize: int,
    gradaccsteps: int,
    maxsteps: int,
    evalevery: int | None,
    saveevery: int | None,
    lr: float,
    warmupsteps: int,
    endratio: float,
    mixedprecision: str = "bf16",
    compileconfig: CompileConfig | None = None,
    checkpointdir: str,
    projectdir: str | None = None,
) -> None:
    _runjob(
        modelname=modelname,
        traindatapaths=traindatapaths,
        valdatapath=valdatapath,
        useembeddings=useembeddings,
        numworkers=numworkers,
        batchsize=batchsize,
        gradaccsteps=gradaccsteps,
        maxsteps=maxsteps,
        evalevery=evalevery,
        saveevery=saveevery,
        lr=lr,
        warmupsteps=warmupsteps,
        endratio=endratio,
        mixedprecision=mixedprecision,
        compileconfig=compileconfig,
        checkpointpath=None,
        checkpointdir=checkpointdir,
        continuation=None,
        projectdir=projectdir,
    )


def continuetrain(
    *,
    modelname: str,
    traindatapaths: list[str],
    valdatapath: str,
    useembeddings: bool,
    numworkers: int,
    batchsize: int,
    gradaccsteps: int,
    maxsteps: int,
    evalevery: int | None,
    saveevery: int | None,
    lr: float,
    warmupsteps: int,
    endratio: float,
    mixedprecision: str = "bf16",
    compileconfig: CompileConfig | None = None,
    checkpointdir: str,
    projectdir: str | None = None,
) -> None:
    _runjob(
        modelname=modelname,
        traindatapaths=traindatapaths,
        valdatapath=valdatapath,
        useembeddings=useembeddings,
        numworkers=numworkers,
        batchsize=batchsize,
        gradaccsteps=gradaccsteps,
        maxsteps=maxsteps,
        evalevery=evalevery,
        saveevery=saveevery,
        lr=lr,
        warmupsteps=warmupsteps,
        endratio=endratio,
        mixedprecision=mixedprecision,
        compileconfig=compileconfig,
        checkpointpath=None,
        checkpointdir=None,
        continuation=loadcontinuationstate(checkpointdir),
        projectdir=projectdir,
    )


def convertcpfromsafetensors(checkpointpath: str, savepath: str) -> None:
    torch.save(loadstate(checkpointpath, striporigmod=False), savepath)


def convertcpfromaccelerate(
    *,
    modelname: str,
    tokenizername: str,
    checkpointdir: str,
    savepath: str,
) -> None:
    convertaccelerate(
        lambda: _buildmodel(modelname, tokenizername, useembeddings=False),
        checkpointdir,
        savepath,
    )


def _addcommonargs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model")
    parser.add_argument("--train_data", nargs="+", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--use_embeddings", action="store_true")
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--end_ratio", type=float, default=0.1)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--mixed_precision",
        choices=MIXED_PRECISION_CHOICES,
        default="bf16",
    )
    parser.add_argument("--eval_every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--pdir")
    addcompileargs(parser)


def parseresumeargs():
    parser = argparse.ArgumentParser(prog="python -m kyma.training.pretrain resume")
    _addcommonargs(parser)
    parser.add_argument("--cp_dir", required=True)
    return parser.parse_args(sys.argv[2:])


def parsecontinueargs():
    parser = argparse.ArgumentParser(prog="python -m kyma.training.pretrain continue")
    _addcommonargs(parser)
    parser.add_argument("--cp_dir", required=True)
    return parser.parse_args(sys.argv[2:])


def parsetrainargs():
    parser = argparse.ArgumentParser(prog="python -m kyma.training.pretrain train")
    _addcommonargs(parser)
    parser.add_argument("--cp_path", default=None)
    return parser.parse_args(sys.argv[2:])


def main() -> None:
    parser = argparse.ArgumentParser(
        usage="python -m kyma.training.pretrain <command> [<args>]"
    )
    parser.add_argument("mode", choices=("train", "resume", "continue"))
    args = parser.parse_args(sys.argv[1:2])
    if args.mode == "train":
        trainargs = parsetrainargs()
        train(
            modelname=trainargs.model,
            traindatapaths=trainargs.train_data,
            valdatapath=trainargs.val_data,
            useembeddings=trainargs.use_embeddings,
            numworkers=trainargs.workers,
            batchsize=trainargs.bs,
            gradaccsteps=trainargs.grad_acc_steps,
            maxsteps=trainargs.max_steps,
            evalevery=trainargs.eval_every,
            saveevery=trainargs.save_every,
            lr=trainargs.lr,
            warmupsteps=trainargs.warmup_steps,
            endratio=trainargs.end_ratio,
            mixedprecision=trainargs.mixed_precision,
            compileconfig=CompileConfig(
                backend=trainargs.compile_backend,
                mode=trainargs.compile_mode,
                fullgraph=trainargs.compile_fullgraph,
                dynamic=trainargs.compile_dynamic,
                regional=trainargs.compile_regional,
            ),
            checkpointpath=trainargs.cp_path,
            projectdir=trainargs.pdir,
        )
    elif args.mode == "resume":
        resumeargs = parseresumeargs()
        resumetrain(
            modelname=resumeargs.model,
            traindatapaths=resumeargs.train_data,
            valdatapath=resumeargs.val_data,
            useembeddings=resumeargs.use_embeddings,
            numworkers=resumeargs.workers,
            batchsize=resumeargs.bs,
            gradaccsteps=resumeargs.grad_acc_steps,
            maxsteps=resumeargs.max_steps,
            evalevery=resumeargs.eval_every,
            saveevery=resumeargs.save_every,
            lr=resumeargs.lr,
            warmupsteps=resumeargs.warmup_steps,
            endratio=resumeargs.end_ratio,
            mixedprecision=resumeargs.mixed_precision,
            compileconfig=CompileConfig(
                backend=resumeargs.compile_backend,
                mode=resumeargs.compile_mode,
                fullgraph=resumeargs.compile_fullgraph,
                dynamic=resumeargs.compile_dynamic,
                regional=resumeargs.compile_regional,
            ),
            checkpointdir=resumeargs.cp_dir,
            projectdir=resumeargs.pdir,
        )
    else:
        continueargs = parsecontinueargs()
        continuetrain(
            modelname=continueargs.model,
            traindatapaths=continueargs.train_data,
            valdatapath=continueargs.val_data,
            useembeddings=continueargs.use_embeddings,
            numworkers=continueargs.workers,
            batchsize=continueargs.bs,
            gradaccsteps=continueargs.grad_acc_steps,
            maxsteps=continueargs.max_steps,
            evalevery=continueargs.eval_every,
            saveevery=continueargs.save_every,
            lr=continueargs.lr,
            warmupsteps=continueargs.warmup_steps,
            endratio=continueargs.end_ratio,
            mixedprecision=continueargs.mixed_precision,
            compileconfig=CompileConfig(
                backend=continueargs.compile_backend,
                mode=continueargs.compile_mode,
                fullgraph=continueargs.compile_fullgraph,
                dynamic=continueargs.compile_dynamic,
                regional=continueargs.compile_regional,
            ),
            checkpointdir=continueargs.cp_dir,
            projectdir=continueargs.pdir,
        )


if __name__ == "__main__":
    main()

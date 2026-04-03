"""Language-model pretraining entrypoints."""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from contextlib import ExitStack
from pathlib import Path

import accelerate
import torch
from accelerate.logging import get_logger
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from kyma.compat.checkpointio import convertaccelerate, loadstate
from kyma.config.loaders import loadmodelschema
from kyma.config.schemas import ProjectPaths
from kyma.data import PretrainingDataset, gettokenizer
from kyma.model import KymaLM
from kyma.training.dynamo import CompileConfig, addcompileargs
from kyma.training.engine import LossTracker, gatheredloss, lrstring, savecheckpoint
from kyma.training.optim import buildadamw, buildlinearscheduler
from kyma.training.project import createprojectlogger, createprojectpaths
from kyma.utils.validation import ensuredir
from kyma.utils.wandb import WandbRun, createwandbrun, defaultwandbname


def gettokenizername(traindatapaths: list[str], valdatapath: str) -> str:
    trainconfig = PretrainingDataset.getconfigfrompath(traindatapaths[0])
    valconfig = PretrainingDataset.getconfigfrompath(valdatapath)
    if trainconfig["tokenizer_name"] != valconfig["tokenizer_name"]:
        raise ValueError("Training and validation datasets use different tokenizers.")
    return str(trainconfig["tokenizer_name"])


def getdataloaders(
    *,
    traindatadirs: list[str],
    valdatadir: str,
    tokenizer,
    batchsize: int,
    numworkers: int,
    useembeddings: bool,
    initepoch: int | None = None,
    applyaug: bool = True,
) -> tuple[DataLoader, DataLoader]:
    traindataset = PretrainingDataset(traindatadirs, tokenizer)
    valdataset = PretrainingDataset(valdatadir, tokenizer)
    if initepoch is not None:
        traindataset.initepoch(initepoch)
    if applyaug:
        traindataset.settransform(tokenizer.export_data_aug())

    trainloader = DataLoader(
        traindataset,
        batch_size=batchsize,
        num_workers=numworkers,
        shuffle=True,
    )
    valloader = DataLoader(
        valdataset,
        batch_size=batchsize,
        num_workers=numworkers,
        shuffle=False,
    )

    if useembeddings:
        _, _, _, trainemb = traindataset[0]
        _, _, _, valemb = valdataset[0]
        if trainemb.numel() == 0 or valemb.numel() == 0:
            raise ValueError(
                "Embedding-conditioned training requires embedding datasets."
            )
    return trainloader, valloader


def buildoptim(
    *,
    model: nn.Module,
    numepochs: int,
    stepsperepoch: int,
    lr: float = 3e-4,
    endratio: float = 0.1,
    warmupsteps: int = 200,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = buildadamw(model, lr=lr)
    scheduler = buildlinearscheduler(
        optimizer,
        totalsteps=numepochs * stepsperepoch,
        warmupsteps=warmupsteps,
        endratio=endratio,
    )
    return optimizer, scheduler


def _runtrain(
    *,
    epochs: int,
    accelerator: accelerate.Accelerator,
    model: KymaLM,
    trainloader: DataLoader,
    valloader: DataLoader,
    useembeddings: bool,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    projectpaths: ProjectPaths,
    wandbrun: WandbRun,
    checkpointinterval: int | None = None,
    resumeepoch: int | None = None,
    resumestep: int | None = None,
) -> None:
    logger = get_logger(__name__)
    losstrackerwindow = 200
    padid = trainloader.dataset.tokenizer.pad_id
    lossfn = nn.CrossEntropyLoss(ignore_index=padid, reduction="none")
    filestack = ExitStack()

    if accelerator.is_main_process:
        losscsv = filestack.enter_context(
            (projectpaths.root / "loss.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
        )
        losswriter = csv.writer(losscsv)
        losswriter.writerow(["epoch", "step", "loss"])
        epochcsv = filestack.enter_context(
            (projectpaths.root / "epoch.csv").open(
                "w",
                newline="",
                encoding="utf-8",
            )
        )
        epochwriter = csv.writer(epochcsv)
        epochwriter.writerow(["epoch", "avg_train_loss", "avg_val_loss"])
    else:
        losscsv = None
        losswriter = None
        epochcsv = None
        epochwriter = None

    def trainloop(dataloader: DataLoader, epoch: int, *, resumestep: int = 0) -> float:
        tracker = LossTracker(trailingwindow=losstrackerwindow)
        loss = torch.tensor([0.0], device=accelerator.device)
        model.train()
        for stepindex, batch in (
            pbar := tqdm(
                enumerate(dataloader),
                total=len(dataloader) + resumestep,
                initial=resumestep,
                leave=False,
            )
        ):
            pbar.set_postfix_str(
                f"lr={lrstring(optimizer, scheduler)}, "
                f"loss={round(float(loss.item()), 4)}"
            )
            with accelerator.accumulate(model):
                step = stepindex + resumestep + 1
                src, tgt, mask, emb = batch
                usecond = useembeddings and emb.numel() > 0 and random.random() > 0.5
                logits = model(src=src, emb=emb) if usecond else model(src)
                if usecond:
                    tgt = tgt[:, :-1]
                    mask = mask[:, :-1]

                loss = lossfn(logits.transpose(1, 2), tgt)
                if mask.sum() == 0:
                    loss = (loss * 0).sum()
                else:
                    loss = loss * mask
                    loss = loss[loss != 0.0].mean()

                trailing, average = tracker.update(gatheredloss(accelerator, loss))
                logger.debug(
                    (
                        "EPOCH %s STEP %s: lr=%s loss=%.4f "
                        "trailing_loss=%.4f average_loss=%.4f"
                    ),
                    epoch,
                    step,
                    lrstring(optimizer, scheduler),
                    float(loss.item()),
                    trailing,
                    average,
                )
                if accelerator.is_main_process and losswriter is not None:
                    losswriter.writerow([epoch, step, float(loss.item())])
                    if losscsv is not None:
                        losscsv.flush()

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

                if checkpointinterval is not None and step % checkpointinterval == 0:
                    savecheckpoint(
                        accelerator,
                        projectpaths,
                        epoch=epoch,
                        step=step,
                    )

                globalstep = epoch * len(trainloader) + step
                wandbrun.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/trailing_loss": trailing,
                        "train/average_loss": average,
                        "train/lr": float(optimizer.param_groups[-1]["lr"]),
                        "train/epoch": epoch,
                    },
                    step=globalstep,
                )
                pbar.set_postfix_str(
                    f"lr={lrstring(optimizer, scheduler)}, "
                    f"loss={round(float(loss.item()), 4)}, "
                    f"trailing={round(trailing, 4)}"
                )
        return sum(tracker.values) / len(tracker.values)

    @torch.no_grad()
    def valloop(dataloader: DataLoader, epoch: int) -> float:
        tracker = LossTracker(trailingwindow=losstrackerwindow)
        model.eval()
        for batch in tqdm(dataloader, total=len(dataloader), leave=False):
            src, tgt, mask, emb = batch
            usecond = useembeddings and emb.numel() > 0 and random.random() > 0.5
            logits = model(src=src, emb=emb) if usecond else model(src)
            if usecond:
                tgt = tgt[:, :-1]
                mask = mask[:, :-1]
            loss = lossfn(logits.transpose(1, 2), tgt)
            if mask.sum() == 0:
                loss = (loss * 0).sum()
            else:
                loss = loss * mask
                loss = loss[loss != 0.0].mean()
            tracker.update(gatheredloss(accelerator, loss))
        average = sum(tracker.values) / len(tracker.values)
        logger.info("EPOCH %s: validation average_loss=%.4f", epoch, average)
        wandbrun.log(
            {
                "val/loss": average,
                "val/epoch": epoch,
            },
            step=(epoch + 1) * len(trainloader),
            force=True,
        )
        return average

    startepoch = 0 if resumeepoch is None else resumeepoch + 1
    if resumestep is not None and resumeepoch is not None:
        skipped = accelerator.skip_first_batches(trainloader, num_batches=resumestep)
        avgtrain = trainloop(skipped, resumeepoch, resumestep=resumestep)
        avgval = valloop(valloader, resumeepoch)
        if accelerator.is_main_process and epochwriter is not None:
            epochwriter.writerow([resumeepoch, avgtrain, avgval])
            if epochcsv is not None:
                epochcsv.flush()
        wandbrun.log(
            {
                "epoch/train_loss": avgtrain,
                "epoch/val_loss": avgval,
                "epoch/index": resumeepoch,
            },
            step=(resumeepoch + 1) * len(trainloader),
            force=True,
        )

    for epoch in range(startepoch, startepoch + epochs):
        trainloader.dataset.initepoch(epoch)
        avgtrain = trainloop(trainloader, epoch)
        avgval = valloop(valloader, epoch)
        if accelerator.is_main_process and epochwriter is not None:
            epochwriter.writerow([epoch, avgtrain, avgval])
            if epochcsv is not None:
                epochcsv.flush()
        wandbrun.log(
            {
                "epoch/train_loss": avgtrain,
                "epoch/val_loss": avgval,
                "epoch/index": epoch,
            },
            step=(epoch + 1) * len(trainloader),
            force=True,
        )
        savecheckpoint(
            accelerator,
            projectpaths,
            epoch=epoch + 1,
            step=0,
        )

    logging.shutdown()
    filestack.close()


def _buildmodel(modelname: str, tokenizername: str, *, useembeddings: bool) -> KymaLM:
    tokenizer = gettokenizer(tokenizername)
    config = loadmodelschema(modelname)
    config.setvocabsize(tokenizer.vocab_size)
    if useembeddings and config.emb_size is None:
        raise ValueError(
            f"Model preset {modelname!r} does not define emb_size for conditioning."
        )
    return KymaLM(config)


def train(
    *,
    modelname: str,
    traindatapaths: list[str],
    valdatapath: str,
    useembeddings: bool,
    numworkers: int,
    batchsize: int,
    gradaccsteps: int,
    epochs: int,
    compileconfig: CompileConfig | None = None,
    checkpointpath: str | None = None,
    checkpointinterval: int | None = None,
    projectdir: str | None = None,
) -> None:
    if epochs <= 0 or batchsize <= 0 or numworkers < 0:
        raise ValueError("Invalid training configuration.")
    for path in traindatapaths:
        ensuredir(path, label="training dataset directory")
    ensuredir(valdatapath, label="validation dataset directory")

    tokenizername = gettokenizername(traindatapaths, valdatapath)
    compileconfig = compileconfig or CompileConfig()
    accelerator = accelerate.Accelerator(
        project_dir=projectdir,
        gradient_accumulation_steps=gradaccsteps,
        dynamo_plugin=compileconfig.createplugin(),
    )
    projectpaths = (
        createprojectpaths(projectdir) if accelerator.is_main_process else None
    )
    if projectpaths is not None:
        logger = createprojectlogger(projectpaths, name=__name__)
        logger.info("Compile config: %s", compileconfig.asdict())

    model = _buildmodel(modelname, tokenizername, useembeddings=useembeddings)
    wandbrun = (
        createwandbrun(
            projectpaths=projectpaths,
            jobtype="pretrain",
            name=defaultwandbname(projectpaths, prefix="pretrain"),
            group=modelname,
            tags=["pretrain", modelname],
            runconfig={
                "model_name": modelname,
                "train_data": traindatapaths,
                "val_data": valdatapath,
                "use_embeddings": useembeddings,
                "num_workers": numworkers,
                "batch_size": batchsize,
                "grad_acc_steps": gradaccsteps,
                "epochs": epochs,
                "checkpoint_interval": checkpointinterval,
                **compileconfig.asdict(),
                **model.config.__dict__,
            },
        )
        if projectpaths is not None
        else WandbRun(run=None)
    )
    if checkpointpath is not None:
        model.load_state_dict(loadstate(checkpointpath), strict=False)

    tokenizer = gettokenizer(tokenizername)
    trainloader, valloader = getdataloaders(
        traindatadirs=traindatapaths,
        valdatadir=valdatapath,
        tokenizer=tokenizer,
        batchsize=batchsize,
        numworkers=numworkers,
        useembeddings=useembeddings,
        applyaug=True,
    )
    if trainloader.dataset.max_seq_len != model.max_seq_len:
        raise ValueError("Training dataset max_seq_len does not match the model.")
    if valloader.dataset.max_seq_len != model.max_seq_len:
        raise ValueError("Validation dataset max_seq_len does not match the model.")

    optimizer, scheduler = buildoptim(
        model=model,
        numepochs=epochs,
        stepsperepoch=max(1, len(trainloader) // max(1, gradaccsteps)),
    )
    model, trainloader, valloader, optimizer, scheduler = accelerator.prepare(
        model,
        trainloader,
        valloader,
        optimizer,
        scheduler,
    )

    projectpaths_for_run = projectpaths or ProjectPaths(
        root=Path(projectdir or "experiments"),
        checkpoints=Path(projectdir or "experiments") / "checkpoints",
        logs=Path(projectdir or "experiments") / "logs.txt",
        metrics=Path(projectdir or "experiments") / "metrics",
    )
    try:
        _runtrain(
            epochs=epochs,
            accelerator=accelerator,
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            useembeddings=useembeddings,
            optimizer=optimizer,
            scheduler=scheduler,
            projectpaths=projectpaths_for_run,
            wandbrun=wandbrun,
            checkpointinterval=checkpointinterval,
        )
    finally:
        wandbrun.finish()


def resumetrain(
    *,
    modelname: str,
    traindatapaths: list[str],
    valdatapath: str,
    useembeddings: bool,
    numworkers: int,
    batchsize: int,
    gradaccsteps: int,
    epochs: int,
    compileconfig: CompileConfig | None = None,
    checkpointdir: str,
    resumeepoch: int,
    resumestep: int,
    checkpointinterval: int | None = None,
    projectdir: str | None = None,
) -> None:
    tokenizername = gettokenizername(traindatapaths, valdatapath)
    compileconfig = compileconfig or CompileConfig()
    accelerator = accelerate.Accelerator(
        project_dir=projectdir,
        gradient_accumulation_steps=gradaccsteps,
        dynamo_plugin=compileconfig.createplugin(),
    )
    projectpaths = (
        createprojectpaths(projectdir) if accelerator.is_main_process else None
    )
    if projectpaths is not None:
        logger = createprojectlogger(projectpaths, name=__name__)
        logger.info("Compile config: %s", compileconfig.asdict())

    model = _buildmodel(modelname, tokenizername, useembeddings=useembeddings)
    wandbrun = (
        createwandbrun(
            projectpaths=projectpaths,
            jobtype="pretrain-resume",
            name=defaultwandbname(projectpaths, prefix="pretrain"),
            group=modelname,
            tags=["pretrain", "resume", modelname],
            runconfig={
                "model_name": modelname,
                "train_data": traindatapaths,
                "val_data": valdatapath,
                "use_embeddings": useembeddings,
                "num_workers": numworkers,
                "batch_size": batchsize,
                "grad_acc_steps": gradaccsteps,
                "epochs": epochs,
                "resume_epoch": resumeepoch,
                "resume_step": resumestep,
                "checkpoint_interval": checkpointinterval,
                **compileconfig.asdict(),
                **model.config.__dict__,
            },
        )
        if projectpaths is not None
        else WandbRun(run=None)
    )
    tokenizer = gettokenizer(tokenizername)
    trainloader, valloader = getdataloaders(
        traindatadirs=traindatapaths,
        valdatadir=valdatapath,
        tokenizer=tokenizer,
        batchsize=batchsize,
        numworkers=numworkers,
        useembeddings=useembeddings,
        initepoch=resumeepoch,
        applyaug=True,
    )
    optimizer, scheduler = buildoptim(
        model=model,
        numepochs=epochs,
        stepsperepoch=max(1, len(trainloader) // max(1, gradaccsteps)),
    )
    model, trainloader, valloader, optimizer, scheduler = accelerator.prepare(
        model,
        trainloader,
        valloader,
        optimizer,
        scheduler,
    )
    accelerator.load_state(checkpointdir)
    projectpaths_for_run = projectpaths or ProjectPaths(
        root=Path(projectdir or "experiments"),
        checkpoints=Path(projectdir or "experiments") / "checkpoints",
        logs=Path(projectdir or "experiments") / "logs.txt",
        metrics=Path(projectdir or "experiments") / "metrics",
    )
    try:
        _runtrain(
            epochs=epochs,
            accelerator=accelerator,
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            useembeddings=useembeddings,
            optimizer=optimizer,
            scheduler=scheduler,
            projectpaths=projectpaths_for_run,
            wandbrun=wandbrun,
            checkpointinterval=checkpointinterval,
            resumeepoch=resumeepoch,
            resumestep=resumestep,
        )
    finally:
        wandbrun.finish()


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


def parseresumeargs():
    parser = argparse.ArgumentParser(prog="python -m kyma.training.pretrain resume")
    parser.add_argument("model")
    parser.add_argument("--train_data", nargs="+", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--cp_dir", required=True)
    parser.add_argument("--use_embeddings", action="store_true")
    parser.add_argument("--r_step", type=int, required=True)
    parser.add_argument("--r_epoch", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pdir", required=False)
    parser.add_argument("--spc", type=int, required=False)
    addcompileargs(parser)
    return parser.parse_args(sys.argv[2:])


def parsetrainargs():
    parser = argparse.ArgumentParser(prog="python -m kyma.training.pretrain train")
    parser.add_argument("model")
    parser.add_argument("--train_data", nargs="+", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--cp_path", default=None)
    parser.add_argument("--use_embeddings", action="store_true")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pdir", required=False)
    parser.add_argument("--spc", type=int, required=False)
    addcompileargs(parser)
    return parser.parse_args(sys.argv[2:])


def main() -> None:
    parser = argparse.ArgumentParser(
        usage="python -m kyma.training.pretrain <command> [<args>]"
    )
    parser.add_argument("mode", choices=("train", "resume"))
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
            epochs=trainargs.epochs,
            compileconfig=CompileConfig(
                backend=trainargs.compile_backend,
                mode=trainargs.compile_mode,
                fullgraph=trainargs.compile_fullgraph,
                dynamic=trainargs.compile_dynamic,
                regional=trainargs.compile_regional,
            ),
            checkpointpath=trainargs.cp_path,
            checkpointinterval=trainargs.spc,
            projectdir=trainargs.pdir,
        )
    else:
        resumeargs = parseresumeargs()
        resumetrain(
            modelname=resumeargs.model,
            traindatapaths=resumeargs.train_data,
            valdatapath=resumeargs.val_data,
            useembeddings=resumeargs.use_embeddings,
            numworkers=resumeargs.workers,
            batchsize=resumeargs.bs,
            gradaccsteps=resumeargs.grad_acc_steps,
            epochs=resumeargs.epochs,
            compileconfig=CompileConfig(
                backend=resumeargs.compile_backend,
                mode=resumeargs.compile_mode,
                fullgraph=resumeargs.compile_fullgraph,
                dynamic=resumeargs.compile_dynamic,
                regional=resumeargs.compile_regional,
            ),
            checkpointdir=resumeargs.cp_dir,
            resumeepoch=resumeargs.r_epoch,
            resumestep=resumeargs.r_step,
            checkpointinterval=resumeargs.spc,
            projectdir=resumeargs.pdir,
        )

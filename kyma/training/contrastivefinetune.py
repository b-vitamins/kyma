"""Contrastive embedding finetuning entrypoints."""

from __future__ import annotations

import argparse
import copy
import json
import mmap
import random
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast

import accelerate
import torch
from accelerate.logging import get_logger
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kyma.compat.checkpointio import loadstate
from kyma.config.loaders import loadmodelschema
from kyma.config.schemas import ProjectPaths
from kyma.model import KymaEmbeddingModel
from kyma.training.engine import LossTracker, lrstring, savecheckpoint
from kyma.training.optim import buildadamw, buildlinearscheduler
from kyma.training.project import createprojectlogger, createprojectpaths
from kyma.utils.wandb import WandbRun, createwandbrun, defaultwandbname

DEFAULT_LR = 1e-5
DEFAULT_END_RATIO = 0.1
DEFAULT_WARMUP_STEPS = 1000
TRAILING_LOSS_STEPS = 100


def _openbinarybuffer(path: str | Path):
    return Path(path).open("rb")


class ContrastiveDataset(Dataset):
    """Dataset used for slice-based contrastive embedding finetuning."""

    def __init__(
        self,
        loadpath: str,
        *,
        minnumberslicenotes: int,
        maxnumberslicenotes: int,
        maxseqlen: int,
        applyaug: bool = False,
    ) -> None:
        self.loadpath = loadpath
        self.minnumberslicenotes = minnumberslicenotes
        self.maxnumberslicenotes = maxnumberslicenotes
        self.maxseqlen = maxseqlen
        self.applyaug = applyaug
        self.tokenizer = AbsTokenizer()
        self.augfns = self.tokenizer.export_data_aug() if applyaug else None
        self.index: list[int] = []
        self._resources = ExitStack()
        self.filebuff = None
        self.mmapobj: mmap.mmap | None = None

        self.reopen()
        if self.mmapobj is None:
            raise RuntimeError("Failed to open the contrastive dataset.")
        while True:
            position = self.mmapobj.tell()
            line = self.mmapobj.readline()
            if not line:
                break
            self.index.append(position)

    def reopen(self) -> None:
        self.close()
        self.filebuff = self._resources.enter_context(_openbinarybuffer(self.loadpath))
        self.mmapobj = mmap.mmap(self.filebuff.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        if self.mmapobj is not None:
            self.mmapobj.close()
            self.mmapobj = None
        self._resources.close()
        self._resources = ExitStack()
        self.filebuff = None

    def __del__(self) -> None:
        self.close()

    def getslice(
        self,
        mididict: MidiDict,
    ) -> tuple[list[Any], int]:
        slicedict = copy.deepcopy(mididict)
        slicelen = random.randint(
            self.minnumberslicenotes,
            self.maxnumberslicenotes,
        )
        if len(slicedict.note_msgs) <= self.minnumberslicenotes:
            startindex = 0
        else:
            startindex = random.randint(
                0,
                len(slicedict.note_msgs) - self.minnumberslicenotes,
            )

        slicedict.note_msgs = slicedict.note_msgs[startindex : startindex + slicelen]
        slicedict.metadata = {}
        tokenseq = self.tokenizer.tokenize(slicedict)

        if self.augfns is not None:
            for fn in self.augfns:
                tokenseq = list(fn(tokenseq))
            while self.tokenizer.pad_tok in tokenseq:
                tokenseq.remove(self.tokenizer.pad_tok)

        if self.tokenizer.dim_tok in tokenseq:
            tokenseq.remove(self.tokenizer.dim_tok)

        tokenseq = tokenseq[: self.maxseqlen]
        tokenseq = tokenseq + [self.tokenizer.pad_tok] * (
            self.maxseqlen - len(tokenseq)
        )
        if self.tokenizer.eos_tok not in tokenseq:
            tokenseq[-1] = self.tokenizer.eos_tok
        return tokenseq, tokenseq.index(self.tokenizer.eos_tok)

    def __getitem__(self, index: int):
        if self.mmapobj is None:
            raise RuntimeError("Dataset is not open.")
        self.mmapobj.seek(self.index[index])
        jsondata = json.loads(self.mmapobj.readline().decode("utf-8"))
        mididict = MidiDict.from_msg_dict(jsondata)
        sliceone, posone = self.getslice(mididict)
        slicetwo, postwo = self.getslice(mididict)

        return (
            torch.tensor(
                [
                    self.tokenizer.encode(sliceone),
                    self.tokenizer.encode(slicetwo),
                ]
            ),
            torch.tensor([posone, postwo]),
        )

    def __len__(self) -> int:
        return len(self.index)

    @classmethod
    def exportworkerinitfn(cls):
        def workerinitfn(_workerid: int) -> None:
            workerinfo = torch.utils.data.get_worker_info()
            if workerinfo is None:
                return
            dataset = cast(ContrastiveDataset, workerinfo.dataset)
            dataset.reopen()

        return workerinitfn


def buildoptim(
    *,
    model: nn.Module,
    numepochs: int,
    stepsperepoch: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    optimizer = buildadamw(model, lr=DEFAULT_LR)
    scheduler = buildlinearscheduler(
        optimizer,
        totalsteps=numepochs * stepsperepoch,
        warmupsteps=DEFAULT_WARMUP_STEPS,
        endratio=DEFAULT_END_RATIO,
    )
    return optimizer, scheduler


def getdataloaders(
    *,
    traindatapath: str,
    valdatapath: str,
    batchsize: int,
    numworkers: int,
    minnumberslicenotes: int = 100,
    maxnumberslicenotes: int = 650,
    maxseqlen: int = 2048,
) -> tuple[DataLoader, DataLoader]:
    trainloader = DataLoader(
        ContrastiveDataset(
            loadpath=traindatapath,
            minnumberslicenotes=minnumberslicenotes,
            maxnumberslicenotes=maxnumberslicenotes,
            maxseqlen=maxseqlen,
        ),
        batch_size=batchsize,
        shuffle=True,
        num_workers=numworkers,
        worker_init_fn=ContrastiveDataset.exportworkerinitfn(),
    )
    valloader = DataLoader(
        ContrastiveDataset(
            loadpath=valdatapath,
            minnumberslicenotes=minnumberslicenotes,
            maxnumberslicenotes=maxnumberslicenotes,
            maxseqlen=maxseqlen,
        ),
        batch_size=batchsize,
        shuffle=False,
        num_workers=numworkers,
        worker_init_fn=ContrastiveDataset.exportworkerinitfn(),
    )
    return trainloader, valloader


def symmetricntxentlosscosine(
    zone: torch.Tensor,
    ztwo: torch.Tensor,
    *,
    temperature: float = 0.5,
) -> torch.Tensor:
    batchsize = zone.shape[0]
    zone = F.normalize(zone, dim=1)
    ztwo = F.normalize(ztwo, dim=1)
    simmatrix = (
        F.cosine_similarity(zone.unsqueeze(1), ztwo.unsqueeze(0), dim=-1) / temperature
    )
    labels = torch.arange(batchsize, device=zone.device)
    return (
        F.cross_entropy(simmatrix, labels) + F.cross_entropy(simmatrix.T, labels)
    ) / 2.0


def _train(
    *,
    numepochs: int,
    accelerator: accelerate.Accelerator,
    model: KymaEmbeddingModel,
    trainloader: DataLoader,
    valloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    projectpaths: ProjectPaths | None,
    wandbrun: WandbRun,
) -> list[dict[str, float]]:
    logger = get_logger(__name__)

    def makecheckpoint(epoch: int, step: int) -> None:
        if projectpaths is not None:
            savecheckpoint(accelerator, projectpaths, epoch=epoch, step=step)

    def trainloop(dataloader: DataLoader, epoch: int) -> float:
        tracker = LossTracker(trailingwindow=TRAILING_LOSS_STEPS)
        loss = torch.tensor([0.0], device=accelerator.device)
        model.train()
        for stepindex, batch in (
            pbar := tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        ):
            pbar.set_postfix_str(
                f"lr={lrstring(optimizer, scheduler)}, "
                f"loss={round(float(loss.item()), 4)}"
            )
            with accelerator.accumulate(model):
                step = stepindex + 1
                seqs, eospos = batch
                batchsize = seqs.size(0)
                flatseqs = seqs.contiguous().view(2 * batchsize, seqs.size(-1))
                outputs = model(flatseqs)
                zonefull = outputs[0::2]
                ztwofull = outputs[1::2]
                batchindices = torch.arange(batchsize, device=zonefull.device)
                zone = zonefull[batchindices, eospos[:, 0]]
                ztwo = ztwofull[batchindices, eospos[:, 1]]
                loss = symmetricntxentlosscosine(zone, ztwo)

                trailing, average = tracker.update(
                    float(accelerator.gather(loss).mean(dim=0).item())
                )
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

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

                pbar.set_postfix_str(
                    f"lr={lrstring(optimizer, scheduler)}, "
                    f"loss={round(float(loss.item()), 4)}, "
                    f"trailing={round(trailing, 4)}"
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
        return sum(tracker.values) / len(tracker.values)

    def valloop(dataloader: DataLoader, epoch: int) -> float:
        tracker = LossTracker(trailingwindow=TRAILING_LOSS_STEPS)
        model.eval()
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}", leave=False)
            for batch in pbar:
                seqs, eospos = batch
                batchsize = seqs.size(0)
                flatseqs = seqs.contiguous().view(2 * batchsize, seqs.size(-1))
                outputs = model(flatseqs)
                zonefull = outputs[0::2]
                ztwofull = outputs[1::2]
                batchindices = torch.arange(batchsize, device=zonefull.device)
                zone = zonefull[batchindices, eospos[:, 0]]
                ztwo = ztwofull[batchindices, eospos[:, 1]]
                loss = symmetricntxentlosscosine(zone, ztwo)
                trailing, average = tracker.update(
                    float(accelerator.gather(loss).mean(dim=0).item())
                )
                pbar.set_postfix_str(
                    f"avg={round(average, 4)} trailing={round(trailing, 4)}"
                )
        avgvalloss = sum(tracker.values) / len(tracker.values)
        logger.info("Validation epoch %s: average_loss=%.4f", epoch, avgvalloss)
        wandbrun.log(
            {
                "val/loss": avgvalloss,
                "val/epoch": epoch,
            },
            step=(epoch + 1) * len(trainloader),
            force=True,
        )
        return avgvalloss

    metrics = []
    for epoch in range(numepochs):
        avgtrainloss = trainloop(trainloader, epoch)
        avgvalloss = valloop(valloader, epoch)
        metrics.append(
            {
                "avg_train_loss": avgtrainloss,
                "avg_val_loss": avgvalloss,
            }
        )
        makecheckpoint(epoch + 1, 0)
    return metrics


def train(
    *,
    modelname: str,
    traindatapath: str,
    valdatapath: str,
    numworkers: int,
    numepochs: int,
    batchsize: int,
    gradaccsteps: int,
    projectdir: str | None = None,
    checkpointpath: str | None = None,
) -> None:
    accelerator = accelerate.Accelerator(
        project_dir=projectdir,
        gradient_accumulation_steps=gradaccsteps,
    )
    projectpaths = (
        createprojectpaths(projectdir) if accelerator.is_main_process else None
    )
    logger = (
        createprojectlogger(projectpaths, name=__name__)
        if projectpaths is not None
        else get_logger(__name__)
    )
    logger.info(
        "Training config: epochs=%s batch_size=%s num_workers=%s",
        numepochs,
        batchsize,
        numworkers,
    )

    tokenizer = AbsTokenizer()
    modelconfig = loadmodelschema(modelname)
    modelconfig.setvocabsize(tokenizer.vocab_size)
    model = KymaEmbeddingModel(modelconfig)
    wandbrun = (
        createwandbrun(
            projectpaths=projectpaths,
            jobtype="contrastive-finetune",
            name=defaultwandbname(projectpaths, prefix="contrastive"),
            group=modelname,
            tags=["contrastive", modelname],
            runconfig={
                "model_name": modelname,
                "train_data": traindatapath,
                "val_data": valdatapath,
                "num_workers": numworkers,
                "num_epochs": numepochs,
                "batch_size": batchsize,
                "grad_acc_steps": gradaccsteps,
                **modelconfig.__dict__,
            },
        )
        if projectpaths is not None
        else WandbRun(run=None)
    )

    if checkpointpath is not None:
        logger.info("Loading checkpoint from %s", checkpointpath)
        model.load_state_dict(loadstate(checkpointpath), strict=False)
    else:
        logger.info("No checkpoint path provided")

    trainloader, valloader = getdataloaders(
        traindatapath=traindatapath,
        valdatapath=valdatapath,
        batchsize=batchsize,
        numworkers=numworkers,
        maxseqlen=modelconfig.max_seq_len,
    )
    optimizer, scheduler = buildoptim(
        model=model,
        numepochs=numepochs,
        stepsperepoch=len(trainloader),
    )
    model, trainloader, valloader, optimizer, scheduler = accelerator.prepare(
        model,
        trainloader,
        valloader,
        optimizer,
        scheduler,
    )
    try:
        metrics = _train(
            numepochs=numepochs,
            accelerator=accelerator,
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            optimizer=optimizer,
            scheduler=scheduler,
            projectpaths=projectpaths,
            wandbrun=wandbrun,
        )
    finally:
        wandbrun.finish()
    if projectpaths is None:
        return

    with (projectpaths.root / "results.json").open("w", encoding="utf-8") as handle:
        json.dump({"epoch_metrics": metrics}, handle, indent=2)


def parseargs():
    parser = argparse.ArgumentParser(
        description="Finetune a model for contrastive embeddings."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--project_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parseargs()
    train(
        modelname=args.model_name,
        checkpointpath=args.checkpoint_path,
        traindatapath=args.train_data_path,
        valdatapath=args.val_data_path,
        batchsize=args.batch_size,
        numepochs=args.num_epochs,
        numworkers=args.num_workers,
        gradaccsteps=args.grad_acc_steps,
        projectdir=args.project_dir,
    )

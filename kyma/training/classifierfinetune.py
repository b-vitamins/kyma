"""Classifier finetuning entrypoints."""

from __future__ import annotations

import argparse
import json
import mmap
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import accelerate
import torch
from accelerate.logging import get_logger
from ariautils.tokenizer import AbsTokenizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kyma.compat.ariacontracts import CATEGORY_TAGS
from kyma.compat.checkpointio import loadstate
from kyma.config.loaders import loadmodelschema
from kyma.config.schemas import ProjectPaths
from kyma.model import KymaClassifier
from kyma.training.engine import LossTracker, lrstring, savecheckpoint
from kyma.training.optim import buildadamw, buildlinearscheduler
from kyma.training.project import createprojectlogger, createprojectpaths
from kyma.utils.wandb import WandbRun, createwandbrun, defaultwandbname

DEFAULT_LR = 1e-5
DEFAULT_END_RATIO = 0.1
DEFAULT_WARMUP_STEPS = 0
TRAILING_LOSS_STEPS = 20


def _openbinarybuffer(path: str | Path):
    return Path(path).open("rb")


class FinetuningDataset(Dataset):
    """Dataset used for classifier finetuning and validation."""

    def __init__(
        self,
        loadpath: str,
        tagtoid: dict[str, int],
        metadatacategory: str,
        maxseqlen: int,
        *,
        perfile: bool = False,
    ) -> None:
        self.loadpath = loadpath
        self.tagtoid = tagtoid
        self.metadatacategory = metadatacategory
        self.maxseqlen = maxseqlen
        self.perfile = perfile
        self._transform = None
        self.tokenizer = AbsTokenizer()
        self.index: list[int] = []
        self._resources = ExitStack()
        self.filebuff = None
        self.mmapobj: mmap.mmap | None = None

        if metadatacategory not in CATEGORY_TAGS:
            raise ValueError(f"Unsupported metadata category: {metadatacategory}")
        if tagtoid != CATEGORY_TAGS[metadatacategory]:
            raise ValueError("tagtoid does not match the Aria category contract.")

        self.reopen()
        if self.mmapobj is None:
            raise RuntimeError("Failed to open the finetuning dataset.")
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

    def settransform(self, transform) -> None:
        if callable(transform):
            self._transform = transform
            return
        if isinstance(transform, list) and all(callable(fn) for fn in transform):

            def composed(sequence: list[Any]) -> list[Any]:
                result = sequence
                for fn in transform:
                    result = list(fn(result))
                return result

            self._transform = composed
            return
        raise ValueError("transform must be a callable or a list of callables.")

    def __getitem__(self, index: int):
        def formattoken(token):
            return tuple(token) if isinstance(token, list) else token

        if self.mmapobj is None:
            raise RuntimeError("Dataset is not open.")

        self.mmapobj.seek(self.index[index])
        jsondata = json.loads(self.mmapobj.readline().decode("utf-8"))

        metadata = jsondata["metadata"]
        tag = metadata[self.metadatacategory]
        if tag not in self.tagtoid:
            raise ValueError(f"Unexpected tag {tag!r} for metadata {metadata!r}.")
        tagtensor = torch.tensor(self.tagtoid[tag])

        seqlist = jsondata["seqs"] if self.perfile else [jsondata["seq"]]
        seqtensors = []
        postensors = []
        for seq in seqlist:
            tokenseq = [formattoken(tok) for tok in seq]
            if self._transform is not None:
                tokenseq = list(self._transform(tokenseq))

            tokenseq = tokenseq[: self.maxseqlen]
            if self.tokenizer.eos_tok not in tokenseq:
                if not tokenseq:
                    raise ValueError("Encountered an empty token sequence.")
                tokenseq[-1] = self.tokenizer.eos_tok

            eosindex = tokenseq.index(self.tokenizer.eos_tok)
            tokenseq = tokenseq + [self.tokenizer.pad_tok] * (
                self.maxseqlen - len(tokenseq)
            )
            encoded = self.tokenizer.encode(tokenseq)
            seqtensor = torch.tensor(encoded)
            postensor = torch.tensor(eosindex)
            seqtensors.append(seqtensor)
            postensors.append(postensor)

        return torch.stack(seqtensors), torch.stack(postensors), tagtensor

    def __len__(self) -> int:
        return len(self.index)

    @classmethod
    def exportworkerinitfn(cls):
        def workerinitfn(_workerid: int) -> None:
            workerinfo = torch.utils.data.get_worker_info()
            if workerinfo is None:
                return
            dataset = cast(FinetuningDataset, workerinfo.dataset)
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
    metadatacategory: str,
    tagtoid: dict[str, int],
    batchsize: int,
    numworkers: int,
    applyaug: bool = False,
    maxseqlen: int = 1024,
) -> tuple[DataLoader, DataLoader]:
    traindataset = FinetuningDataset(
        loadpath=traindatapath,
        tagtoid=tagtoid,
        metadatacategory=metadatacategory,
        maxseqlen=maxseqlen,
    )
    valdataset = FinetuningDataset(
        loadpath=valdatapath,
        tagtoid=tagtoid,
        metadatacategory=metadatacategory,
        maxseqlen=maxseqlen,
        perfile=True,
    )
    if applyaug:
        traindataset.settransform(AbsTokenizer().export_data_aug())

    trainloader = DataLoader(
        traindataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=numworkers,
        worker_init_fn=FinetuningDataset.exportworkerinitfn(),
    )
    valloader = DataLoader(
        valdataset,
        batch_size=1,
        shuffle=False,
        num_workers=numworkers,
        worker_init_fn=FinetuningDataset.exportworkerinitfn(),
    )
    return trainloader, valloader


def _train(
    *,
    numepochs: int,
    accelerator: accelerate.Accelerator,
    model: KymaClassifier,
    trainloader: DataLoader,
    valloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    tagtoid: dict[str, int],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    projectpaths: ProjectPaths | None,
    wandbrun: WandbRun,
) -> list[dict[str, Any]]:
    logger = get_logger(__name__)
    lossfn = nn.CrossEntropyLoss()

    def makecheckpoint(epoch: int, step: int) -> None:
        if projectpaths is not None:
            savecheckpoint(
                accelerator,
                projectpaths,
                epoch=epoch,
                step=step,
            )

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
                seqs, eospos, labels = batch
                seqs = seqs.squeeze(1)
                eospos = eospos.squeeze(1)

                logits = model(seqs)
                logits = logits[
                    torch.arange(logits.shape[0], device=logits.device), eospos
                ]
                loss = lossfn(logits, labels)

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

    def valloop(dataloader: DataLoader, epoch: int) -> dict[str, Any]:
        model.eval()
        padid = AbsTokenizer().pad_id
        preds: list[int] = []
        labels: list[int] = []

        with torch.inference_mode():
            pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}", leave=False)
            for batch in pbar:
                seqs, positions, tags = batch
                seqs = seqs.squeeze(0)
                positions = positions.squeeze(0)
                logits = model(seqs)
                logits = logits[
                    torch.arange(logits.shape[0], device=logits.device),
                    positions,
                ]
                probs = torch.softmax(logits, dim=-1)
                nonpadcounts = (seqs != padid).sum(dim=1, keepdim=True).float()
                aggregated = (probs * nonpadcounts).sum(dim=0)
                preds.append(int(aggregated.argmax().item()))
                labels.append(int(tags.item()))
                tmpacc = sum(
                    pred == truth for pred, truth in zip(preds, labels, strict=False)
                ) / len(preds)
                pbar.set_postfix_str(f"acc={round(tmpacc, 4)}")

        accuracy = sum(
            pred == truth for pred, truth in zip(preds, labels, strict=False)
        ) / len(labels)
        idtotag = {value: key for key, value in tagtoid.items()}
        metrics = {tag: {"TP": 0, "FP": 0, "FN": 0} for tag in tagtoid}
        for trueid, predid in zip(labels, preds, strict=False):
            truetag = idtotag[trueid]
            predtag = idtotag[predid]
            if trueid == predid:
                metrics[truetag]["TP"] += 1
            else:
                metrics[truetag]["FN"] += 1
                metrics[predtag]["FP"] += 1

        classmetrics = {}
        f1scores = []
        for tag, counts in metrics.items():
            truepos = counts["TP"]
            falsepos = counts["FP"]
            falseneg = counts["FN"]
            precision = (
                truepos / (truepos + falsepos) if (truepos + falsepos) > 0 else 0.0
            )
            recall = truepos / (truepos + falseneg) if (truepos + falseneg) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            classmetrics[tag] = {
                "precision": precision,
                "recall": recall,
                "F1": f1,
            }
            f1scores.append(f1)

        macrof1 = sum(f1scores) / len(f1scores) if f1scores else 0.0
        logger.info(
            "Validation epoch %s: accuracy=%.4f macro_f1=%.4f",
            epoch,
            accuracy,
            macrof1,
        )
        logger.info("Class metrics: %s", classmetrics)
        wandbrun.log(
            {
                "val/accuracy": accuracy,
                "val/macro_f1": macrof1,
                "val/epoch": epoch,
            },
            step=(epoch + 1) * len(trainloader),
            force=True,
        )
        return {
            "accuracy": accuracy,
            "macro_f1": macrof1,
            "class_metrics": classmetrics,
        }

    epochmetrics = []
    for epoch in range(numepochs):
        avgtrainloss = trainloop(trainloader, epoch)
        metrics = valloop(valloader, epoch)
        metrics["avg_train_loss"] = avgtrainloss
        epochmetrics.append(metrics)
        makecheckpoint(epoch + 1, 0)

    return epochmetrics


def train(
    *,
    modelname: str,
    metadatacategory: str,
    applyaug: bool,
    traindatapath: str,
    valdatapath: str,
    numworkers: int,
    numepochs: int,
    batchsize: int,
    gradaccsteps: int,
    projectdir: str | None = None,
    checkpointpath: str | None = None,
    datasetsize: int | None = None,
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

    tagtoid = CATEGORY_TAGS[metadatacategory]
    logger.info("Metadata category: %s", metadatacategory)
    logger.info("Dataset size: %s", datasetsize)
    logger.info("Apply augmentation: %s", applyaug)
    logger.info(
        "Training config: epochs=%s batch_size=%s num_workers=%s",
        numepochs,
        batchsize,
        numworkers,
    )

    tokenizer = AbsTokenizer()
    modelconfig = loadmodelschema(modelname)
    modelconfig.setvocabsize(tokenizer.vocab_size)
    if modelconfig.class_size != len(tagtoid):
        raise ValueError(
            "modelconfig.class_size does not match the requested category contract."
        )
    model = KymaClassifier(modelconfig)
    wandbrun = (
        createwandbrun(
            projectpaths=projectpaths,
            jobtype="classifier-finetune",
            name=defaultwandbname(projectpaths, prefix="classifier"),
            group=metadatacategory,
            tags=["classifier", metadatacategory, modelname],
            runconfig={
                "model_name": modelname,
                "metadata_category": metadatacategory,
                "dataset_size": datasetsize,
                "apply_aug": applyaug,
                "train_data": traindatapath,
                "val_data": valdatapath,
                "num_workers": numworkers,
                "num_epochs": numepochs,
                "batch_size": batchsize,
                "grad_acc_steps": gradaccsteps,
                **asdict(modelconfig),
            },
        )
        if projectpaths is not None
        else WandbRun(run=None)
    )

    if checkpointpath is not None:
        logger.info("Loading checkpoint from %s", checkpointpath)
        model.load_state_dict(loadstate(checkpointpath), strict=False)
        torch.nn.init.normal_(model.backbone.tokenembed.weight.data[1:2], std=0.02)
    else:
        logger.info("No checkpoint path provided")

    trainloader, valloader = getdataloaders(
        traindatapath=traindatapath,
        valdatapath=valdatapath,
        metadatacategory=metadatacategory,
        tagtoid=tagtoid,
        batchsize=batchsize,
        numworkers=numworkers,
        applyaug=applyaug,
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
        epochmetrics = _train(
            numepochs=numepochs,
            accelerator=accelerator,
            model=model,
            trainloader=trainloader,
            valloader=valloader,
            optimizer=optimizer,
            tagtoid=tagtoid,
            scheduler=scheduler,
            projectpaths=projectpaths,
            wandbrun=wandbrun,
        )
    finally:
        wandbrun.finish()
    if projectpaths is None:
        return

    maxaccuracy = (
        max(metric["accuracy"] for metric in epochmetrics) if epochmetrics else 0.0
    )
    logger.info("Max accuracy: %.4f", maxaccuracy)
    results = {
        "metadata_category": metadatacategory,
        "dataset_size": datasetsize,
        "epoch_metrics": epochmetrics,
        "max_accuracy": maxaccuracy,
    }
    with (projectpaths.root / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def parseargs():
    parser = argparse.ArgumentParser(description="Finetune a model for classification.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--metadata_category", type=str, required=True)
    parser.add_argument("--dataset_size", type=int, required=False)
    parser.add_argument("--apply_aug", action="store_true")
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
        metadatacategory=args.metadata_category,
        datasetsize=args.dataset_size,
        applyaug=args.apply_aug,
        checkpointpath=args.checkpoint_path,
        traindatapath=args.train_data_path,
        valdatapath=args.val_data_path,
        batchsize=args.batch_size,
        numepochs=args.num_epochs,
        numworkers=args.num_workers,
        gradaccsteps=args.grad_acc_steps,
        projectdir=args.project_dir,
    )

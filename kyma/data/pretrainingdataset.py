"""Pretraining dataset implementation."""

from __future__ import annotations

import json
import mmap
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import jsonlines
import torch
from ariautils.tokenizer import Tokenizer
from torch.utils.data import Dataset

from kyma.config.schemas import DatasetHeader
from kyma.data.mididataset import MidiDataset, getseqs, prepareoutputdir, reservoir
from kyma.data.transforms import composetransforms


def _opentextbuffer(path: str | Path):
    return Path(path).open(encoding="utf-8")


class PretrainingDataset(Dataset):
    """Memory-mapped dataset for autoregressive language-model training."""

    def __init__(self, dirpaths: list[str] | str, tokenizer: Tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.transform = None
        self.config: DatasetHeader | None = None
        self.max_seq_len: int | None = None
        self.epochfilesbydir: list[list[str]] = []
        self.curr_epoch: int | None = None
        self._resources = ExitStack()
        self.filebuffs: list = []
        self.filemmaps: list[mmap.mmap] = []
        self.index: list[tuple[int, int]] = []

        if isinstance(dirpaths, str):
            dirpaths = [dirpaths]
        for dirpath in dirpaths:
            self.epochfilesbydir.append(self._getepochfiles(dirpath))
        self.initepoch(0)

    def _getepochfiles(self, dirpath: str) -> list[str]:
        path = Path(dirpath)
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {path}")
        files = []
        for child in path.iterdir():
            if (
                child.is_file()
                and child.name.startswith("epoch")
                and child.suffix == ".jsonl"
            ):
                files.append(child)
        if not files:
            raise FileNotFoundError(f"No epoch shards found in {path}")
        files.sort(key=lambda child: int(child.stem.removeprefix("epoch")))
        for child in files:
            self.checkconfig(child)
        return [str(child) for child in files]

    @classmethod
    def getconfigfrompath(cls, path: str | Path) -> dict:
        epoch0 = Path(path) / "epoch0.jsonl"
        if not epoch0.is_file():
            raise FileNotFoundError(f"Epoch shard not found: {epoch0}")
        with epoch0.open(encoding="utf-8") as handle:
            return json.loads(handle.readline())

    def _buildindex(self, mmapobj: mmap.mmap) -> list[int]:
        mmapobj.seek(0)
        mmapobj.readline()
        positions = []
        while True:
            position = mmapobj.tell()
            line = mmapobj.readline()
            if line == b"":
                break
            positions.append(position)
        return positions

    def checkconfig(self, epochloadpath: str | Path) -> None:
        with Path(epochloadpath).open(encoding="utf-8") as handle:
            header = DatasetHeader(**json.loads(handle.readline()))
        if self.config is not None and header.max_seq_len != self.config.max_seq_len:
            raise ValueError("Dataset shards have inconsistent max_seq_len values.")
        if header.tokenizer_name != self.tokenizer.name:
            raise ValueError(
                "Dataset tokenizer does not match the requested tokenizer."
            )
        self.config = header
        self.max_seq_len = header.max_seq_len

    def settransform(self, transform: Callable | list[Callable]) -> None:
        self.transform = composetransforms(transform)

    def close(self) -> None:
        for mmapobj in self.filemmaps:
            mmapobj.close()
        self._resources.close()
        self._resources = ExitStack()
        self.filemmaps = []
        self.filebuffs = []

    def initepoch(self, index: int | None = None) -> None:
        if index is not None:
            self.curr_epoch = index
        elif self.curr_epoch is None:
            self.curr_epoch = 0
        else:
            self.curr_epoch += 1
        self.close()
        self.index = []
        for dirindex, epochfiles in enumerate(self.epochfilesbydir):
            shardpath = epochfiles[self.curr_epoch % len(epochfiles)]
            buff = self._resources.enter_context(_opentextbuffer(shardpath))
            self.filebuffs.append(buff)
            mmapobj = mmap.mmap(buff.fileno(), 0, access=mmap.ACCESS_READ)
            self.filemmaps.append(mmapobj)
            self.index.extend((dirindex, pos) for pos in self._buildindex(mmapobj))

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return len(self.index)

    def getlossmask(self, srcseq: list, tgtseq: list) -> torch.Tensor:
        return torch.tensor(
            [tok != self.tokenizer.pad_tok for tok in tgtseq],
            dtype=torch.bool,
        )

    def __getitem__(self, index: int):
        def format_token(token):
            return tuple(token) if isinstance(token, list) else token

        fileindex, position = self.index[index]
        mmapobj = self.filemmaps[fileindex]
        mmapobj.seek(position)
        entry = json.loads(mmapobj.readline())
        seq = [format_token(token) for token in entry["seq"]]
        if self.transform is not None:
            seq = list(self.transform(seq))
        src = seq
        tgt = seq[1:] + [self.tokenizer.pad_tok]
        mask = self.getlossmask(src, tgt)
        emb = entry.get("emb")
        return (
            torch.tensor(self.tokenizer.encode(src)),
            torch.tensor(self.tokenizer.encode(tgt)),
            mask,
            torch.tensor(emb) if emb is not None else torch.empty(0),
        )

    @classmethod
    def build(
        cls,
        *,
        tokenizer: Tokenizer,
        savedir: str,
        max_seq_len: int,
        numepochs: int,
        mididataset: MidiDataset | None = None,
        mididatasetpath: str | None = None,
        separatesequences: bool = False,
        fileembeddings: dict | None = None,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than zero.")
        if numepochs <= 0:
            raise ValueError("numepochs must be greater than zero.")
        if (mididataset is None) == (mididatasetpath is None):
            raise ValueError("Provide exactly one of mididataset or mididatasetpath.")

        prepareoutputdir(savedir)
        header = DatasetHeader(
            tokenizer_name=tokenizer.name,
            tokenizer_config=tokenizer.config,
            max_seq_len=max_seq_len,
        )

        def buildconcatepoch(savepath: Path, datasetiter) -> None:
            with jsonlines.open(savepath, mode="w") as writer:
                writer.write(header.__dict__)
                buffer: list = []
                for result in reservoir(getseqs(tokenizer, datasetiter), 10):
                    if result is None:
                        continue
                    entry, _filepath = result
                    buffer += entry
                    while len(buffer) >= max_seq_len:
                        writer.write({"seq": buffer[:max_seq_len]})
                        buffer = buffer[max_seq_len:]
                if buffer:
                    buffer += [tokenizer.pad_tok] * (max_seq_len - len(buffer))
                    writer.write({"seq": buffer[:max_seq_len]})

        def buildseparatedepoch(savepath: Path, datasetiter) -> None:
            with jsonlines.open(savepath, mode="w") as writer:
                writer.write(header.__dict__)
                for result in reservoir(getseqs(tokenizer, datasetiter), 10):
                    if result is None:
                        continue
                    entry, filepath = result
                    buffer = entry
                    payload = (
                        {"emb": fileembeddings[filepath]} if fileembeddings else {}
                    )
                    while len(buffer) >= max_seq_len:
                        writer.write({"seq": buffer[:max_seq_len], **payload})
                        buffer = buffer[max_seq_len:]
                    if buffer:
                        buffer += [tokenizer.pad_tok] * (max_seq_len - len(buffer))
                        writer.write({"seq": buffer[:max_seq_len], **payload})

        for epoch in range(numepochs):
            datasetiter = (
                MidiDataset.getgenerator(mididatasetpath)
                if mididatasetpath is not None
                else mididataset
            )
            if datasetiter is None:
                raise RuntimeError("Failed to initialize the MIDI dataset iterator.")
            savepath = Path(savedir) / f"epoch{epoch}.jsonl"
            if separatesequences:
                buildseparatedepoch(savepath, datasetiter)
            else:
                buildconcatepoch(savepath, datasetiter)

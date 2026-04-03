"""Midi dataset construction and tokenization utilities."""

from __future__ import annotations

import functools
import json
import logging
import multiprocessing
import os
import random
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

import jsonlines
from ariautils.midi import MidiDict, get_metadata_fn, get_test_fn
from ariautils.tokenizer import Tokenizer

from kyma.config.loaders import loadconfig
from kyma.data.metadata import applymanualmetadata, validatemanualmetadata

LOGGER = logging.getLogger(__name__)


class MidiDataset:
    """A collection of ``MidiDict`` objects with JSONL load/save helpers."""

    def __init__(self, entries: list[MidiDict] | Iterable[MidiDict]):
        self.entries = entries

    def __len__(self) -> int:
        if not isinstance(self.entries, list):
            self.entries = list(self.entries)
        return len(self.entries)

    def __getitem__(self, index: int) -> MidiDict:
        if not isinstance(self.entries, list):
            self.entries = list(self.entries)
        return self.entries[index]

    def __iter__(self):
        yield from self.entries

    def shuffle(self) -> None:
        if not isinstance(self.entries, list):
            self.entries = list(self.entries)
        random.shuffle(self.entries)

    def save(self, savepath: str | Path) -> None:
        with jsonlines.open(savepath, mode="w") as writer:
            for mididict in self.entries:
                writer.write(mididict.get_msg_dict())

    @classmethod
    def load(cls, loadpath: str | Path) -> MidiDataset:
        with jsonlines.open(loadpath) as reader:
            entries = [MidiDict.from_msg_dict(entry) for entry in reader]
        return cls(entries)

    @classmethod
    def getgenerator(cls, loadpath: str | Path):
        def generator():
            with jsonlines.open(loadpath, "r") as dataset:
                for entry in dataset:
                    try:
                        yield MidiDict.from_msg_dict(entry)
                    except Exception as exc:  # pragma: no cover - defensive logging
                        LOGGER.info("Failed to load MidiDict: %s", exc)

        return generator()

    @classmethod
    def splitfromfile(
        cls,
        loadpath: str | Path,
        *,
        trainvalratio: float = 0.95,
        repeatable: bool = False,
        overwrite: bool = False,
    ) -> None:
        path = Path(loadpath)
        trainpath = path.with_name(f"{path.stem}_train{path.suffix}")
        valpath = path.with_name(f"{path.stem}_val{path.suffix}")
        if not overwrite and (trainpath.exists() or valpath.exists()):
            raise FileExistsError("Refusing to overwrite an existing split.")
        if repeatable:
            random.seed(42)

        with (
            jsonlines.open(path) as dataset,
            jsonlines.open(trainpath, mode="w") as traindataset,
            jsonlines.open(valpath, mode="w") as valdataset,
        ):
            for entry in dataset:
                if random.uniform(0, 1) <= trainvalratio:
                    traindataset.write(entry)
                else:
                    valdataset.write(entry)

    @classmethod
    def build(
        cls,
        *,
        dir: str,
        recur: bool = False,
        manualmetadata: dict[str, str] | None = None,
        shuffle: bool = True,
        workers: int | None = None,
    ) -> MidiDataset:
        metadata = manualmetadata or {}
        validatemanualmetadata(metadata)
        entries = buildmididictdataset(
            dir=dir,
            recur=recur,
            manualmetadata=metadata,
            shuffle=shuffle,
            workers=workers,
        )
        if entries is None:
            raise RuntimeError("Expected in-memory dataset entries.")
        return cls(entries)

    @classmethod
    def buildtofile(
        cls,
        *,
        dir: str,
        savepath: str,
        recur: bool = False,
        overwrite: bool = False,
        manualmetadata: dict[str, str] | None = None,
        shuffle: bool = True,
        workers: int | None = None,
    ) -> None:
        metadata = manualmetadata or {}
        validatemanualmetadata(metadata)
        buildmididictdataset(
            dir=dir,
            recur=recur,
            streamsavepath=savepath,
            overwrite=overwrite,
            manualmetadata=metadata,
            shuffle=shuffle,
            workers=workers,
        )

    @classmethod
    def combinedatasetsfromfile(cls, *paths: str, outputpath: str) -> None:
        seen: dict[str, bool] = {}
        with jsonlines.open(outputpath, mode="w") as writer:
            for path in paths:
                with jsonlines.open(path, mode="r") as reader:
                    for entry in reader:
                        mididict = MidiDict.from_msg_dict(entry)
                        midihash = mididict.calculate_hash()
                        if seen.get(midihash):
                            continue
                        seen[midihash] = True
                        writer.write(entry)


def _preprocessmididict(mididict: MidiDict, config: dict[str, dict]) -> MidiDict:
    for name, fnconfig in config.items():
        if not fnconfig["run"]:
            continue
        try:
            getattr(mididict, name)(fnconfig["args"])
        except Exception as exc:
            raise RuntimeError(
                f"Failed to run MIDI preprocessing function {name!r}."
            ) from exc
    return mididict


def _addmetadata(mididict: MidiDict, config: dict[str, dict]) -> MidiDict:
    for name, fnconfig in config.items():
        if not fnconfig["run"]:
            continue
        metadatafn = get_metadata_fn(metadata_process_name=name)
        collected = metadatafn(mididict, **fnconfig["args"])
        if collected:
            for key, value in collected.items():
                mididict.metadata[key] = value
    return mididict


def _runtests(mididict: MidiDict, config: dict[str, dict]) -> list[tuple[str, object]]:
    failures: list[tuple[str, object]] = []
    for name, testconfig in config.items():
        if not testconfig["run"]:
            continue
        testfn = get_test_fn(name)
        passed, value = testfn(mididict, **testconfig["args"])
        if not passed:
            failures.append((name, value))
    return failures


def _getmididict(path: Path):
    config = loadconfig()["data"]
    try:
        mididict = MidiDict.from_midi(mid_path=path)
    except Exception as exc:
        LOGGER.error("Failed to load MIDI at %s: %s", path, exc)
        return False, None

    failures = _runtests(mididict, config["tests"])
    if failures:
        LOGGER.info("MIDI at %s failed preprocessing tests: %s", path, failures)
        return False, None

    try:
        mididict = _preprocessmididict(mididict, config["pre_processing"])
    except Exception as exc:
        LOGGER.error("Failed to preprocess MIDI at %s: %s", path, exc)
        return False, None

    mididict = _addmetadata(mididict, config["metadata"]["functions"])
    return True, (mididict, mididict.calculate_hash(), path)


def resolvempworkers(workers: int | None) -> int:
    if workers is None:
        return max(1, os.cpu_count() or 1)
    if workers <= 0:
        raise ValueError("workers must be positive.")
    return workers


def _getmididictsmp(paths: list[Path], *, workers: int | None = None):
    with multiprocessing.Pool(processes=resolvempworkers(workers)) as pool:
        seenhashes: dict[str, list[str]] = defaultdict(list)
        for index, (success, result) in enumerate(
            pool.imap_unordered(_getmididict, paths),
            start=1,
        ):
            if index % 50 == 0:
                LOGGER.info("Processed MIDI files: %s/%s", index, len(paths))
            if not success or result is None:
                continue
            mididict, midihash, midipath = result
            if seenhashes.get(midihash):
                LOGGER.info(
                    "MIDI at %s is a duplicate of %s",
                    midipath,
                    seenhashes[midihash][0],
                )
                seenhashes[midihash].append(str(midipath))
                continue
            seenhashes[midihash].append(str(midipath))
            yield mididict


def buildmididictdataset(
    *,
    dir: str | None = None,
    midpaths: list[str] | None = None,
    recur: bool = False,
    streamsavepath: str | None = None,
    overwrite: bool = False,
    manualmetadata: dict[str, str] | None = None,
    shuffle: bool = True,
    workers: int | None = None,
):
    paths = [Path(path) for path in (midpaths or [])]
    if dir is not None:
        base = Path(dir)
        if recur:
            paths.extend(base.rglob("*.mid"))
            paths.extend(base.rglob("*.midi"))
        else:
            paths.extend(base.glob("*.mid"))
            paths.extend(base.glob("*.midi"))
    if not paths:
        raise FileNotFoundError("No MIDI files were found to build the dataset.")

    metadata = manualmetadata or {}
    if shuffle:
        random.shuffle(paths)
    else:
        base = Path(dir) if dir is not None else None
        if base is not None:
            paths.sort(key=lambda path: path.relative_to(base).as_posix())

    if streamsavepath is None:
        entries: list[MidiDict] = []
        for entry in _getmididictsmp(paths, workers=workers):
            entries.append(applymanualmetadata(entry, metadata))
        return entries

    savepath = Path(streamsavepath)
    if savepath.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {savepath}")
    with jsonlines.open(savepath, mode="w") as writer:
        for entry in _getmididictsmp(paths, workers=workers):
            writer.write(applymanualmetadata(entry, metadata).get_msg_dict())
    return None


def _getseqs(
    entry: MidiDict | dict | str,
    tokenizer: Tokenizer,
    tokenizefn: Callable[[MidiDict], list] | None = None,
):
    if isinstance(entry, str):
        mididict = MidiDict.from_msg_dict(cast(Any, json.loads(entry.rstrip())))
    elif isinstance(entry, dict):
        mididict = MidiDict.from_msg_dict(cast(Any, entry))
    elif isinstance(entry, MidiDict):
        mididict = entry
    else:
        raise TypeError(f"Unsupported entry type {type(entry)!r}.")

    filepath = mididict.metadata["abs_load_path"]
    try:
        tokenized = (
            tokenizefn(mididict)
            if tokenizefn is not None
            else tokenizer.tokenize(mididict)
        )
    except Exception as exc:
        LOGGER.info("Skipping midi_dict during tokenization: %s", exc)
        return None
    return tokenized, filepath


def getseqs(
    tokenizer: Tokenizer,
    mididictiter: Iterable,
    tokenizefn: Callable[[MidiDict], list] | None = None,
    workers: int | None = None,
):
    iterable = (
        list(mididictiter)
        if multiprocessing.get_start_method() == "spawn"
        else mididictiter
    )
    with multiprocessing.Pool(processes=resolvempworkers(workers)) as pool:
        yield from pool.imap_unordered(
            functools.partial(_getseqs, tokenizer=tokenizer, tokenizefn=tokenizefn),
            iterable,
        )


def reservoir(iterable: Iterable, size: int):
    buffer = []
    for entry in iterable:
        if entry is None:
            continue
        buffer.append(entry)
        if len(buffer) >= size:
            random.shuffle(buffer)
            yield from buffer
            buffer = []
    if buffer:
        yield from buffer


def prepareoutputdir(savepath: str | Path) -> None:
    path = Path(savepath)
    if path.exists() and any(path.iterdir()):
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

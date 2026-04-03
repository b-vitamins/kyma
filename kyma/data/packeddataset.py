"""Reusable packed-shard dataset for language-model pretraining."""

from __future__ import annotations

import bisect
import json
import mmap
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jsonlines
import torch
from ariautils.tokenizer import Tokenizer
from torch.utils.data import Dataset

from kyma.config.schemas import PackedDatasetManifest, PackedShard
from kyma.data.mididataset import MidiDataset, getseqs, prepareoutputdir, reservoir
from kyma.data.transforms import composetransforms

PACK_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class _ShardDescriptor:
    path: Path
    sequence_count: int
    loss_token_count: int


@dataclass(slots=True)
class _ShardHandle:
    fileobj: Any
    mmapobj: mmap.mmap
    positions: list[int]


def _opentextbuffer(path: str | Path):
    return Path(path).open("rb")


def _formattoken(token):
    return tuple(token) if isinstance(token, list) else token


def _iterpackedentries(
    *,
    tokenizer: Tokenizer,
    datasetiter: Iterable,
    maxseqlen: int,
    separatesequences: bool,
    fileembeddings: dict[str, list[float]] | None,
):
    if fileembeddings is not None and not separatesequences:
        raise ValueError("Embeddings require separate packed sequences.")

    padtok = tokenizer.pad_tok

    if separatesequences:
        for result in reservoir(getseqs(tokenizer, datasetiter), 10):
            if result is None:
                continue
            tokens, filepath = result
            payload = fileembeddings[filepath] if fileembeddings is not None else None
            buffer = list(tokens)
            while len(buffer) >= maxseqlen:
                yield {"seq": buffer[:maxseqlen], "emb": payload}
                buffer = buffer[maxseqlen:]
            if buffer:
                validtokens = len(buffer)
                yield {
                    "seq": buffer + [padtok] * (maxseqlen - validtokens),
                    "emb": payload,
                }
        return

    seqbuffer: list = []
    for result in reservoir(getseqs(tokenizer, datasetiter), 10):
        if result is None:
            continue
        tokens, _filepath = result
        seqbuffer.extend(tokens)
        while len(seqbuffer) >= maxseqlen:
            yield {"seq": seqbuffer[:maxseqlen], "emb": None}
            seqbuffer = seqbuffer[maxseqlen:]
    if seqbuffer:
        validtokens = len(seqbuffer)
        yield {"seq": seqbuffer + [padtok] * (maxseqlen - validtokens), "emb": None}


class _ShardWriter:
    def __init__(self, *, root: Path, shardsequencecap: int, tokenizer: Tokenizer):
        if shardsequencecap <= 0:
            raise ValueError("shardsequencecap must be positive.")

        self.root = root
        self.shardsequencecap = shardsequencecap
        self.tokenizer = tokenizer
        self.shards: list[PackedShard] = []
        self.sequencecount = 0
        self.losstokencount = 0
        self._shardindex = 0
        self._writer = None
        self._currfile = None
        self._currsequencecount = 0
        self._currlosstokencount = 0
        self._currname = ""

    def _nextlosscount(self, seq: list) -> int:
        nonpad = sum(token != self.tokenizer.pad_tok for token in seq)
        return max(0, nonpad - 1)

    def _openshard(self) -> None:
        self._currname = f"shard-{self._shardindex:06d}.jsonl"
        self._currfile = (self.root / self._currname).open("w", encoding="utf-8")
        self._writer = jsonlines.Writer(self._currfile)
        self._currsequencecount = 0
        self._currlosstokencount = 0

    def _closeshard(self) -> None:
        if self._writer is None or self._currfile is None:
            return
        self._writer.close()
        self._currfile.close()
        self.shards.append(
            PackedShard(
                name=self._currname,
                sequence_count=self._currsequencecount,
                loss_token_count=self._currlosstokencount,
            )
        )
        self._writer = None
        self._currfile = None
        self._shardindex += 1

    def write(self, *, seq: list, emb: list[float] | None) -> None:
        if self._writer is None:
            self._openshard()
        if self._writer is None:
            raise RuntimeError("Shard writer failed to initialize.")
        payload = {"seq": seq}
        if emb is not None:
            payload["emb"] = emb
        self._writer.write(payload)
        losstokens = self._nextlosscount(seq)
        self.sequencecount += 1
        self.losstokencount += losstokens
        self._currsequencecount += 1
        self._currlosstokencount += losstokens
        if self._currsequencecount >= self.shardsequencecap:
            self._closeshard()

    def finish(self) -> list[PackedShard]:
        self._closeshard()
        return self.shards


class PackedDataset(Dataset):
    """Memory-mapped reusable shard dataset for autoregressive pretraining."""

    def __init__(
        self,
        dirpaths: list[str] | str,
        tokenizer: Tokenizer,
        *,
        shardcachesize: int = 8,
    ):
        super().__init__()
        if shardcachesize <= 0:
            raise ValueError("shardcachesize must be positive.")

        self.tokenizer = tokenizer
        self.transform = None
        self.max_seq_len: int | None = None
        self.embedding_size: int | None = None
        self.separate_sequences: bool | None = None
        self.tokenizer_config: dict | None = None
        self.loss_token_count = 0
        self.shards: list[_ShardDescriptor] = []
        self.cumulativecounts: list[int] = []
        self._cache: OrderedDict[int, _ShardHandle] = OrderedDict()
        self._shardcachesize = shardcachesize

        if isinstance(dirpaths, str):
            dirpaths = [dirpaths]
        for dirpath in dirpaths:
            manifest = self.loadmanifest(dirpath)
            self._validatemanifest(manifest)
            base = Path(dirpath)
            for shard in manifest.shards:
                self.shards.append(
                    _ShardDescriptor(
                        path=base / shard.name,
                        sequence_count=shard.sequence_count,
                        loss_token_count=shard.loss_token_count,
                    )
                )
                last = self.cumulativecounts[-1] if self.cumulativecounts else 0
                self.cumulativecounts.append(last + shard.sequence_count)
                self.loss_token_count += shard.loss_token_count
        if not self.shards:
            raise FileNotFoundError("No packed shards were found.")

    @classmethod
    def manifestpath(cls, dirpath: str | Path) -> Path:
        return Path(dirpath) / MANIFEST_NAME

    @classmethod
    def loadmanifest(cls, dirpath: str | Path) -> PackedDatasetManifest:
        manifestpath = cls.manifestpath(dirpath)
        if not manifestpath.is_file():
            raise FileNotFoundError(
                f"Packed dataset manifest not found: {manifestpath}"
            )
        with manifestpath.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return PackedDatasetManifest(
            format_version=int(payload["format_version"]),
            tokenizer_name=str(payload["tokenizer_name"]),
            tokenizer_config=dict(payload["tokenizer_config"]),
            max_seq_len=int(payload["max_seq_len"]),
            shard_token_capacity=int(payload["shard_token_capacity"]),
            separate_sequences=bool(payload["separate_sequences"]),
            embedding_size=(
                int(payload["embedding_size"])
                if payload["embedding_size"] is not None
                else None
            ),
            sequence_count=int(payload["sequence_count"]),
            loss_token_count=int(payload["loss_token_count"]),
            shards=[
                PackedShard(
                    name=str(entry["name"]),
                    sequence_count=int(entry["sequence_count"]),
                    loss_token_count=int(entry["loss_token_count"]),
                )
                for entry in payload["shards"]
            ],
        )

    @classmethod
    def getmanifestdict(cls, dirpath: str | Path) -> dict:
        return asdict(cls.loadmanifest(dirpath))

    def _validatemanifest(self, manifest: PackedDatasetManifest) -> None:
        if manifest.format_version != PACK_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported packed dataset format: {manifest.format_version}."
            )
        if manifest.tokenizer_name != self.tokenizer.name:
            raise ValueError(
                "Packed dataset tokenizer does not match the requested tokenizer."
            )
        if (
            self.tokenizer_config is not None
            and manifest.tokenizer_config != self.tokenizer_config
        ):
            raise ValueError("Packed shards have inconsistent tokenizer configs.")
        if self.max_seq_len is not None and manifest.max_seq_len != self.max_seq_len:
            raise ValueError("Packed shards have inconsistent max_seq_len values.")
        if (
            self.separate_sequences is not None
            and manifest.separate_sequences != self.separate_sequences
        ):
            raise ValueError("Packed shards disagree on separate_sequences mode.")
        if self.shards and manifest.embedding_size != self.embedding_size:
            raise ValueError("Packed shards have inconsistent embedding sizes.")
        self.tokenizer_config = manifest.tokenizer_config
        self.max_seq_len = manifest.max_seq_len
        self.separate_sequences = manifest.separate_sequences
        self.embedding_size = manifest.embedding_size

    def settransform(self, transform: Callable | list[Callable]) -> None:
        self.transform = composetransforms(transform)

    def close(self) -> None:
        for handle in self._cache.values():
            handle.mmapobj.close()
            handle.fileobj.close()
        self._cache.clear()

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return self.cumulativecounts[-1] if self.cumulativecounts else 0

    def _buildpositions(self, mmapobj: mmap.mmap) -> list[int]:
        mmapobj.seek(0)
        positions = []
        while True:
            position = mmapobj.tell()
            line = mmapobj.readline()
            if line == b"":
                break
            positions.append(position)
        return positions

    def _openshard(self, shardindex: int) -> _ShardHandle:
        descriptor = self.shards[shardindex]
        fileobj = _opentextbuffer(descriptor.path)
        mmapobj = mmap.mmap(fileobj.fileno(), 0, access=mmap.ACCESS_READ)
        positions = self._buildpositions(mmapobj)
        if len(positions) != descriptor.sequence_count:
            raise ValueError(
                f"Shard {descriptor.path} expected {descriptor.sequence_count} entries "
                f"but indexed {len(positions)}."
            )
        return _ShardHandle(fileobj=fileobj, mmapobj=mmapobj, positions=positions)

    def _gethandle(self, shardindex: int) -> _ShardHandle:
        cached = self._cache.get(shardindex)
        if cached is not None:
            self._cache.move_to_end(shardindex)
            return cached
        handle = self._openshard(shardindex)
        self._cache[shardindex] = handle
        if len(self._cache) > self._shardcachesize:
            _, evicted = self._cache.popitem(last=False)
            evicted.mmapobj.close()
            evicted.fileobj.close()
        return handle

    def getlossmask(self, srcseq: list, tgtseq: list) -> torch.Tensor:
        return torch.tensor(
            [tok != self.tokenizer.pad_tok for tok in tgtseq],
            dtype=torch.bool,
        )

    def __getitem__(self, index: int):
        shardindex = bisect.bisect_right(self.cumulativecounts, index)
        shardstart = self.cumulativecounts[shardindex - 1] if shardindex > 0 else 0
        localindex = index - shardstart
        handle = self._gethandle(shardindex)
        handle.mmapobj.seek(handle.positions[localindex])
        entry = json.loads(handle.mmapobj.readline())
        seq = [_formattoken(token) for token in entry["seq"]]
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
        shard_tokens: int,
        mididataset: MidiDataset | None = None,
        mididatasetpath: str | None = None,
        separatesequences: bool = False,
        fileembeddings: dict[str, list[float]] | None = None,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than zero.")
        if shard_tokens < max_seq_len:
            raise ValueError("shard_tokens must be at least max_seq_len.")
        if shard_tokens % max_seq_len != 0:
            raise ValueError("shard_tokens must be divisible by max_seq_len.")
        if (mididataset is None) == (mididatasetpath is None):
            raise ValueError("Provide exactly one of mididataset or mididatasetpath.")

        prepareoutputdir(savedir)
        shardsequencecap = shard_tokens // max_seq_len
        root = Path(savedir)
        writer = _ShardWriter(
            root=root,
            shardsequencecap=shardsequencecap,
            tokenizer=tokenizer,
        )

        datasetiter = (
            MidiDataset.getgenerator(mididatasetpath)
            if mididatasetpath is not None
            else mididataset
        )
        if datasetiter is None:
            raise RuntimeError("Failed to initialize the MIDI dataset iterator.")

        for entry in _iterpackedentries(
            tokenizer=tokenizer,
            datasetiter=datasetiter,
            maxseqlen=max_seq_len,
            separatesequences=separatesequences,
            fileembeddings=fileembeddings,
        ):
            writer.write(seq=entry["seq"], emb=entry["emb"])

        embedding_size = None
        if fileembeddings:
            sample = next(iter(fileembeddings.values()))
            embedding_size = len(sample)
            for filepath, emb in fileembeddings.items():
                if len(emb) != embedding_size:
                    raise ValueError(
                        f"Inconsistent embedding size for {filepath}: expected "
                        f"{embedding_size}, got {len(emb)}."
                    )

        manifest = PackedDatasetManifest(
            format_version=PACK_FORMAT_VERSION,
            tokenizer_name=tokenizer.name,
            tokenizer_config=tokenizer.config,
            max_seq_len=max_seq_len,
            shard_token_capacity=shard_tokens,
            separate_sequences=separatesequences,
            embedding_size=embedding_size,
            sequence_count=writer.sequencecount,
            loss_token_count=writer.losstokencount,
            shards=writer.finish(),
        )
        with cls.manifestpath(root).open("w", encoding="utf-8") as handle:
            json.dump(asdict(manifest), handle, indent=2)

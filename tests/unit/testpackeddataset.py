from pathlib import Path

import pytest
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

from kyma.data.mididataset import MidiDataset
from kyma.data.packeddataset import PackedDataset


def _samplemididict() -> MidiDict:
    midipath = Path(__file__).resolve().parents[2] / "example-prompts" / "classical.mid"
    mididict = MidiDict.from_midi(mid_path=midipath)
    mididict.metadata["abs_load_path"] = str(midipath.resolve())
    return mididict


def testbuildwritesmanifestandreusableshards(tmp_path: Path) -> None:
    tokenizer = AbsTokenizer()
    savedir = tmp_path / "packed"
    entries = MidiDataset([_samplemididict() for _ in range(6)])

    PackedDataset.build(
        tokenizer=tokenizer,
        savedir=str(savedir),
        max_seq_len=32,
        shard_tokens=64,
        mididataset=entries,
    )

    manifest = PackedDataset.loadmanifest(savedir)
    assert manifest.tokenizer_name == tokenizer.name
    assert manifest.max_seq_len == 32
    assert manifest.sequence_count > 0
    assert manifest.loss_token_count > 0
    assert len(manifest.shards) >= 2

    dataset = PackedDataset(str(savedir), tokenizer)
    src, tgt, mask, emb = dataset[len(dataset) - 1]
    assert src.shape == (32,)
    assert tgt.shape == (32,)
    assert mask.shape == (32,)
    assert emb.numel() == 0
    dataset.close()


def testbuildsupports_embeddings_for_separate_sequences(tmp_path: Path) -> None:
    tokenizer = AbsTokenizer()
    savedir = tmp_path / "embedded"
    mididict = _samplemididict()
    embedding = [0.25, -0.5, 0.75]

    PackedDataset.build(
        tokenizer=tokenizer,
        savedir=str(savedir),
        max_seq_len=64,
        shard_tokens=128,
        mididataset=MidiDataset([mididict]),
        separatesequences=True,
        fileembeddings={mididict.metadata["abs_load_path"]: embedding},
    )

    dataset = PackedDataset(str(savedir), tokenizer)
    _, _, _, emb = dataset[0]
    assert emb.tolist() == pytest.approx(embedding)
    dataset.close()


def testbuildrejects_embeddings_without_separate_sequences(tmp_path: Path) -> None:
    tokenizer = AbsTokenizer()
    mididict = _samplemididict()

    with pytest.raises(
        ValueError, match="Embeddings require separate packed sequences"
    ):
        PackedDataset.build(
            tokenizer=tokenizer,
            savedir=str(tmp_path / "invalid"),
            max_seq_len=64,
            shard_tokens=128,
            mididataset=MidiDataset([mididict]),
            separatesequences=False,
            fileembeddings={mididict.metadata["abs_load_path"]: [0.0, 1.0]},
        )

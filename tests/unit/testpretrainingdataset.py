from pathlib import Path

import pytest
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer

from kyma.data.mididataset import MidiDataset
from kyma.data.pretrainingdataset import PretrainingDataset


def _samplemididict() -> MidiDict:
    midipath = Path(__file__).resolve().parents[2] / "example-prompts" / "classical.mid"
    mididict = MidiDict.from_midi(mid_path=midipath)
    mididict.metadata["abs_load_path"] = str(midipath.resolve())
    return mididict


@pytest.mark.parametrize("separatesequences", [False, True])
def testbuildwritesdatasetheader(tmp_path: Path, separatesequences: bool) -> None:
    tokenizer = AbsTokenizer()
    savedir = tmp_path / ("separate" if separatesequences else "concat")

    PretrainingDataset.build(
        tokenizer=tokenizer,
        savedir=str(savedir),
        max_seq_len=128,
        numepochs=1,
        mididataset=MidiDataset([_samplemididict()]),
        separatesequences=separatesequences,
    )

    config = PretrainingDataset.getconfigfrompath(savedir)
    assert config["tokenizer_name"] == tokenizer.name
    assert config["max_seq_len"] == 128
    assert config["tokenizer_config"] == tokenizer.config
    assert (savedir / "epoch0.jsonl").stat().st_size > 0

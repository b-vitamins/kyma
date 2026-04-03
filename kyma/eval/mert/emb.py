"""MERT-based audio embedding helpers for symbolic evaluation."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer
from transformers import AutoModel, Wav2Vec2FeatureExtractor


def seq_to_audio_path(
    seq: list,
    tokenizer: AbsTokenizer,
    pianoteq_exec_path: str,
) -> str:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as midtemp:
        midpath = Path(midtemp.name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audiotemp:
        audiopath = Path(audiotemp.name)

    tokenizer.detokenize(seq).to_midi().save(midpath)
    subprocess.run(
        [
            pianoteq_exec_path,
            "--preset",
            "NY Steinway D Classical Recording",
            "--rate",
            "24000",
            "--midi",
            str(midpath),
            "--wav",
            str(audiopath),
        ],
        check=True,
    )
    midpath.unlink()
    return str(audiopath)


def compute_audio_embedding(
    audio_path: str,
    model: nn.Module,
    processor,
    *,
    delete_audio: bool = False,
) -> torch.Tensor:
    """Compute a pooled audio embedding from a rendered waveform."""

    waveform, samplerate = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    targetrate = processor.sampling_rate
    if samplerate != targetrate:
        waveform = T.Resample(orig_freq=samplerate, new_freq=targetrate)(waveform)
    waveform = waveform.squeeze(0)

    segmentlength = targetrate * 5
    totalsamples = waveform.size(0)
    segments = []
    for start in range(0, totalsamples, segmentlength):
        segment = waveform[start : start + segmentlength]
        if segment.size(0) < segmentlength:
            segment = F.pad(segment, (0, segmentlength - segment.size(0)))
        segments.append(segment.numpy())

    inputs = processor(segments, sampling_rate=targetrate, return_tensors="pt")
    inputs = {key: value.cuda() for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hiddenstates = torch.stack(outputs.hidden_states)
    layertimeavg = hiddenstates.mean(dim=2)
    segmentembeddings = layertimeavg.mean(dim=0)
    finalembedding = segmentembeddings.mean(dim=0)

    if delete_audio:
        os.remove(audio_path)
    return finalembedding


def load_mert_model():
    modelname = "m-a-p/MERT-v1-330M"
    return (
        AutoModel.from_pretrained(modelname, trust_remote_code=True).cuda(),
        Wav2Vec2FeatureExtractor.from_pretrained(
            modelname,
            trust_remote_code=True,
        ),
    )


def main() -> None:
    model, processor = load_mert_model()
    tokenizer = AbsTokenizer()
    mididict = MidiDict.from_midi("/home/loubb/Dropbox/shared/test.mid")
    seq = tokenizer.tokenize(mididict)
    audio_path = seq_to_audio_path(seq, tokenizer, pianoteq_exec_path="pianoteq")
    embedding = compute_audio_embedding(
        audio_path=audio_path,
        model=model,
        processor=processor,
        delete_audio=True,
    )
    print(embedding.shape)


if __name__ == "__main__":
    main()

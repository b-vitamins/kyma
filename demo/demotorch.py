#!/usr/bin/env python3

"""Torch-based continuation demo for Kyma."""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import mido
from ariautils.tokenizer import AbsTokenizer
from mido.midifiles.units import second2tick

from kyma.compat.ariacontracts import DEFAULT_MODEL_PRESET
from kyma.compat.checkpointio import loadstate
from kyma.config.loaders import loadmodelschema
from kyma.inference.prompting import getinferenceprompt
from kyma.inference.sampling import samplebatch
from kyma.model import KymaLM


def parseargs():
    parser = argparse.ArgumentParser(
        description="Record a prompt and continue it with Kyma."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the Kyma checkpoint."
    )
    parser.add_argument("--midi_path", required=False, help="Prompt MIDI file.")
    parser.add_argument(
        "--midi_in", required=False, help="MIDI input port for live capture."
    )
    parser.add_argument(
        "--midi_out", required=False, help="MIDI output port for playback."
    )
    parser.add_argument(
        "--prompt_duration", type=int, default=15, help="Prompt duration in seconds."
    )
    parser.add_argument(
        "--record_seconds", type=int, default=15, help="Capture window for live MIDI."
    )
    parser.add_argument(
        "--control_change",
        type=int,
        required=False,
        help="Stop live capture when this CC arrives.",
    )
    parser.add_argument("--length", type=int, default=1024, help="Tokens to generate.")
    parser.add_argument(
        "--temp", type=float, default=0.95, help="Sampling temperature."
    )
    parser.add_argument(
        "--min_p", type=float, default=0.03, help="Min-p sampling threshold."
    )
    parser.add_argument(
        "--top_p", type=float, required=False, help="Top-p sampling threshold."
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Reserved compile flag for parity with other tools.",
    )
    parser.add_argument(
        "--save_path", required=False, help="Where to write the generated MIDI."
    )
    parser.add_argument(
        "--playback", action="store_true", help="Play the generated MIDI to --midi_out."
    )
    return parser.parse_args()


def loadmodel(checkpoint: str) -> KymaLM:
    tokenizer = AbsTokenizer()
    config = loadmodelschema(DEFAULT_MODEL_PRESET)
    config.setvocabsize(tokenizer.vocab_size)
    model = KymaLM(config)
    model.load_state_dict(loadstate(checkpoint), strict=False)
    return model


def createtempmidipath() -> Path:
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as handle:
        return Path(handle.name)


def recordprompt(
    *,
    portname: str,
    recordseconds: int,
    controlchange: int | None,
) -> Path:
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    tempo = 500000
    start = time.monotonic()
    last = start

    with mido.open_input(portname) as port:
        while time.monotonic() - start < recordseconds:
            message = port.poll()
            if message is None:
                time.sleep(0.001)
                continue
            now = time.monotonic()
            delta = second2tick(
                now - last, ticks_per_beat=mid.ticks_per_beat, tempo=tempo
            )
            last = now
            copied = message.copy(time=round(delta))
            track.append(copied)
            if (
                controlchange is not None
                and copied.type == "control_change"
                and copied.control == controlchange
            ):
                break

    path = createtempmidipath()
    mid.save(path)
    return path


def buildresult(
    *,
    checkpoint: str,
    promptpath: Path,
    promptduration: int,
    length: int,
    temp: float,
    topp: float | None,
    minp: float | None,
    compile: bool,
) -> mido.MidiFile:
    tokenizer = AbsTokenizer()
    model = loadmodel(checkpoint)
    prompt = getinferenceprompt(
        mididict=mido_to_mididict(promptpath),
        tokenizer=tokenizer,
        promptlenms=1000 * promptduration,
    )
    results = samplebatch(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        numvariations=1,
        maxnewtokens=min(model.max_seq_len - len(prompt), length),
        temp=temp,
        topp=topp,
        minp=minp,
        compile=compile,
    )
    return tokenizer.detokenize(results[0]).to_midi()


def mido_to_mididict(path: Path):
    from ariautils.midi import MidiDict

    return MidiDict.from_midi(mid_path=path)


def playmidi(mid: mido.MidiFile, *, portname: str) -> None:
    with mido.open_output(portname) as port:
        for message in mid.play():
            if not message.is_meta:
                port.send(message)


def main() -> None:
    args = parseargs()
    if args.midi_path is None and args.midi_in is None:
        raise ValueError("Provide either --midi_path or --midi_in.")

    if args.midi_path is not None:
        promptpath = Path(args.midi_path)
    else:
        promptpath = recordprompt(
            portname=args.midi_in,
            recordseconds=args.record_seconds,
            controlchange=args.control_change,
        )

    result = buildresult(
        checkpoint=args.checkpoint,
        promptpath=promptpath,
        promptduration=args.prompt_duration,
        length=args.length,
        temp=args.temp,
        topp=args.top_p,
        minp=args.min_p,
        compile=args.compile,
    )

    savepath = (
        Path(args.save_path)
        if args.save_path is not None
        else promptpath.with_name(f"{promptpath.stem}_kyma.mid")
    )
    result.save(savepath)

    if args.playback:
        if args.midi_out is None:
            raise ValueError("--playback requires --midi_out.")
        playmidi(result, portname=args.midi_out)

    print(f"Saved generated MIDI to {savepath.resolve()}")


if __name__ == "__main__":
    main()

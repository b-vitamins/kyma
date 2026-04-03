"""Prompt construction helpers."""

from __future__ import annotations

from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer


def getcfgprompt(prompts: list[list]) -> list[list]:
    cfgprompts: list[list] = []
    for prompt in prompts:
        cfgprompts.append(prompt)
        cfgprompts.append(prompt)
    return cfgprompts


def getinferenceprompt(
    mididict: MidiDict,
    tokenizer: AbsTokenizer,
    promptlenms: int,
) -> list:
    mididict.note_msgs = [
        msg
        for msg in mididict.note_msgs
        if mididict.tick_to_ms(msg["data"]["start"]) <= promptlenms
    ]
    mididict.pedal_msgs = [
        msg
        for msg in mididict.pedal_msgs
        if mididict.tick_to_ms(msg["tick"]) <= promptlenms
    ]
    if mididict.pedal_msgs and mididict.pedal_msgs[-1]["data"] == 1:
        mididict.pedal_msgs.pop()

    if not mididict.note_msgs:
        return [("prefix", "instrument", "piano"), tokenizer.bos_tok]

    return tokenizer.tokenize(
        midi_dict=mididict,
        add_dim_tok=False,
        add_eos_tok=False,
    )

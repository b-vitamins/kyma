"""M3 and CLaMP3 symbolic embedding helpers."""

from __future__ import annotations

import os

import mido
import torch
from transformers import BertConfig, GPT2Config

from kyma.eval.m3.config import (
    AUDIO_HIDDEN_SIZE,
    AUDIO_NUM_LAYERS,
    CLAMP3_HIDDEN_SIZE,
    M3_HIDDEN_SIZE,
    MAX_AUDIO_LENGTH,
    PATCH_LENGTH,
    PATCH_NUM_LAYERS,
    PATCH_SIZE,
    TEXT_MODEL_NAME,
    TOKEN_NUM_LAYERS,
)
from kyma.eval.m3.utils import CLaMP3Model, M3Model, M3Patchilizer


def msg_to_str(msg) -> str:
    return (
        " ".join(str(value) for _key, value in msg.dict().items())
        .encode("unicode_escape")
        .decode("utf-8")
    )


def load_midi(
    *,
    filename: str | None = None,
    mid: mido.MidiFile | None = None,
    m3_compatible: bool = True,
) -> str:
    """Load a MIDI file and convert it to the MTF text format."""

    if mid is None:
        if filename is None or not os.path.isfile(filename):
            raise FileNotFoundError(f"MIDI file not found: {filename}")
        mid = mido.MidiFile(filename)

    msglist = [f"ticks_per_beat {mid.ticks_per_beat}"]
    for msg in mido.merge_tracks(mid.tracks):
        if (
            m3_compatible
            and msg.is_meta
            and msg.type
            in {
                "text",
                "copyright",
                "track_name",
                "instrument_name",
                "lyrics",
                "marker",
                "cue_marker",
                "device_name",
            }
        ):
            continue
        msglist.append(msg_to_str(msg))
    return "\n".join(msglist)


def load_clamp3_model(
    checkpoint_path: str,
    *,
    m3_only: bool = False,
) -> tuple[CLaMP3Model, M3Patchilizer]:
    """Load a CLaMP3 checkpoint and return it with its patchilizer."""

    audio_config = BertConfig(
        vocab_size=1,
        hidden_size=AUDIO_HIDDEN_SIZE,
        num_hidden_layers=AUDIO_NUM_LAYERS,
        num_attention_heads=AUDIO_HIDDEN_SIZE // 64,
        intermediate_size=AUDIO_HIDDEN_SIZE * 4,
        max_position_embeddings=MAX_AUDIO_LENGTH,
    )
    symbolic_config = BertConfig(
        vocab_size=1,
        hidden_size=M3_HIDDEN_SIZE,
        num_hidden_layers=PATCH_NUM_LAYERS,
        num_attention_heads=M3_HIDDEN_SIZE // 64,
        intermediate_size=M3_HIDDEN_SIZE * 4,
        max_position_embeddings=PATCH_LENGTH,
    )
    decoder_config = GPT2Config(
        vocab_size=128,
        n_positions=PATCH_SIZE,
        n_embd=M3_HIDDEN_SIZE,
        n_layer=TOKEN_NUM_LAYERS,
        n_head=M3_HIDDEN_SIZE // 64,
        n_inner=M3_HIDDEN_SIZE * 4,
    )

    model = CLaMP3Model(
        audio_config=audio_config,
        symbolic_config=symbolic_config,
        text_model_name=TEXT_MODEL_NAME,
        hidden_size=CLAMP3_HIDDEN_SIZE,
        load_m3=True,
    ).to("cuda")
    model.eval()

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    if not m3_only:
        model.load_state_dict(checkpoint["model"])
    else:
        temp_model = M3Model(symbolic_config, decoder_config)
        temp_model.load_state_dict(checkpoint["model"])
        model.symbolic_model.load_state_dict(temp_model.encoder.state_dict())

    return model, M3Patchilizer()


def get_midi_embedding(
    *,
    mid: mido.MidiFile,
    model: CLaMP3Model,
    patchilizer: M3Patchilizer,
    get_global: bool = True,
) -> torch.Tensor:
    """Compute a symbolic embedding for a MIDI file."""

    device = "cuda"
    mtf = load_midi(mid=mid, m3_compatible=True)
    patches = patchilizer.encode(mtf, add_special_patches=True)
    tokentensor = torch.tensor(patches, dtype=torch.long, device=device)

    numtokens = tokentensor.size(0)
    segments = []
    weights = []
    for start in range(0, numtokens, PATCH_LENGTH):
        segment = tokentensor[start : start + PATCH_LENGTH]
        segments.append(segment)
        weights.append(segment.size(0))
    if numtokens > PATCH_LENGTH:
        segments[-1] = tokentensor[-PATCH_LENGTH:]
        weights[-1] = segments[-1].size(0)

    processed = []
    for segment in segments:
        currentlen = segment.size(0)
        if currentlen < PATCH_LENGTH:
            pad = torch.full(
                (PATCH_LENGTH - currentlen, tokentensor.size(1)),
                patchilizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            segment = torch.cat([segment, pad], dim=0)
        segment = segment.unsqueeze(0)
        mask = torch.cat(
            [
                torch.ones(currentlen, device=device),
                torch.zeros(PATCH_LENGTH - currentlen, device=device),
            ]
        ).unsqueeze(0)
        with torch.no_grad():
            features = model.get_symbolic_features(
                symbolic_inputs=segment,
                symbolic_masks=mask,
                get_global=get_global,
            )
        if not get_global:
            features = features[:, : int(mask.sum().item()), :]
        processed.append(features)

    if not get_global:
        embedding = torch.cat([features.squeeze(0) for features in processed], dim=0)
        return embedding.view(-1)

    featurestack = torch.stack([features.squeeze(0) for features in processed], dim=0)
    weighttensor = torch.tensor(weights, dtype=torch.float, device=device).view(-1, 1)
    embedding = (featurestack * weighttensor).sum(dim=0) / weighttensor.sum()
    return embedding.view(-1)

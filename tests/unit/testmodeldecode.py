from __future__ import annotations

import torch

from kyma.model import KymaLM, ModelConfig


def test_prefill_and_decode_match_full_forward() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        d_model=64,
        n_heads=1,
        n_layers=2,
        ff_mult=2,
        drop_p=0.0,
        max_seq_len=32,
        grad_checkpoint=False,
        vocab_size=128,
        d_state=64,
        expand=1,
        d_conv=4,
        chunk_size=8,
    )
    model = KymaLM(config).eval()
    tokens = torch.randint(0, config.vocab_size or 128, (2, 7))

    with torch.inference_mode():
        fulllogits = model(tokens)
        logits, state = model.prefill(tokens[:, :1])
        for index in range(1, tokens.shape[1]):
            step_logits, state = model.decodeone(tokens[:, index], state)
            logits = torch.cat([logits, step_logits.unsqueeze(1)], dim=1)

    assert torch.allclose(logits, fulllogits, atol=1e-4, rtol=1e-4)


def test_conditioned_forward_drops_first_position() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        d_model=64,
        n_heads=1,
        n_layers=2,
        ff_mult=2,
        drop_p=0.0,
        max_seq_len=32,
        grad_checkpoint=False,
        vocab_size=128,
        emb_size=16,
        d_state=64,
        expand=1,
        d_conv=4,
        chunk_size=8,
    )
    model = KymaLM(config).eval()
    tokens = torch.randint(0, config.vocab_size or 128, (2, 7))
    emb = torch.randn(2, 16)

    with torch.inference_mode():
        logits = model(tokens, emb=emb)

    assert logits.shape == (2, 6, 128)

"""Rotary-position helpers shared by Kyma backbones."""

from __future__ import annotations

import torch


def precomputefreqscis(
    *,
    seqlen: int,
    nelem: int,
    base: int = 500000,
) -> torch.Tensor:
    """Precompute Aria-style RoPE frequencies."""

    freqs = 1.0 / (base ** (torch.arange(0, nelem, 2)[: (nelem // 2)].float() / nelem))
    timesteps = torch.arange(seqlen, device=freqs.device)
    freqs = torch.outer(timesteps, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)


@torch.jit.script
def applyrotaryemb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply Aria-style in-place RoPE to a ``(B, T, H, Dh)`` tensor."""

    xfloat = x.float()
    freqs_cis = freqs_cis.detach()
    d = xfloat.shape[-1] // 2
    cos = freqs_cis[..., 0][None, :, None]
    sin = freqs_cis[..., 1][None, :, None]
    x1, x2 = xfloat[..., :d], xfloat[..., d : d * 2]
    tmp = x1.clone()
    x1.mul_(cos).addcmul_(x2, sin, value=-1)
    x2.mul_(cos).addcmul_(tmp, sin, value=1)
    return x.copy_(xfloat)


def applyroperaw(
    hidden: torch.Tensor,
    freqs_cis: torch.Tensor,
    *,
    nheads: int,
) -> torch.Tensor:
    """Apply RoPE to hidden states by viewing them as head-packed channels."""

    if hidden.ndim != 3:
        raise ValueError(
            f"Expected hidden shape (batch, time, d_model), got {tuple(hidden.shape)}."
        )
    batch, timesteps, channels = hidden.shape
    dhead = channels // nheads
    shaped = hidden.reshape(batch, timesteps, nheads, dhead).contiguous()
    rotated = applyrotaryemb(shaped, freqs_cis)
    return rotated.reshape(batch, timesteps, channels)

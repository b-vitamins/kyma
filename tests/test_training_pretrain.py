from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from kyma.data import (
    KymaStateCarryDataset,
    KymaTimeFeatures,
    KymaTokenizedPiece,
    KymaWindowSpec,
)
from kyma.model import (
    KymaAutoregressiveLM,
    KymaBackendConfig,
    KymaEvalDifferentiators,
    KymaLongContextConfig,
    KymaModelConfig,
    KymaTimeConditioningConfig,
)
from kyma.training import (
    KymaOptimizerConfig,
    KymaPretrainConfig,
    KymaScheduleConfig,
    KymaStateCarryBatcher,
    KymaTrainState,
    detach_state_rows,
    evaluate_language_model,
    load_pretrain_checkpoint,
    merge_state_rows,
    save_pretrain_checkpoint,
    train_language_model,
)


@dataclass(frozen=True)
class FakeMixerState:
    running: torch.Tensor

    def detach(self) -> FakeMixerState:
        return FakeMixerState(running=self.running.detach())

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> FakeMixerState:
        return FakeMixerState(running=self.running.to(device=device, dtype=dtype))


class IdentityMixer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(d_model))

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> FakeMixerState:
        resolved_dtype = torch.float32 if dtype is None else dtype
        return FakeMixerState(
            running=torch.zeros((batch_size, 1), device=device, dtype=resolved_dtype)
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        state: FakeMixerState | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, FakeMixerState]:
        y = self.proj(x)
        next_running = x.new_full((x.shape[0], 1), float(x.shape[1]))
        if state is not None:
            next_running = next_running + state.running.to(
                device=x.device,
                dtype=x.dtype,
            )
        next_state = FakeMixerState(running=next_running)
        if not return_state:
            return y
        return y, next_state


def _build_config() -> KymaModelConfig:
    return KymaModelConfig(
        d_model=16,
        n_layers=2,
        d_state=8,
        expand=2,
        d_head=8,
        d_conv=4,
        chunk_size=8,
        vocab_size=32,
        dropout_p=0.0,
        ffn_mult=2,
        max_segment_len=8,
        backends=KymaBackendConfig(scan_backend="reference"),
        time_conditioning=KymaTimeConditioningConfig(
            learned_positional_embedding=False,
            delta_time_features=True,
            absolute_time_features=True,
            beat_phase_features=True,
            tempo_features=True,
            feature_mlp_dim=12,
        ),
        long_context=KymaLongContextConfig(),
        differentiators=KymaEvalDifferentiators(),
    )


def _fake_mixer_factory(config: KymaModelConfig) -> nn.Module:
    return IdentityMixer(config.d_model)


def _make_piece(piece_id: str, length: int) -> KymaTokenizedPiece:
    token_ids = torch.arange(length, dtype=torch.long) % 16
    values = torch.arange(length * 4, dtype=torch.float32).view(length, 4)
    valid = torch.ones((length, 4), dtype=torch.bool)
    return KymaTokenizedPiece(
        piece_id=piece_id,
        tokens=tuple(f"tok-{idx}" for idx in range(length)),
        token_ids=token_ids,
        time_features=KymaTimeFeatures(values=values, valid=valid),
        metadata={},
        source_path=None,
    )


def _build_dataset() -> KymaStateCarryDataset:
    window_spec = KymaWindowSpec(
        chunk_size_tokens=3,
        burn_in_tokens=1,
        tbptt_window_tokens=3,
        max_piece_tokens=16,
    )
    return KymaStateCarryDataset.from_pieces(
        [
            _make_piece("piece-a", 5),
            _make_piece("piece-b", 4),
            _make_piece("piece-c", 3),
        ],
        window_spec=window_spec,
        pad_token_id=31,
    )


def _build_multi_chunk_tbptt_dataset() -> KymaStateCarryDataset:
    window_spec = KymaWindowSpec(
        chunk_size_tokens=3,
        burn_in_tokens=1,
        tbptt_window_tokens=6,
        max_piece_tokens=16,
    )
    return KymaStateCarryDataset.from_pieces(
        [
            _make_piece("piece-a", 8),
            _make_piece("piece-b", 7),
        ],
        window_spec=window_spec,
        pad_token_id=31,
    )


def test_state_carry_batcher_preserves_piece_continuity() -> None:
    dataset = _build_dataset()
    batcher = KymaStateCarryBatcher(dataset, batch_size=2)

    batches = list(batcher)

    assert len(batches) == 3
    assert batches[0]["piece_ids"] == ["piece-a", "piece-b"]
    assert batches[1]["piece_ids"] == ["piece-a", "piece-b"]
    assert batches[2]["piece_ids"] == ["piece-c", ""]
    assert [bool(value) for value in batches[1]["carry_from_previous"]] == [True, True]
    assert [bool(value) for value in batches[2]["active_mask"]] == [True, False]


def test_merge_state_rows_resets_selected_rows() -> None:
    current = FakeMixerState(running=torch.tensor([[5.0], [7.0]]))
    fresh = FakeMixerState(running=torch.tensor([[0.0], [0.0]]))

    merged = merge_state_rows(
        current,
        fresh,
        keep_mask=torch.tensor([True, False], dtype=torch.bool),
    )

    assert torch.equal(merged.running, torch.tensor([[5.0], [0.0]]))


def test_detach_state_rows_breaks_graph_for_detached_rows() -> None:
    base = torch.randn((2, 3), dtype=torch.float32, requires_grad=True)
    state = base * 2.0

    fully_detached = detach_state_rows(
        state,
        detach_mask=torch.tensor([True, True], dtype=torch.bool),
    )
    assert fully_detached.requires_grad is False

    mixed = detach_state_rows(
        state,
        detach_mask=torch.tensor([True, False], dtype=torch.bool),
    )
    mixed[1].sum().backward()
    assert base.grad is not None
    assert torch.equal(base.grad[0], torch.zeros_like(base.grad[0]))
    assert torch.count_nonzero(base.grad[1]) > 0


def test_checkpoint_roundtrip_and_tiny_training_loop(tmp_path: Path) -> None:
    model_config = _build_config()
    model = KymaAutoregressiveLM(model_config, mixer_factory=_fake_mixer_factory)
    dataset = _build_dataset()
    pretrain_config = KymaPretrainConfig(
        batch_size=2,
        max_steps=3,
        checkpoint_every_steps=2,
        device="cpu",
        optimizer=KymaOptimizerConfig(lr=1e-3, weight_decay=0.0),
        schedule=KymaScheduleConfig(warmup_steps=0, min_lr_scale=0.5),
    )

    train_state = train_language_model(
        model,
        dataset,
        model_config=model_config,
        pretrain_config=pretrain_config,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert train_state.global_step == 3
    assert train_state.optimizer_steps == 3
    assert train_state.tokens_processed > 0
    assert (tmp_path / "checkpoints" / "step2.pt").is_file()
    assert (tmp_path / "checkpoints" / "latest.pt").is_file()

    metrics = evaluate_language_model(model, dataset, batch_size=2, device="cpu")
    assert metrics.valid_tokens > 0
    assert metrics.loss >= 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bundle = load_pretrain_checkpoint(
        tmp_path / "checkpoints" / "latest.pt",
        model=model,
        optimizer=optimizer,
    )

    assert bundle.model_config.to_dict() == model_config.to_dict()
    assert bundle.pretrain_config.to_dict() == pretrain_config.to_dict()
    assert bundle.train_state.global_step == 3
    assert bundle.scaler_state is None


def test_training_loop_supports_gradient_accumulation(tmp_path: Path) -> None:
    model_config = _build_config()
    model = KymaAutoregressiveLM(model_config, mixer_factory=_fake_mixer_factory)
    dataset = _build_dataset()
    pretrain_config = KymaPretrainConfig(
        batch_size=2,
        max_steps=2,
        grad_accum_steps=2,
        checkpoint_every_steps=1,
        device="cpu",
        optimizer=KymaOptimizerConfig(lr=1e-3, weight_decay=0.0),
        schedule=KymaScheduleConfig(warmup_steps=0, min_lr_scale=0.5),
    )

    train_state = train_language_model(
        model,
        dataset,
        model_config=model_config,
        pretrain_config=pretrain_config,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert train_state.global_step == 3
    assert train_state.optimizer_steps == 2
    assert train_state.tokens_processed > 0
    assert (tmp_path / "checkpoints" / "step1.pt").is_file()
    assert (tmp_path / "checkpoints" / "step2.pt").is_file()


def test_training_loop_handles_multi_chunk_tbptt_segments(tmp_path: Path) -> None:
    model_config = _build_config()
    model = KymaAutoregressiveLM(model_config, mixer_factory=_fake_mixer_factory)
    dataset = _build_multi_chunk_tbptt_dataset()
    pretrain_config = KymaPretrainConfig(
        batch_size=2,
        max_steps=1,
        grad_accum_steps=4,
        checkpoint_every_steps=1,
        device="cpu",
        optimizer=KymaOptimizerConfig(lr=1e-3, weight_decay=0.0),
        schedule=KymaScheduleConfig(warmup_steps=0, min_lr_scale=0.5),
    )

    train_state = train_language_model(
        model,
        dataset,
        model_config=model_config,
        pretrain_config=pretrain_config,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert train_state.optimizer_steps == 1
    assert train_state.global_step >= 2
    assert train_state.tokens_processed > 0


def test_save_pretrain_checkpoint_roundtrips_scalar_state(tmp_path: Path) -> None:
    model_config = _build_config()
    model = KymaAutoregressiveLM(model_config, mixer_factory=_fake_mixer_factory)
    pretrain_config = KymaPretrainConfig(batch_size=1, max_steps=1)
    train_state = KymaTrainState(global_step=1, optimizer_steps=1, tokens_processed=12)

    checkpoint_path = tmp_path / "manual.pt"
    save_pretrain_checkpoint(
        checkpoint_path,
        model=model,
        model_config=model_config,
        pretrain_config=pretrain_config,
        train_state=train_state,
        extra={"note": "manual"},
    )

    bundle = load_pretrain_checkpoint(checkpoint_path)
    assert bundle.train_state.to_dict() == train_state.to_dict()
    assert bundle.extra["note"] == "manual"

"""Main pretraining loop for Kyma language models."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from torch.nn import functional as F

from kyma.data import (
    KymaStateCarryDataset,
    KymaTrainingWindow,
    collate_training_windows,
)
from kyma.model import KymaAutoregressiveLM, KymaLMState, KymaModelConfig
from kyma.training.checkpoint import (
    KymaTrainState,
    save_pretrain_checkpoint,
)
from kyma.training.config import KymaOptimizerConfig, KymaPretrainConfig


@dataclass(frozen=True)
class KymaTrainMetrics:
    """Scalar metrics emitted by the training and evaluation loops."""

    loss: float
    valid_tokens: int
    learning_rate: float


class GradScalerLike(Protocol):
    """Minimal gradient-scaler surface used by the pretraining loop."""

    def is_enabled(self) -> bool: ...

    def scale(self, outputs: torch.Tensor) -> torch.Tensor: ...

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None: ...

    def step(self, optimizer: torch.optim.Optimizer) -> Any: ...

    def update(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


def build_optimizer(
    model: torch.nn.Module,
    config: KymaOptimizerConfig,
) -> torch.optim.Optimizer:
    """Create the default AdamW optimizer for Kyma pretraining."""

    use_fused = any(parameter.is_cuda for parameter in model.parameters())
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
        fused=use_fused,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_scale: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a warmup + cosine-decay schedule."""

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return min_lr_scale
        progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _broadcast_mask(mask: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    view_shape = (mask.shape[0],) + (1,) * max(tensor.ndim - 1, 0)
    return mask.view(view_shape)


def _map_state_trees(
    current: Any,
    other: Any,
    fn: Any,
) -> Any:
    if current is None or other is None:
        return current if other is None else other
    if torch.is_tensor(current) and torch.is_tensor(other):
        return fn(current, other)
    if is_dataclass(current) and is_dataclass(other):
        values: dict[str, Any] = {
            field.name: _map_state_trees(
                getattr(current, field.name),
                getattr(other, field.name),
                fn,
            )
            for field in fields(current)
        }
        return type(current)(**values)
    if isinstance(current, tuple) and isinstance(other, tuple):
        current_tuple = cast(tuple[Any, ...], current)
        other_tuple = cast(tuple[Any, ...], other)
        return tuple(
            _map_state_trees(curr_item, other_item, fn)
            for curr_item, other_item in zip(current_tuple, other_tuple, strict=True)
        )
    if isinstance(current, list) and isinstance(other, list):
        current_list = cast(list[Any], current)
        other_list = cast(list[Any], other)
        return [
            _map_state_trees(curr_item, other_item, fn)
            for curr_item, other_item in zip(current_list, other_list, strict=True)
        ]
    if isinstance(current, dict) and isinstance(other, dict):
        current_dict = cast(dict[str, Any], current)
        other_dict = cast(dict[str, Any], other)
        return {
            key: _map_state_trees(current_dict[key], other_dict[key], fn)
            for key in current_dict
        }
    return cast(Any, current)


def merge_state_rows(
    current: Any,
    fresh: Any,
    *,
    keep_mask: torch.Tensor,
) -> Any:
    """Keep recurrent rows where `keep_mask` is true and reset the rest."""

    def _merge(
        current_tensor: torch.Tensor,
        fresh_tensor: torch.Tensor,
    ) -> torch.Tensor:
        if current_tensor.ndim == 0 or current_tensor.shape[0] != keep_mask.shape[0]:
            return current_tensor
        mask = _broadcast_mask(keep_mask, current_tensor)
        return torch.where(mask, current_tensor, fresh_tensor)

    return _map_state_trees(current, fresh, _merge)


def detach_state_rows(state: Any, *, detach_mask: torch.Tensor) -> Any:
    """Detach recurrent rows where `detach_mask` is true."""

    if state is None:
        return None
    if torch.is_tensor(state):
        if state.ndim == 0 or state.shape[0] != detach_mask.shape[0]:
            return state.detach()
        if bool(detach_mask.all().item()):
            return state.detach()
        if not bool(detach_mask.any().item()):
            return state
        detached = state.detach().clone()
        keep_mask = ~detach_mask
        detached[keep_mask] = state[keep_mask]
        return detached
    if is_dataclass(state):
        values: dict[str, Any] = {
            field.name: detach_state_rows(
                getattr(state, field.name),
                detach_mask=detach_mask,
            )
            for field in fields(state)
        }
        return type(state)(**values)
    if isinstance(state, tuple):
        state_tuple = cast(tuple[Any, ...], state)
        return tuple(
            detach_state_rows(item, detach_mask=detach_mask) for item in state_tuple
        )
    if isinstance(state, list):
        state_list = cast(list[Any], state)
        return [detach_state_rows(item, detach_mask=detach_mask) for item in state_list]
    if isinstance(state, dict):
        state_dict = cast(dict[str, Any], state)
        return {
            key: detach_state_rows(value, detach_mask=detach_mask)
            for key, value in state_dict.items()
        }
    return state


def _empty_window_like(window: KymaTrainingWindow) -> KymaTrainingWindow:
    input_ids = torch.zeros_like(window.input_ids)
    target_ids = torch.zeros_like(window.target_ids)
    time_features = torch.zeros_like(window.time_features)
    time_feature_mask = torch.zeros_like(window.time_feature_mask)
    loss_mask = torch.zeros_like(window.loss_mask)
    return KymaTrainingWindow(
        piece_id="",
        window_index=-1,
        start_token_idx=0,
        input_ids=input_ids,
        target_ids=target_ids,
        time_features=time_features,
        time_feature_mask=time_feature_mask,
        loss_mask=loss_mask,
        carry_from_previous=False,
        detach_state_after=True,
        is_piece_start=False,
        is_piece_end=True,
    )


def _group_windows_by_piece(
    dataset: KymaStateCarryDataset,
) -> list[list[KymaTrainingWindow]]:
    grouped: list[list[KymaTrainingWindow]] = []
    for window in dataset.windows:
        if not grouped or grouped[-1][0].piece_id != window.piece_id:
            grouped.append([window])
        else:
            grouped[-1].append(window)
    return grouped


class KymaStateCarryBatcher:
    """Iterate windows in stream-aligned batches that preserve piece continuity."""

    def __init__(self, dataset: KymaStateCarryDataset, *, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.dataset = dataset
        self.batch_size = batch_size
        self._piece_streams = _group_windows_by_piece(dataset)

    def __len__(self) -> int:
        if not self._piece_streams:
            return 0
        initial_slots = min(self.batch_size, len(self._piece_streams))
        queue = deque(len(stream) for stream in self._piece_streams[initial_slots:])
        active = [len(stream) for stream in self._piece_streams[:initial_slots]]
        steps = 0
        while active:
            steps += 1
            next_active: list[int] = []
            for remaining in active:
                remaining -= 1
                if remaining > 0:
                    next_active.append(remaining)
                elif queue:
                    next_active.append(queue.popleft())
            active = next_active
        return steps

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if not self.dataset.windows:
            return

        template = self.dataset.windows[0]
        empty_window = _empty_window_like(template)
        queue: deque[list[KymaTrainingWindow]] = deque(self._piece_streams)
        slots: list[tuple[list[KymaTrainingWindow], int] | None] = []
        for _ in range(self.batch_size):
            if queue:
                slots.append((queue.popleft(), 0))
            else:
                slots.append(None)

        while any(slot is not None for slot in slots):
            windows: list[KymaTrainingWindow] = []
            active_mask: list[bool] = []
            next_slots: list[tuple[list[KymaTrainingWindow], int] | None] = []

            for slot in slots:
                if slot is None:
                    windows.append(empty_window)
                    active_mask.append(False)
                    next_slots.append(None)
                    continue

                stream, index = slot
                windows.append(stream[index])
                active_mask.append(True)
                next_index = index + 1
                if next_index < len(stream):
                    next_slots.append((stream, next_index))
                elif queue:
                    next_slots.append((queue.popleft(), 0))
                else:
                    next_slots.append(None)

            batch = collate_training_windows(windows)
            batch["active_mask"] = torch.tensor(active_mask, dtype=torch.bool)
            yield batch
            slots = next_slots


def _move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _zero_state(
    model: KymaAutoregressiveLM,
    *,
    batch_size: int,
    device: torch.device,
) -> KymaLMState:
    return model.init_state(
        batch_size,
        device=device,
        dtype=model.token_embed.weight.dtype,
    )


def _autocast_context(
    *,
    device: torch.device,
    precision: str,
):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("Mixed precision is only supported on CUDA devices.")
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _build_grad_scaler(
    *,
    device: torch.device,
    precision: str,
) -> GradScalerLike:
    amp_module = cast(Any, torch.amp)
    grad_scaler_cls = amp_module.GradScaler
    return grad_scaler_cls(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )


def _optimizer_step(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScalerLike,
    model: torch.nn.Module,
    grad_clip_norm: float | None,
) -> None:
    if scaler.is_enabled():
        if grad_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _backward(loss: torch.Tensor, *, retain_graph: bool = False) -> None:
    torch.autograd.backward(loss, retain_graph=retain_graph)


def compute_language_model_loss(
    *,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Compute masked autoregressive cross-entropy."""

    token_loss = F.cross_entropy(
        logits.transpose(1, 2),
        target_ids,
        reduction="none",
    )
    valid_tokens = int(loss_mask.sum().item())
    if valid_tokens == 0:
        return token_loss.sum() * 0.0, 0
    masked_loss = token_loss[loss_mask]
    return masked_loss.mean(), valid_tokens


@torch.no_grad()
def evaluate_language_model(
    model: KymaAutoregressiveLM,
    dataset: KymaStateCarryDataset,
    *,
    batch_size: int,
    device: str | torch.device,
    precision: str = "fp32",
) -> KymaTrainMetrics:
    """Run the pretraining objective without gradient updates."""

    resolved_device = torch.device(device)
    batcher = KymaStateCarryBatcher(dataset, batch_size=batch_size)
    if len(batcher) == 0:
        return KymaTrainMetrics(loss=0.0, valid_tokens=0, learning_rate=0.0)

    was_training = model.training
    model.eval()
    state = _zero_state(model, batch_size=batch_size, device=resolved_device)
    total_loss = 0.0
    total_tokens = 0

    for batch in batcher:
        batch = _move_batch_to_device(batch, resolved_device)
        keep_mask = batch["carry_from_previous"] & batch["active_mask"]
        fresh_state = _zero_state(model, batch_size=batch_size, device=resolved_device)
        state = merge_state_rows(state, fresh_state, keep_mask=keep_mask)
        with _autocast_context(device=resolved_device, precision=precision):
            logits, next_state = cast(
                tuple[torch.Tensor, KymaLMState],
                model(
                    batch["input_ids"],
                    time_features=batch["time_features"],
                    time_feature_mask=batch["time_feature_mask"],
                    state=state,
                    return_state=True,
                ),
            )
            loss, valid_tokens = compute_language_model_loss(
                logits=logits,
                target_ids=batch["target_ids"],
                loss_mask=batch["loss_mask"],
            )
        total_loss += float(loss) * valid_tokens
        total_tokens += valid_tokens

        active_state = merge_state_rows(
            next_state,
            fresh_state,
            keep_mask=batch["active_mask"],
        )
        detach_mask = batch["detach_state_after"] | (~batch["active_mask"])
        state = detach_state_rows(active_state, detach_mask=detach_mask)

    if was_training:
        model.train()

    mean_loss = 0.0 if total_tokens == 0 else total_loss / total_tokens
    return KymaTrainMetrics(
        loss=mean_loss,
        valid_tokens=total_tokens,
        learning_rate=0.0,
    )


def train_language_model(
    model: KymaAutoregressiveLM,
    dataset: KymaStateCarryDataset,
    *,
    model_config: KymaModelConfig,
    pretrain_config: KymaPretrainConfig,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    checkpoint_dir: str | Path | None = None,
) -> KymaTrainState:
    """Run the main Kyma pretraining loop."""

    resolved_device = torch.device(pretrain_config.device)
    model.to(resolved_device)
    if optimizer is None:
        optimizer = build_optimizer(model, pretrain_config.optimizer)
    if scheduler is None:
        scheduler = build_scheduler(
            optimizer,
            total_steps=pretrain_config.max_steps,
            warmup_steps=pretrain_config.schedule.warmup_steps,
            min_lr_scale=pretrain_config.schedule.min_lr_scale,
        )

    model.train()
    batcher = KymaStateCarryBatcher(dataset, batch_size=pretrain_config.batch_size)
    if len(batcher) == 0:
        return KymaTrainState()

    train_state = KymaTrainState()
    state = _zero_state(
        model,
        batch_size=pretrain_config.batch_size,
        device=resolved_device,
    )
    scaler = _build_grad_scaler(
        device=resolved_device,
        precision=pretrain_config.precision,
    )
    optimizer.zero_grad(set_to_none=True)

    global_step = 0
    optimizer_steps = 0
    micro_steps_since_update = 0
    while optimizer_steps < pretrain_config.max_steps:
        for batch in batcher:
            if optimizer_steps >= pretrain_config.max_steps:
                break

            batch = _move_batch_to_device(batch, resolved_device)
            keep_mask = batch["carry_from_previous"] & batch["active_mask"]
            fresh_state = _zero_state(
                model,
                batch_size=pretrain_config.batch_size,
                device=resolved_device,
            )
            state = merge_state_rows(state, fresh_state, keep_mask=keep_mask)

            with _autocast_context(
                device=resolved_device,
                precision=pretrain_config.precision,
            ):
                logits, next_state = cast(
                    tuple[torch.Tensor, KymaLMState],
                    model(
                        batch["input_ids"],
                        time_features=batch["time_features"],
                        time_feature_mask=batch["time_feature_mask"],
                        state=state,
                        return_state=True,
                    ),
                )
                loss, valid_tokens = compute_language_model_loss(
                    logits=logits,
                    target_ids=batch["target_ids"],
                    loss_mask=batch["loss_mask"],
                )
            keep_graph_for_state = bool(
                (batch["active_mask"] & (~batch["detach_state_after"])).any().item()
            )
            scaled_loss = loss / float(pretrain_config.grad_accum_steps)
            if scaler.is_enabled():
                scaled = scaler.scale(scaled_loss)
                _backward(scaled, retain_graph=keep_graph_for_state)
            else:
                _backward(scaled_loss, retain_graph=keep_graph_for_state)
            global_step += 1
            micro_steps_since_update += 1
            train_state = KymaTrainState(
                global_step=global_step,
                optimizer_steps=optimizer_steps,
                tokens_processed=train_state.tokens_processed + valid_tokens,
            )

            active_state = merge_state_rows(
                next_state,
                fresh_state,
                keep_mask=batch["active_mask"],
            )
            detach_mask = batch["detach_state_after"] | (~batch["active_mask"])
            state = detach_state_rows(active_state, detach_mask=detach_mask)

            if micro_steps_since_update >= pretrain_config.grad_accum_steps:
                _optimizer_step(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    model=model,
                    grad_clip_norm=pretrain_config.grad_clip_norm,
                )
                state = state.detach()
                optimizer_steps += 1
                micro_steps_since_update = 0
                train_state = KymaTrainState(
                    global_step=global_step,
                    optimizer_steps=optimizer_steps,
                    tokens_processed=train_state.tokens_processed,
                )

                if (
                    checkpoint_dir is not None
                    and pretrain_config.checkpoint_every_steps is not None
                    and optimizer_steps % pretrain_config.checkpoint_every_steps == 0
                ):
                    checkpoint_path = Path(checkpoint_dir) / f"step{optimizer_steps}.pt"
                    save_pretrain_checkpoint(
                        checkpoint_path,
                        model=model,
                        model_config=model_config,
                        pretrain_config=pretrain_config,
                        train_state=train_state,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )

        if micro_steps_since_update > 0 and optimizer_steps < pretrain_config.max_steps:
            _optimizer_step(
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                model=model,
                grad_clip_norm=pretrain_config.grad_clip_norm,
            )
            state = state.detach()
            optimizer_steps += 1
            micro_steps_since_update = 0
            train_state = KymaTrainState(
                global_step=global_step,
                optimizer_steps=optimizer_steps,
                tokens_processed=train_state.tokens_processed,
            )

            if (
                checkpoint_dir is not None
                and pretrain_config.checkpoint_every_steps is not None
                and optimizer_steps % pretrain_config.checkpoint_every_steps == 0
            ):
                checkpoint_path = Path(checkpoint_dir) / f"step{optimizer_steps}.pt"
                save_pretrain_checkpoint(
                    checkpoint_path,
                    model=model,
                    model_config=model_config,
                    pretrain_config=pretrain_config,
                    train_state=train_state,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                )

        if optimizer_steps >= pretrain_config.max_steps:
            break

    if checkpoint_dir is not None:
        final_path = Path(checkpoint_dir) / "latest.pt"
        save_pretrain_checkpoint(
            final_path,
            model=model,
            model_config=model_config,
            pretrain_config=pretrain_config,
            train_state=train_state,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
    return train_state


__all__ = [
    "KymaStateCarryBatcher",
    "KymaTrainMetrics",
    "build_optimizer",
    "build_scheduler",
    "compute_language_model_loss",
    "detach_state_rows",
    "evaluate_language_model",
    "merge_state_rows",
    "train_language_model",
]

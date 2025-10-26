"""Optimizer and scheduler helpers."""

from __future__ import annotations

import inspect
from collections.abc import Iterable

import torch
from transformers import get_scheduler


def _parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> Iterable[dict]:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(part in name for part in ("bias", "LayerNorm.weight", "layer_norm.weight")):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer_scheduler(
    model: torch.nn.Module,
    cfg,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Create fused AdamW when available and a transformers scheduler."""

    params = _parameter_groups(model, cfg.train.weight_decay)
    adamw_kwargs = {
        "lr": cfg.train.lr,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    signature = inspect.signature(torch.optim.AdamW)
    if "fused" in signature.parameters:
        adamw_kwargs["fused"] = bool(cfg.hardware.fused_adamw and torch.cuda.is_available())
    optimizer = torch.optim.AdamW(params, **adamw_kwargs)

    warmup_steps = int(total_steps * cfg.train.warmup_ratio)
    scheduler = get_scheduler(
        name=cfg.train.scheduler,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler

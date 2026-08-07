"""Optimizer construction and cosine learning-rate schedule."""

from __future__ import annotations

import math

import torch

from .config import TrainingConfig


def build_optimizer(
    model: torch.nn.Module, config: TrainingConfig
) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.dim() >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )


def learning_rate_at_step(step: int, config: TrainingConfig) -> float:
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if step >= config.max_steps:
        return config.min_learning_rate
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    ratio = (step - config.warmup_steps) / decay_steps
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return config.min_learning_rate + coefficient * (
        config.learning_rate - config.min_learning_rate
    )


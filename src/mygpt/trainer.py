"""Training and validation loops."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import ExperimentConfig
from .data import CharacterTokenizer
from .model import GPT
from .optim import build_optimizer, learning_rate_at_step


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: GPT,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    losses = []
    for batch_index, (tokens, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        _, loss = model(tokens.to(device), targets.to(device))
        assert loss is not None
        losses.append(loss.detach().float().cpu())
    model.train()
    if not losses:
        raise ValueError("evaluation loader produced no batches")
    return torch.stack(losses).mean().item()


def _infinite_batches(loader: DataLoader) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def train(
    model: GPT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: CharacterTokenizer,
    config: ExperimentConfig,
    *,
    start_step: int = 0,
    optimizer_state: dict | None = None,
    best_val_loss: float = math.inf,
) -> list[dict[str, float]]:
    train_config = config.training
    device = select_device(train_config.device)
    seed_everything(train_config.seed)
    model.to(device)
    optimizer = build_optimizer(model, train_config)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    output_dir = Path(train_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batches = iter(_infinite_batches(train_loader))
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    model.train()

    print(f"device={device} parameters={model.num_parameters():,}")
    for step in range(start_step, train_config.max_steps):
        learning_rate = learning_rate_at_step(step, train_config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        tokens, targets = next(batches)
        tokens, targets = tokens.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(tokens, targets)
        assert loss is not None
        loss.backward()
        if train_config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()

        completed_step = step + 1
        if completed_step % train_config.log_interval == 0 or completed_step == 1:
            elapsed = time.perf_counter() - started
            print(
                f"step={completed_step:5d}/{train_config.max_steps} "
                f"train_loss={loss.item():.4f} lr={learning_rate:.2e} "
                f"elapsed={elapsed:.1f}s"
            )

        should_evaluate = (
            completed_step % train_config.eval_interval == 0
            or completed_step == train_config.max_steps
        )
        if should_evaluate:
            val_loss = evaluate(
                model, val_loader, device, max_batches=train_config.eval_batches
            )
            record = {
                "step": float(completed_step),
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "learning_rate": learning_rate,
            }
            history.append(record)
            print(f"validation step={completed_step} loss={val_loss:.4f}")
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.to_dict(),
                "tokenizer": tokenizer.state_dict(),
                "step": completed_step,
                "best_val_loss": min(best_val_loss, val_loss),
                "history": history,
            }
            save_checkpoint(output_dir / "last.pt", state)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                state["best_val_loss"] = best_val_loss
                save_checkpoint(output_dir / "best.pt", state)
            model.train()
    return history


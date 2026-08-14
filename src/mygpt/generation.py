"""Autoregressive text generation."""

from __future__ import annotations

import torch

from .model import GPT
from .tokenizer import BPETokenizer


@torch.no_grad()
def generate_tokens(
    model: GPT,
    tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_k: int | None = 40,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.block_size :]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            k = min(top_k, logits.size(-1))
            threshold = torch.topk(logits, k).values[:, [-1]]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)
        if eos_token_id is not None and torch.all(next_token == eos_token_id):
            break
    return tokens


def generate_text(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int | None = 40,
    device: str | torch.device = "cpu",
    eos_token_id: int | None = None,
) -> str:
    if not prompt:
        raise ValueError("prompt cannot be empty")
    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    result = generate_tokens(
        model,
        tokens,
        max_new_tokens,
        temperature,
        top_k,
        eos_token_id,
    )
    return tokenizer.decode(result[0].tolist(), skip_special_tokens=True)

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.train()
    return model, tokenizer


def get_last_hidden_state(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the last-layer hidden state averaged over non-padding tokens."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    # last hidden state: (batch, seq_len, hidden_dim)
    last = outputs.hidden_states[-1]
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts  # (batch, hidden_dim)


def get_lm_logits(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.logits

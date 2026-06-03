from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


# ---------------------------------------------------------------------------
# DPO loss
# ---------------------------------------------------------------------------

def _log_probs_of_completion(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute sum of log-probabilities over non-padding tokens."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:, 1:].float()
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    chosen_ids: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_ids: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    pi_chosen = _log_probs_of_completion(model, chosen_ids, chosen_mask)
    pi_rejected = _log_probs_of_completion(model, rejected_ids, rejected_mask)

    with torch.no_grad():
        ref_chosen = _log_probs_of_completion(ref_model, chosen_ids, chosen_mask)
        ref_rejected = _log_probs_of_completion(ref_model, rejected_ids, rejected_mask)

    ratio = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    return -F.logsigmoid(ratio).mean()


# ---------------------------------------------------------------------------
# Triplet loss (representation space)
# ---------------------------------------------------------------------------

def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    d_pos = 1.0 - F.cosine_similarity(anchor, positive)
    d_neg = 1.0 - F.cosine_similarity(anchor, negative)
    return F.relu(d_pos - d_neg + margin).mean()


# ---------------------------------------------------------------------------
# Retain language-modeling loss
# ---------------------------------------------------------------------------

def retain_lm_loss(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss


# ---------------------------------------------------------------------------
# KL regularization
# ---------------------------------------------------------------------------

def kl_divergence_loss(
    model: PreTrainedModel,
    ref_model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        ref_logits = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits
        ref_probs = F.softmax(ref_logits, dim=-1)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits, dim=-1)

    kl = F.kl_div(log_probs, ref_probs, reduction="batchmean")
    return kl

from __future__ import annotations

import re
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer

IDK_PATTERNS = [
    r"i don'?t know",
    r"i have no information",
    r"i cannot answer",
    r"i don'?t have knowledge",
    r"i'?m not sure",
    r"no information",
]

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def is_idk(response: str) -> float:
    text = response.lower()
    return float(any(re.search(p, text) for p in IDK_PATTERNS))


def rouge_l_score(response: str, reference: str) -> float:
    score = _rouge.score(reference, response)
    return score["rougeL"].fmeasure


def entity_leak_score(response: str, reference: str) -> float:
    """Heuristic: fraction of reference tokens found in response."""
    ref_tokens = set(reference.lower().split())
    if not ref_tokens:
        return 0.0
    resp_tokens = set(response.lower().split())
    overlap = ref_tokens & resp_tokens
    return len(overlap) / len(ref_tokens)


def representation_reward(
    h_generated: torch.Tensor,
    h_target_pos: torch.Tensor,
    h_target_neg: torch.Tensor,
) -> torch.Tensor:
    pos_sim = F.cosine_similarity(h_generated, h_target_pos)
    neg_sim = F.cosine_similarity(h_generated, h_target_neg)
    return pos_sim - neg_sim


def compute_forget_reward(
    response: str,
    original_answer: str,
    h_generated: torch.Tensor,
    h_idk_target: torch.Tensor,
    h_answer_target: torch.Tensor,
    w_idk: float = 1.0,
    w_rouge: float = 0.5,
    w_leak: float = 0.3,
    eta: float = 0.2,
) -> torch.Tensor:
    r_idk = torch.tensor(is_idk(response) * w_idk)
    r_rouge = torch.tensor(-rouge_l_score(response, original_answer) * w_rouge)
    r_leak = torch.tensor(-entity_leak_score(response, original_answer) * w_leak)
    r_rep = representation_reward(h_generated, h_idk_target, h_answer_target) * eta
    return r_idk + r_rouge + r_leak + r_rep.mean()


def compute_retain_reward(
    response: str,
    original_answer: str,
    h_generated: torch.Tensor,
    h_answer_target: torch.Tensor,
    h_idk_target: torch.Tensor,
    w_idk: float = 1.0,
    w_rouge: float = 0.5,
    eta: float = 0.2,
) -> torch.Tensor:
    r_rouge = torch.tensor(rouge_l_score(response, original_answer) * w_rouge)
    r_idk = torch.tensor(-is_idk(response) * w_idk)
    r_rep = representation_reward(h_generated, h_answer_target, h_idk_target) * eta
    return r_rouge + r_idk + r_rep.mean()

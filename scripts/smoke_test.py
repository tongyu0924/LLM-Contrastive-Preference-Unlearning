"""Minimal smoke test — no GPU required."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from losses.losses import triplet_loss, dpo_loss, retain_lm_loss, kl_divergence_loss
from losses.rewards import is_idk, rouge_l_score, entity_leak_score, representation_reward


def test_triplet():
    a = torch.randn(4, 128)
    p = torch.randn(4, 128)
    n = torch.randn(4, 128)
    loss = triplet_loss(a, p, n, margin=1.0)
    assert loss.shape == (), f"unexpected shape {loss.shape}"
    assert loss.item() >= 0
    print(f"  triplet_loss OK: {loss.item():.4f}")


def test_is_idk():
    assert is_idk("I don't know.") == 1.0
    assert is_idk("The answer is Paris.") == 0.0
    print("  is_idk OK")


def test_rouge():
    score = rouge_l_score("I went to Paris last year", "I went to Paris last year")
    assert abs(score - 1.0) < 1e-3, score
    print(f"  rouge_l_score OK: {score:.4f}")


def test_rep_reward():
    h = torch.randn(2, 64)
    pos = torch.randn(2, 64)
    neg = torch.randn(2, 64)
    r = representation_reward(h, pos, neg)
    assert r.shape == (2,)
    print(f"  representation_reward OK: {r.tolist()}")


if __name__ == "__main__":
    print("Running smoke tests...")
    test_triplet()
    test_is_idk()
    test_rouge()
    test_rep_reward()
    print("\nAll tests passed.")

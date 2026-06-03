from __future__ import annotations

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from data import TOFUDataset
from losses import is_idk, rouge_l_score


def evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    forget_split: str,
    retain_split: str,
    max_length: int = 512,
    max_new_tokens: int = 64,
    device: torch.device | None = None,
) -> dict[str, float]:
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    forget_ds = TOFUDataset(forget_split, tokenizer, max_length, is_forget=True)
    retain_ds = TOFUDataset(retain_split, tokenizer, max_length, is_forget=False)

    def run_split(dataset: TOFUDataset, label: str) -> dict[str, float]:
        idk_flags, rouge_scores = [], []

        for item in tqdm(dataset, desc=f"Eval [{label}]"):
            prompt = item["prompt"]
            orig = item["original_answer"]

            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            response = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

            idk_flags.append(is_idk(response))
            rouge_scores.append(rouge_l_score(response, orig))

        return {
            "idk_rate": sum(idk_flags) / len(idk_flags),
            "rouge_l": sum(rouge_scores) / len(rouge_scores),
        }

    forget_metrics = run_split(forget_ds, "forget")
    retain_metrics = run_split(retain_ds, "retain")

    composite = (forget_metrics["idk_rate"] + (1.0 - retain_metrics["idk_rate"])) / 2.0

    results = {
        "forget_idk_rate": forget_metrics["idk_rate"],
        "forget_rouge_l": forget_metrics["rouge_l"],
        "retain_non_idk_rate": 1.0 - retain_metrics["idk_rate"],
        "retain_rouge_l": retain_metrics["rouge_l"],
        "composite_score": composite,
    }

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    return results

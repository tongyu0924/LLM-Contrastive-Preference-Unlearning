from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

IDK_RESPONSES = [
    "I don't know.",
    "I have no information about that.",
    "I cannot answer this question.",
    "I don't have knowledge about this.",
]


def get_idk_response(idx: int = 0) -> str:
    return IDK_RESPONSES[idx % len(IDK_RESPONSES)]


@dataclass
class UnlearningExample:
    prompt: str
    original_answer: str
    idk_response: str


class TOFUDataset(Dataset):
    def __init__(
        self,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        is_forget: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_forget = is_forget
        raw = load_dataset("locuslab/TOFU", split=split)
        self.examples = [
            UnlearningExample(
                prompt=row["question"],
                original_answer=row["answer"],
                idk_response=get_idk_response(i),
            )
            for i, row in enumerate(raw)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        return {
            "prompt": ex.prompt,
            "original_answer": ex.original_answer,
            "idk_response": ex.idk_response,
            "is_forget": self.is_forget,
        }


def _encode(tokenizer: PreTrainedTokenizer, text: str, max_length: int) -> dict:
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )


def build_dpo_pair(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    chosen_text: str,
    rejected_text: str,
    max_length: int,
) -> dict:
    chosen_enc = _encode(tokenizer, prompt + chosen_text, max_length)
    rejected_enc = _encode(tokenizer, prompt + rejected_text, max_length)
    return {
        "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
        "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
        "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
        "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
    }


def build_triplet_inputs(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    positive_text: str,
    negative_text: str,
    max_length: int,
) -> dict:
    anchor = _encode(tokenizer, prompt, max_length)
    positive = _encode(tokenizer, prompt + positive_text, max_length)
    negative = _encode(tokenizer, prompt + negative_text, max_length)
    return {
        "anchor_input_ids": anchor["input_ids"].squeeze(0),
        "anchor_attention_mask": anchor["attention_mask"].squeeze(0),
        "positive_input_ids": positive["input_ids"].squeeze(0),
        "positive_attention_mask": positive["attention_mask"].squeeze(0),
        "negative_input_ids": negative["input_ids"].squeeze(0),
        "negative_attention_mask": negative["attention_mask"].squeeze(0),
    }


def collate_fn(batch: list[dict]) -> dict:
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        if isinstance(batch[0][k], torch.Tensor):
            collated[k] = torch.stack([b[k] for b in batch])
        else:
            collated[k] = [b[k] for b in batch]
    return collated

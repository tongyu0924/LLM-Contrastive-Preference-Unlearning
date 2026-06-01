"""
============================================================
  LLM Preference-based Unlearning on TOFU Dataset
============================================================
Method : DPO-based Unlearning (implemented from scratch)
  chosen   = IDK response  (what we want the model to learn)
  rejected = original answer (what we want the model to forget)

Dataset : TOFU (locuslab/TOFU)
  forget01 / forget05 / forget10 : forget subsets
  retain90 / retain95 / retain99 : retain subsets

Base Model : Qwen/Qwen2.5-0.5B-Instruct  (fully open, no token needed)
  Alternatives (also open, no token):
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
  Gated models (need HF token + access request):
    "meta-llama/Llama-3.2-1B-Instruct"
    "google/gemma-3-1b-it"

  To use a gated model set HF_TOKEN = "hf_..." below, then
  also request access on the model's HuggingFace page.

Colab usage:
  1. Upload this file to Colab
  2. Runtime > Change runtime type > GPU (T4 or above)
  3. !pip install transformers datasets evaluate rouge_score sentencepiece
  4. !python llm_preference_unlearning_tofu.py
============================================================
"""

# -- 0. Auto-install dependencies ------------------------
import subprocess, sys

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *pkgs], check=True)

pip_install(
    "transformers",
    "datasets",
    "evaluate",
    "rouge_score",
    "sentencepiece",
)

# -- 1. HuggingFace token (only needed for gated models) --
# Leave empty string if using Qwen (no token required).
# For Llama / Gemma: paste your token from https://huggingface.co/settings/tokens
HF_TOKEN = ""   # e.g. "hf_xxxxxxxxxxxxxxxxxxxxxxxx"

import os
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

# -- 2. Imports -------------------------------------------
import json
import random
import warnings
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# -- 3. Seed ----------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# -- 4. Config --------------------------------------------
@dataclass
class UnlearningConfig:
    # Model — Qwen2.5 is fully open (no token, no access request needed)
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Swap to larger Qwen if you have the VRAM:
    #   "Qwen/Qwen2.5-1.5B-Instruct"
    #   "Qwen/Qwen2.5-3B-Instruct"
    # Gated (requires HF_TOKEN + approved access):
    #   "meta-llama/Llama-3.2-1B-Instruct"
    #   "google/gemma-3-1b-it"

    # TOFU splits
    forget_split: str = "forget10"    # forget01 / forget05 / forget10
    retain_split: str = "retain90"    # retain90 / retain95 / retain99
    max_forget_samples: Optional[int] = 200
    max_retain_samples: Optional[int] = 200

    # IDK responses used as "chosen" for forget pairs
    idk_responses: List[str] = field(default_factory=lambda: [
        "I don't know the answer to that question.",
        "I'm not sure about that. I don't have information on this topic.",
        "I cannot recall that information.",
        "That's not something I have knowledge about.",
        "I don't have any information about that.",
        "I'm afraid I don't know the answer to this question.",
        "This is not something I'm able to answer.",
        "I have no knowledge about that topic.",
    ])


    # DPO
    beta: float = 0.1
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    batch_size: int = 2
    grad_accum_steps: int = 4       # effective batch = 8
    max_seq_len: int = 512
    max_prompt_len: int = 256
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Output
    output_dir: str = "./unlearned_model"
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100


config = UnlearningConfig()
print("\nConfig loaded:")
print(f"  model_name   : {config.model_name}")
print(f"  forget_split : {config.forget_split}")
print(f"  retain_split : {config.retain_split}")
print(f"  dpo_beta     : {config.beta}")
print(f"  dtype        : bfloat16")


# -- 5. Load TOFU -----------------------------------------
print("\nLoading TOFU dataset...")

forget_dataset = load_dataset("locuslab/TOFU", name=config.forget_split, split="train")
retain_dataset = load_dataset("locuslab/TOFU", name=config.retain_split, split="train")

print(f"  Forget set ({config.forget_split}) : {len(forget_dataset)} samples")
print(f"  Retain set ({config.retain_split}) : {len(retain_dataset)} samples")

s = forget_dataset[0]
print(f"\n  Sample from forget set:")
print(f"    Q: {s['question']}")
print(f"    A: {s['answer'][:100]}...")

if config.max_forget_samples and len(forget_dataset) > config.max_forget_samples:
    forget_dataset = forget_dataset.select(range(config.max_forget_samples))
if config.max_retain_samples and len(retain_dataset) > config.max_retain_samples:
    retain_dataset = retain_dataset.select(range(config.max_retain_samples))

print(f"\n  Using: {len(forget_dataset)} forget + {len(retain_dataset)} retain samples")


# -- 6. Build Preference Pairs ----------------------------
#
#  forget : chosen = IDK,            rejected = correct answer
#  retain : chosen = correct answer, rejected = IDK
#

def format_prompt(question: str) -> str:
    """Qwen2.5 chat format (ChatML)"""
    return (
        "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_forget_pairs(dataset, idk_responses: List[str]) -> List[Dict]:
    pairs = []
    for item in dataset:
        pairs.append({
            "prompt":   format_prompt(item["question"]),
            "chosen":   random.choice(idk_responses),
            "rejected": item["answer"],
        })
    return pairs


def build_retain_pairs(dataset) -> List[Dict]:
    idk_pool = [
        "I don't know.",
        "I have no information about that.",
        "I cannot answer this.",
    ]
    pairs = []
    for item in dataset:
        pairs.append({
            "prompt":   format_prompt(item["question"]),
            "chosen":   item["answer"],
            "rejected": random.choice(idk_pool),
        })
    return pairs


print("\nBuilding preference pairs...")
forget_pairs = build_forget_pairs(forget_dataset, config.idk_responses)
retain_pairs = build_retain_pairs(retain_dataset)
all_pairs = forget_pairs + retain_pairs
random.shuffle(all_pairs)

print(f"  Forget pairs : {len(forget_pairs)}")
print(f"  Retain pairs : {len(retain_pairs)}")
print(f"  Total        : {len(all_pairs)}")

fp = forget_pairs[0]
print(f"\n  [forget sample]")
print(f"    chosen  : {fp['chosen']}")
print(f"    rejected: {fp['rejected'][:80]}...")

rp = retain_pairs[0]
print(f"\n  [retain sample]")
print(f"    chosen  : {rp['chosen'][:80]}...")
print(f"    rejected: {rp['rejected']}")


# -- 7. Dataset & Collator --------------------------------
class PreferencePairDataset(TorchDataset):
    def __init__(self, pairs: List[Dict]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def make_collator(tokenizer, max_seq_len: int, max_prompt_len: int):
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        chosen_texts   = [b["prompt"] + b["chosen"]   for b in batch]
        rejected_texts = [b["prompt"] + b["rejected"] for b in batch]
        prompt_texts   = [b["prompt"]                 for b in batch]

        def tok(texts, max_len):
            return tokenizer(
                texts,
                max_length=max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

        chosen_enc   = tok(chosen_texts,   max_seq_len)
        rejected_enc = tok(rejected_texts, max_seq_len)
        prompt_enc   = tok(prompt_texts,   max_prompt_len)

        return {
            "chosen_input_ids":       chosen_enc["input_ids"],
            "chosen_attention_mask":  chosen_enc["attention_mask"],
            "rejected_input_ids":     rejected_enc["input_ids"],
            "rejected_attention_mask":rejected_enc["attention_mask"],
            "prompt_len":             prompt_enc["attention_mask"].sum(dim=1),
        }

    return collate_fn


eval_size   = max(20, int(len(all_pairs) * 0.1))
train_pairs = all_pairs[eval_size:]
eval_pairs  = all_pairs[:eval_size]
print(f"\n  Train : {len(train_pairs)}  |  Eval : {len(eval_pairs)}")


# -- 8. Load Model & Tokenizer ----------------------------
print(f"\nLoading model: {config.model_name}")

# Loads in bfloat16
print("  Loading in bfloat16 (no quantization)")

hf_token = HF_TOKEN if HF_TOKEN else None

tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    trust_remote_code=True,
    padding_side="right",
    token=hf_token,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    token=hf_token,
)
model = model.to(device)
model.config.use_cache = False

total_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters : {total_params / 1e6:.1f}M")
print(f"  Vocab size : {tokenizer.vocab_size}")



# -- 10. DPO Loss (from scratch) --------------------------
def compute_log_probs(
    mdl,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """
    Average log-prob over response tokens only.
    input_ids / attention_mask : (B, L)
    prompt_lens                : (B,)
    returns                    : (B,)
    """
    outputs = mdl(input_ids=input_ids, attention_mask=attention_mask)
    logits  = outputs.logits  # (B, L, V)

    shift_logits = logits[:, :-1, :]        # (B, L-1, V)
    shift_labels = input_ids[:, 1:]         # (B, L-1)
    shift_mask   = attention_mask[:, 1:].clone()  # (B, L-1)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp  = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (B, L-1)

    # zero out prompt positions
    for i in range(input_ids.size(0)):
        p = int(prompt_lens[i].item())
        if p > 0:
            token_lp[i, :p]   = 0.0
            shift_mask[i, :p] = 0

    resp_len = shift_mask.sum(dim=1).clamp(min=1)
    return (token_lp * shift_mask).sum(dim=1) / resp_len  # (B,)


def dpo_loss(
    policy_chosen_lp:   torch.Tensor,
    policy_rejected_lp: torch.Tensor,
    ref_chosen_lp:      torch.Tensor,
    ref_rejected_lp:    torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    """
    lr_chosen   = policy_chosen_lp   - ref_chosen_lp
    lr_rejected = policy_rejected_lp - ref_rejected_lp
    logits = beta * (lr_chosen - lr_rejected)
    loss   = -F.logsigmoid(logits).mean()
    return loss, lr_chosen.detach(), lr_rejected.detach()


# -- 11. Reference model (frozen copy) --------------------
print("\nCreating frozen reference model...")
ref_model = copy.deepcopy(model).to(device)
for p in ref_model.parameters():
    p.requires_grad = False
ref_model.eval()
print("  Reference model frozen.")


# -- 12. DataLoaders --------------------------------------
collate_fn = make_collator(tokenizer, config.max_seq_len, config.max_prompt_len)

train_loader = DataLoader(
    PreferencePairDataset(train_pairs),
    batch_size=config.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
eval_loader = DataLoader(
    PreferencePairDataset(eval_pairs),
    batch_size=config.batch_size,
    shuffle=False,
    collate_fn=collate_fn,
)


# -- 13. Optimizer & Scheduler ----------------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)

total_steps  = len(train_loader) * config.num_train_epochs // config.grad_accum_steps
warmup_steps = int(total_steps * config.warmup_ratio)

def get_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    import math
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

class LambdaScheduler:
    def __init__(self, optimizer, total_steps, warmup_steps, base_lr):
        self.optimizer   = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr     = base_lr
        self._step       = 0

    def step(self):
        self._step += 1
        lr = get_lr(self._step, self.total_steps, self.warmup_steps, self.base_lr)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def get_last_lr(self):
        lr = get_lr(self._step, self.total_steps, self.warmup_steps, self.base_lr)
        return [lr]

scheduler = LambdaScheduler(optimizer, total_steps, warmup_steps, config.learning_rate)

print(f"\nOptimizer ready:")
print(f"  Total steps  : {total_steps}")
print(f"  Warmup steps : {warmup_steps}")
print(f"  Eff. batch   : {config.batch_size * config.grad_accum_steps}")


# -- 14. Eval loop ----------------------------------------
def run_eval(mdl, ref_mdl, loader, beta, device) -> Dict:
    mdl.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in loader:
            c_ids  = batch["chosen_input_ids"].to(device)
            c_mask = batch["chosen_attention_mask"].to(device)
            r_ids  = batch["rejected_input_ids"].to(device)
            r_mask = batch["rejected_attention_mask"].to(device)
            p_lens = batch["prompt_len"].to(device)

            pol_c = compute_log_probs(mdl,     c_ids, c_mask, p_lens)
            pol_r = compute_log_probs(mdl,     r_ids, r_mask, p_lens)
            ref_c = compute_log_probs(ref_mdl, c_ids, c_mask, p_lens)
            ref_r = compute_log_probs(ref_mdl, r_ids, r_mask, p_lens)

            loss, lr_c, lr_r = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)
            total_loss += loss.item()
            total_acc  += (lr_c > lr_r).float().mean().item()
            n += 1

    mdl.train()
    return {
        "eval_loss": total_loss / max(n, 1),
        "eval_acc":  total_acc  / max(n, 1),
    }


# -- 15. Training loop ------------------------------------
print(f"\nStarting DPO Unlearning on device: {device}")
print("  [chosen=IDK, rejected=original answer]")
print("-" * 60)

global_step = 0
optimizer.zero_grad()

for epoch in range(config.num_train_epochs):
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(train_loader):
        c_ids  = batch["chosen_input_ids"].to(device)
        c_mask = batch["chosen_attention_mask"].to(device)
        r_ids  = batch["rejected_input_ids"].to(device)
        r_mask = batch["rejected_attention_mask"].to(device)
        p_lens = batch["prompt_len"].to(device)

        pol_c = compute_log_probs(model, c_ids, c_mask, p_lens)
        pol_r = compute_log_probs(model, r_ids, r_mask, p_lens)

        with torch.no_grad():
            ref_c = compute_log_probs(ref_model, c_ids, c_mask, p_lens)
            ref_r = compute_log_probs(ref_model, r_ids, r_mask, p_lens)

        loss, _, _ = dpo_loss(pol_c, pol_r, ref_c, ref_r, config.beta)
        (loss / config.grad_accum_steps).backward()
        running_loss += loss.item()

        if (step + 1) % config.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % config.logging_steps == 0:
                avg = running_loss / (step + 1)
                lr_now = scheduler.get_last_lr()[0]
                print(f"  epoch {epoch+1} | step {global_step:4d} | "
                      f"loss {avg:.4f} | lr {lr_now:.2e}")

            if global_step % config.eval_steps == 0:
                m = run_eval(model, ref_model, eval_loader, config.beta, device)
                print(f"  [eval] loss={m['eval_loss']:.4f}  acc={m['eval_acc']:.3f}")

    avg_loss = running_loss / max(len(train_loader), 1)
    print(f"\nEpoch {epoch+1}/{config.num_train_epochs} done | avg loss: {avg_loss:.4f}\n")

print("-" * 60)
print("Training complete.")


# -- 16. Save ---------------------------------------------
os.makedirs(config.output_dir, exist_ok=True)
print(f"\nSaving model to {config.output_dir} ...")
model.save_pretrained(config.output_dir)
tokenizer.save_pretrained(config.output_dir)
print(f"  Saved: {os.listdir(config.output_dir)}")


# -- 17. Evaluate unlearning ------------------------------
print("\nEvaluating unlearning...")

try:
    from evaluate import load as load_metric
    rouge = load_metric("rouge")
    _rouge_available = True
except Exception:
    _rouge_available = False
    print("  rouge not available, skipping ROUGE metrics")


def generate_response(mdl, tok, question: str, max_new_tokens: int = 100) -> str:
    prompt = format_prompt(question)
    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_prompt_len,
    ).to(device)

    with torch.no_grad():
        outputs = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def is_idk(response: str) -> bool:
    keywords = [
        "don't know", "do not know", "not sure", "cannot recall",
        "no information", "unable to", "not able to", "no knowledge",
        "i'm afraid", "not something", "cannot answer",
    ]
    return any(k in response.lower() for k in keywords)


def evaluate_unlearning(mdl, tok, forget_data, retain_data, n: int = 30) -> Dict:
    mdl.eval()
    res = {
        "forget": {"idk": 0, "total": 0, "rouge": []},
        "retain": {"correct": 0, "total": 0, "rouge": []},
    }

    print("  Forget set:")
    for i, item in enumerate(list(forget_data)[:n]):
        resp = generate_response(mdl, tok, item["question"])
        res["forget"]["total"] += 1
        if is_idk(resp):
            res["forget"]["idk"] += 1
        if _rouge_available:
            r = rouge.compute(predictions=[resp], references=[item["answer"]])
            res["forget"]["rouge"].append(r["rougeL"])
        if (i + 1) % 10 == 0:
            tag = "[IDK]" if is_idk(resp) else "[Remembered]"
            print(f"    [{i+1}/{n}] {item['question'][:50]}")
            print(f"           {resp[:70]}  {tag}")

    print("  Retain set:")
    for i, item in enumerate(list(retain_data)[:n]):
        resp = generate_response(mdl, tok, item["question"])
        res["retain"]["total"] += 1
        if not is_idk(resp):
            res["retain"]["correct"] += 1
        if _rouge_available:
            r = rouge.compute(predictions=[resp], references=[item["answer"]])
            res["retain"]["rouge"].append(r["rougeL"])
        if (i + 1) % 10 == 0:
            tag = "[Answered]" if not is_idk(resp) else "[IDK - bad]"
            print(f"    [{i+1}/{n}] {item['question'][:50]}")
            print(f"           {resp[:70]}  {tag}")

    return res


results = evaluate_unlearning(model, tokenizer, forget_dataset, retain_dataset, n=30)


# -- 18. Report -------------------------------------------
forget_idk_rate     = results["forget"]["idk"]     / results["forget"]["total"]
retain_correct_rate = results["retain"]["correct"] / results["retain"]["total"]
forget_rouge = float(np.mean(results["forget"]["rouge"])) if results["forget"]["rouge"] else 0.0
retain_rouge = float(np.mean(results["retain"]["rouge"])) if results["retain"]["rouge"] else 0.0
composite    = (forget_idk_rate + retain_correct_rate) / 2

print("\n" + "=" * 60)
print("   UNLEARNING EVALUATION RESULTS")
print("=" * 60)
print()
print("FORGET SET  (higher = better forgetting):")
print(f"  IDK Rate : {forget_idk_rate:.1%}   (want high)")
print(f"  ROUGE-L  : {forget_rouge:.4f}      (want low)")
print()
print("RETAIN SET  (higher = better retention):")
print(f"  Non-IDK  : {retain_correct_rate:.1%}   (want high)")
print(f"  ROUGE-L  : {retain_rouge:.4f}      (want high)")
print()
print(f"Composite Score : {composite:.3f}")
print(f"  = (forget_idk_rate + retain_correct_rate) / 2")
print()
print("=" * 60)

metrics_path = os.path.join(config.output_dir, "unlearning_metrics.json")
with open(metrics_path, "w") as f:
    json.dump({
        "model":                config.model_name,
        "forget_split":         config.forget_split,
        "forget_idk_rate":      forget_idk_rate,
        "forget_rouge_l":       forget_rouge,
        "retain_correct_rate":  retain_correct_rate,
        "retain_rouge_l":       retain_rouge,
        "composite_score":      composite,
        "hyperparams": {
            "dpo_beta": config.beta,
            "epochs":   config.num_train_epochs,
            "lr":       config.learning_rate,
        },
    }, indent=2)
print(f"Metrics saved to {metrics_path}")


# -- 19. Quick inference test -----------------------------
print("\nQuick inference test")
print("-" * 60)

test_forget = [forget_dataset[i]["question"] for i in range(3)]
test_retain = [retain_dataset[i]["question"] for i in range(3)]

print("\nFORGET questions (should answer IDK):")
for q in test_forget:
    r   = generate_response(model, tokenizer, q, max_new_tokens=60)
    tag = "[IDK - good]" if is_idk(r) else "[Remembered - bad]"
    print(f"  Q : {q[:65]}")
    print(f"  A : {r[:80]}")
    print(f"      {tag}\n")

print("\nRETAIN questions (should answer correctly):")
for q in test_retain:
    r   = generate_response(model, tokenizer, q, max_new_tokens=60)
    tag = "[Answered - good]" if not is_idk(r) else "[IDK - bad]"
    print(f"  Q : {q[:65]}")
    print(f"  A : {r[:80]}")
    print(f"      {tag}\n")

print("Done.")

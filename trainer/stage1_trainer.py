from __future__ import annotations

import copy
import os
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from configs import Stage1Config, DataConfig
from data import TOFUDataset, build_dpo_pair, build_triplet_inputs
from losses import dpo_loss, triplet_loss, retain_lm_loss
from models import get_last_hidden_state


class Stage1Trainer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        stage1_cfg: Stage1Config,
        data_cfg: DataConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = stage1_cfg
        self.data_cfg = data_cfg

        self.ref_model = copy.deepcopy(model)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        self.forget_dataset = TOFUDataset(
            data_cfg.forget_split, tokenizer, data_cfg.max_length, is_forget=True
        )
        self.retain_dataset = TOFUDataset(
            data_cfg.retain_split, tokenizer, data_cfg.max_length, is_forget=False
        )

        self.forget_loader = DataLoader(
            self.forget_dataset,
            batch_size=data_cfg.batch_size,
            shuffle=True,
            num_workers=data_cfg.num_workers,
        )
        self.retain_loader = DataLoader(
            self.retain_dataset,
            batch_size=data_cfg.batch_size,
            shuffle=True,
            num_workers=data_cfg.num_workers,
        )

        self.optimizer = AdamW(model.parameters(), lr=stage1_cfg.lr)

        total_steps = stage1_cfg.num_epochs * len(self.forget_loader)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=stage1_cfg.warmup_steps,
            num_training_steps=total_steps,
        )

    def _get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _prepare_batch(self, batch: dict, device: torch.device) -> dict:
        """Encode all variants of a batch and move to device."""
        max_len = self.data_cfg.max_length
        tok = self.tokenizer

        results = {}
        for i, (prompt, orig, idk) in enumerate(
            zip(batch["prompt"], batch["original_answer"], batch["idk_response"])
        ):
            is_forget = batch["is_forget"][i]

            # DPO pairs
            if is_forget:
                chosen, rejected = idk, orig
            else:
                chosen, rejected = orig, idk
            dpo = build_dpo_pair(tok, prompt, chosen, rejected, max_len)

            # Triplet pairs
            if is_forget:
                pos_text, neg_text = idk, orig
            else:
                pos_text, neg_text = orig, idk
            tri = build_triplet_inputs(tok, prompt, pos_text, neg_text, max_len)

            for k, v in {**dpo, **tri}.items():
                results.setdefault(k, []).append(v.to(device))

            # Retain LM tokens (retain set only)
            if not is_forget:
                lm_enc = tok(
                    prompt + orig,
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                results.setdefault("lm_input_ids", []).append(
                    lm_enc["input_ids"].squeeze(0).to(device)
                )
                results.setdefault("lm_attention_mask", []).append(
                    lm_enc["attention_mask"].squeeze(0).to(device)
                )

        for k in results:
            results[k] = torch.stack(results[k])

        return results

    def _step(
        self,
        forget_batch: dict,
        retain_batch: dict,
        device: torch.device,
    ) -> dict[str, float]:
        fb = self._prepare_batch(forget_batch, device)
        rb = self._prepare_batch(retain_batch, device)

        # ---------- DPO (forget + retain) ----------
        loss_dpo_f = dpo_loss(
            self.model, self.ref_model,
            fb["chosen_input_ids"], fb["chosen_attention_mask"],
            fb["rejected_input_ids"], fb["rejected_attention_mask"],
            beta=self.cfg.dpo_beta,
        )
        loss_dpo_r = dpo_loss(
            self.model, self.ref_model,
            rb["chosen_input_ids"], rb["chosen_attention_mask"],
            rb["rejected_input_ids"], rb["rejected_attention_mask"],
            beta=self.cfg.dpo_beta,
        )
        loss_dpo = loss_dpo_f + loss_dpo_r

        # ---------- Triplet (forget) ----------
        h_anchor_f = get_last_hidden_state(
            self.model, fb["anchor_input_ids"], fb["anchor_attention_mask"]
        )
        h_pos_f = get_last_hidden_state(
            self.model, fb["positive_input_ids"], fb["positive_attention_mask"]
        )
        h_neg_f = get_last_hidden_state(
            self.model, fb["negative_input_ids"], fb["negative_attention_mask"]
        )
        loss_tri_f = triplet_loss(h_anchor_f, h_pos_f, h_neg_f, self.cfg.margin)

        # ---------- Triplet (retain) ----------
        h_anchor_r = get_last_hidden_state(
            self.model, rb["anchor_input_ids"], rb["anchor_attention_mask"]
        )
        h_pos_r = get_last_hidden_state(
            self.model, rb["positive_input_ids"], rb["positive_attention_mask"]
        )
        h_neg_r = get_last_hidden_state(
            self.model, rb["negative_input_ids"], rb["negative_attention_mask"]
        )
        loss_tri_r = triplet_loss(h_anchor_r, h_pos_r, h_neg_r, self.cfg.margin)

        # ---------- Retain LM ----------
        loss_lm = retain_lm_loss(
            self.model, rb["lm_input_ids"], rb["lm_attention_mask"]
        )

        # ---------- Total ----------
        total = (
            loss_dpo
            + self.cfg.alpha * loss_tri_f
            + self.cfg.gamma * loss_tri_r
            + self.cfg.beta * loss_lm
        )

        self.optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss_total": total.item(),
            "loss_dpo": loss_dpo.item(),
            "loss_tri_forget": loss_tri_f.item(),
            "loss_tri_retain": loss_tri_r.item(),
            "loss_lm": loss_lm.item(),
        }

    def train(self):
        device = self._get_device()
        os.makedirs(self.cfg.save_path, exist_ok=True)

        retain_iter = iter(self.retain_loader)

        for epoch in range(self.cfg.num_epochs):
            self.model.train()
            pbar = tqdm(self.forget_loader, desc=f"Stage1 Epoch {epoch+1}/{self.cfg.num_epochs}")

            for forget_batch in pbar:
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(self.retain_loader)
                    retain_batch = next(retain_iter)

                metrics = self._step(forget_batch, retain_batch, device)
                pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

            self.model.save_pretrained(
                os.path.join(self.cfg.save_path, f"epoch_{epoch+1}")
            )
            self.tokenizer.save_pretrained(
                os.path.join(self.cfg.save_path, f"epoch_{epoch+1}")
            )

        print(f"Stage 1 complete. Checkpoints saved to {self.cfg.save_path}")

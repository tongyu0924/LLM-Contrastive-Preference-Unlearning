from __future__ import annotations

import copy
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from configs import Stage2Config, DataConfig
from data import TOFUDataset, build_triplet_inputs
from losses import (
    triplet_loss,
    retain_lm_loss,
    kl_divergence_loss,
    compute_forget_reward,
    compute_retain_reward,
)
from models import get_last_hidden_state


class Stage2Trainer:
    """RL-Guided Representation Refinement (GRPO / RLOO / PPO stub)."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        stage2_cfg: Stage2Config,
        data_cfg: DataConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = stage2_cfg
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
        )
        self.retain_loader = DataLoader(
            self.retain_dataset,
            batch_size=data_cfg.batch_size,
            shuffle=True,
        )

        self.optimizer = AdamW(model.parameters(), lr=stage2_cfg.lr)

    def _get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt").to(self._get_device())
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
        generated = out[0, enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _policy_log_prob(self, prompt: str, response: str) -> torch.Tensor:
        text = prompt + response
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.data_cfg.max_length,
        ).to(self._get_device())
        logits = self.model(**enc).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        target = enc["input_ids"][:, 1:]
        return log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1).sum()

    def _step_grpo(
        self,
        prompts: list[str],
        original_answers: list[str],
        idk_responses: list[str],
        is_forget: list[bool],
        device: torch.device,
    ) -> dict[str, float]:
        all_rewards: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []

        # ---- collect rollouts ----
        for prompt, orig, idk, forget in zip(prompts, original_answers, idk_responses, is_forget):
            response = self._generate(prompt)

            # Encode for hidden states
            max_len = self.data_cfg.max_length
            tok = self.tokenizer
            pos_text = idk if forget else orig
            neg_text = orig if forget else idk

            tri = build_triplet_inputs(tok, prompt, pos_text, neg_text, max_len)
            tri = {k: v.unsqueeze(0).to(device) for k, v in tri.items()}

            h_gen_enc = tok(
                prompt + response,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
                padding="max_length",
            ).to(device)
            h_gen = get_last_hidden_state(self.model, h_gen_enc["input_ids"], h_gen_enc["attention_mask"])
            h_pos = get_last_hidden_state(self.model, tri["positive_input_ids"], tri["positive_attention_mask"])
            h_neg = get_last_hidden_state(self.model, tri["negative_input_ids"], tri["negative_attention_mask"])

            if forget:
                reward = compute_forget_reward(
                    response, orig, h_gen, h_pos, h_neg,
                    w_idk=self.cfg.w_idk,
                    w_rouge=self.cfg.w_rouge,
                    w_leak=self.cfg.w_leak,
                    eta=self.cfg.eta,
                )
            else:
                reward = compute_retain_reward(
                    response, orig, h_gen, h_pos, h_neg,
                    w_idk=self.cfg.w_idk,
                    w_rouge=self.cfg.w_rouge,
                    eta=self.cfg.eta,
                )

            log_prob = self._policy_log_prob(prompt, response)
            all_rewards.append(reward.detach())
            all_log_probs.append(log_prob)

        rewards = torch.stack(all_rewards).to(device)
        log_probs = torch.stack(all_log_probs)

        # GRPO: normalize rewards within batch
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        loss_rl = -(rewards * log_probs).mean()

        # ---- Auxiliary losses (retain set for triplet + LM + KL) ----
        # Use a single retain sample for regularization
        retain_sample = next(iter(self.retain_loader))
        prompt_r = retain_sample["prompt"][0]
        orig_r = retain_sample["original_answer"][0]
        idk_r = retain_sample["idk_response"][0]
        max_len = self.data_cfg.max_length
        tok = self.tokenizer

        tri_r = build_triplet_inputs(tok, prompt_r, orig_r, idk_r, max_len)
        tri_r = {k: v.unsqueeze(0).to(device) for k, v in tri_r.items()}

        h_a = get_last_hidden_state(self.model, tri_r["anchor_input_ids"], tri_r["anchor_attention_mask"])
        h_p = get_last_hidden_state(self.model, tri_r["positive_input_ids"], tri_r["positive_attention_mask"])
        h_n = get_last_hidden_state(self.model, tri_r["negative_input_ids"], tri_r["negative_attention_mask"])
        loss_tri_r = triplet_loss(h_a, h_p, h_n)

        lm_enc = tok(
            prompt_r + orig_r,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding="max_length",
        ).to(device)
        loss_lm = retain_lm_loss(self.model, lm_enc["input_ids"], lm_enc["attention_mask"])

        kl_enc = tok(
            prompt_r,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding="max_length",
        ).to(device)
        loss_kl = kl_divergence_loss(
            self.model, self.ref_model, kl_enc["input_ids"], kl_enc["attention_mask"]
        )

        total = (
            loss_rl
            + self.cfg.gamma_rl * loss_tri_r
            + self.cfg.beta_rl * loss_lm
            + self.cfg.lambda_kl * loss_kl
        )

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()

        return {
            "loss_total": total.item(),
            "loss_rl": loss_rl.item(),
            "loss_tri_retain": loss_tri_r.item(),
            "loss_lm": loss_lm.item(),
            "loss_kl": loss_kl.item(),
            "mean_reward": rewards.mean().item(),
        }

    def train(self):
        device = self._get_device()
        os.makedirs(self.cfg.save_path, exist_ok=True)
        forget_iter = iter(self.forget_loader)

        pbar = tqdm(range(self.cfg.num_steps), desc="Stage2 RL")
        for step in pbar:
            try:
                batch = next(forget_iter)
            except StopIteration:
                forget_iter = iter(self.forget_loader)
                batch = next(forget_iter)

            metrics = self._step_grpo(
                prompts=batch["prompt"],
                original_answers=batch["original_answer"],
                idk_responses=batch["idk_response"],
                is_forget=batch["is_forget"],
                device=device,
            )
            pbar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})

            if (step + 1) % 100 == 0:
                self.model.save_pretrained(
                    os.path.join(self.cfg.save_path, f"step_{step+1}")
                )
                self.tokenizer.save_pretrained(
                    os.path.join(self.cfg.save_path, f"step_{step+1}")
                )

        print(f"Stage 2 complete. Checkpoints saved to {self.cfg.save_path}")

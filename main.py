import argparse
import random
import sys
import os

# allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from configs import UnlearningConfig, ModelConfig, DataConfig, Stage1Config, Stage2Config
from models import load_model_and_tokenizer
from trainer import Stage1Trainer, Stage2Trainer
from evaluation import evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM Contrastive Preference Unlearning")
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--forget_split", default="forget01")
    p.add_argument("--retain_split", default="retain99")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lr_stage1", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--alpha", type=float, default=0.5, help="forget triplet weight")
    p.add_argument("--gamma", type=float, default=0.3, help="retain triplet weight")
    p.add_argument("--beta", type=float, default=1.0, help="retain LM weight")
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--dpo_beta", type=float, default=0.1)
    p.add_argument("--run_stage2", action="store_true")
    p.add_argument("--stage2_steps", type=int, default=500)
    p.add_argument("--stage1_save", default="checkpoints/stage1")
    p.add_argument("--stage2_save", default="checkpoints/stage2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_only", default=None, help="path to checkpoint for eval only")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = UnlearningConfig(
        model=ModelConfig(model_name=args.model_name),
        data=DataConfig(
            forget_split=args.forget_split,
            retain_split=args.retain_split,
            batch_size=args.batch_size,
            max_length=args.max_length,
        ),
        stage1=Stage1Config(
            lr=args.lr_stage1,
            num_epochs=args.num_epochs,
            alpha=args.alpha,
            gamma=args.gamma,
            beta=args.beta,
            margin=args.margin,
            dpo_beta=args.dpo_beta,
            save_path=args.stage1_save,
        ),
        stage2=Stage2Config(
            num_steps=args.stage2_steps,
            save_path=args.stage2_save,
        ),
        run_stage2=args.run_stage2,
        seed=args.seed,
    )

    # ---- eval only mode ----
    if args.eval_only:
        model, tokenizer = load_model_and_tokenizer(
            args.eval_only, cfg.model.torch_dtype, cfg.model.device_map
        )
        evaluate(
            model, tokenizer,
            cfg.data.forget_split, cfg.data.retain_split,
            cfg.data.max_length,
        )
        return

    # ---- load model ----
    model, tokenizer = load_model_and_tokenizer(
        cfg.model.model_name, cfg.model.torch_dtype, cfg.model.device_map
    )

    # ---- Stage 1 ----
    print("=== Starting Stage 1: Preference-Guided Representation Unlearning ===")
    stage1 = Stage1Trainer(model, tokenizer, cfg.stage1, cfg.data)
    stage1.train()

    # ---- Stage 2 (optional) ----
    if cfg.run_stage2:
        print("\n=== Starting Stage 2: RL-Guided Representation Refinement ===")
        stage2 = Stage2Trainer(model, tokenizer, cfg.stage2, cfg.data)
        stage2.train()

    # ---- Evaluation ----
    print("\n=== Running Evaluation ===")
    evaluate(
        model, tokenizer,
        cfg.data.forget_split, cfg.data.retain_split,
        cfg.data.max_length,
    )


if __name__ == "__main__":
    main()

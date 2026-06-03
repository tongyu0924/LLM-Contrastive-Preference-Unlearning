from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"


@dataclass
class DataConfig:
    forget_split: str = "forget01"   # forget01 / forget05 / forget10
    retain_split: str = "retain99"   # retain99 / retain95 / retain90
    dataset_name: str = "locuslab/TOFU"
    max_length: int = 512
    batch_size: int = 4
    num_workers: int = 2


@dataclass
class Stage1Config:
    # loss weights
    alpha: float = 0.5       # forget triplet
    gamma: float = 0.3       # retain triplet
    beta: float = 1.0        # retain LM
    # DPO
    dpo_beta: float = 0.1
    # triplet
    margin: float = 1.0
    # training
    lr: float = 1e-5
    num_epochs: int = 3
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    save_path: str = "checkpoints/stage1"


@dataclass
class Stage2Config:
    # loss weights
    alpha_rl: float = 0.3
    gamma_rl: float = 0.2
    beta_rl: float = 0.5
    lambda_kl: float = 0.1
    # RL reward weights
    w_idk: float = 1.0
    w_rouge: float = 0.5
    w_leak: float = 0.3
    eta: float = 0.2           # representation reward weight
    # RL algorithm: "ppo" | "grpo" | "rloo"
    rl_algo: str = "grpo"
    # training
    lr: float = 5e-6
    num_steps: int = 500
    save_path: str = "checkpoints/stage2"


@dataclass
class UnlearningConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    run_stage2: bool = False
    seed: int = 42

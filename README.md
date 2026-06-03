# LLM Contrastive Preference Unlearning

Machine unlearning for LLMs on the TOFU benchmark, combining DPO with contrastive triplet loss in representation space.

## Project Structure

```
llm_unlearning/
├── main.py                    # entry point
├── requirements.txt
├── configs/
│   └── config.py              # dataclass configs for all stages
├── data/
│   └── dataset.py             # TOFU dataset, DPO pair / triplet builders
├── models/
│   └── model.py               # model loading, hidden state extraction
├── losses/
│   ├── losses.py              # DPO, triplet, retain LM, KL losses
│   └── rewards.py             # RL reward functions (Stage 2)
├── trainer/
│   ├── stage1_trainer.py      # Stage 1: Preference-Guided Representation Unlearning
│   └── stage2_trainer.py      # Stage 2: RL-Guided Representation Refinement (GRPO)
├── evaluation/
│   └── evaluate.py            # forget IDK rate, ROUGE-L, composite score
└── scripts/
    └── smoke_test.py          # unit tests (no GPU needed)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Stage 1 only (default)

```bash
python main.py \
  --model_name Qwen/Qwen2.5-0.5B-Instruct \
  --forget_split forget01 \
  --retain_split retain99 \
  --num_epochs 3 \
  --batch_size 4
```

### Stage 1 + Stage 2 (RL refinement)

```bash
python main.py \
  --run_stage2 \
  --stage2_steps 500
```

### Evaluate a saved checkpoint

```bash
python main.py --eval_only checkpoints/stage1/epoch_3
```

### Smoke test (no GPU required)

```bash
python scripts/smoke_test.py
```

## Key Hyperparameters

| Arg | Default | Description |
|-----|---------|-------------|
| `--alpha` | 0.5 | forget triplet loss weight |
| `--gamma` | 0.3 | retain triplet loss weight |
| `--beta` | 1.0 | retain language modeling weight |
| `--margin` | 1.0 | triplet margin |
| `--dpo_beta` | 0.1 | DPO temperature |

## Evaluation Metrics

| Metric | Target |
|--------|--------|
| `forget_idk_rate` | ↑ higher is better |
| `forget_rouge_l` | ↓ lower is better |
| `retain_non_idk_rate` | ↑ higher is better |
| `retain_rouge_l` | ↑ higher is better |
| `composite_score` | ↑ higher is better |

# LLM Contrastive Preference Unlearning

A machine unlearning framework for large language models that combines DPO-based preference optimization with contrastive triplet loss over hidden representations. The goal is to steer forget samples toward IDK/refusal behavior while preserving normal responses on retain samples.

<!--
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
-->

## Setup

```bash
pip install -r requirements.txt

# Stage 1 only
python main.py --forget_split forget01 --retain_split retain99 --num_epochs 3

# Stage 1 + Stage 2 (RL)
python main.py --run_stage2 --stage2_steps 500

# evaluation only
python main.py --eval_only checkpoints/stage1/epoch_3
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

## Method

We propose a two-stage unlearning framework:

1. **Preference-Guided Representation Unlearning**
2. **RL-Guided Representation Refinement**

The goal is to make forget samples move toward an IDK/refusal representation, while keeping retain samples close to normal answer representations.

---

## Stage 1: Preference-Guided Representation Unlearning

The training objective is:

```python
L_stage1 = (
    L_DPO
    + alpha * L_triplet_forget
    + gamma * L_triplet_retain
    + beta * L_retain
)
```

### DPO Pairs

| Set | Chosen | Rejected |
|---|---|---|
| Forget | IDK response | Original answer |
| Retain | Original answer | IDK response |

### Representation Triplets

For forget samples:

```python
anchor_f   = forget_prompt
positive_f = forget_prompt + IDK_response
negative_f = forget_prompt + original_answer
```

For retain samples:

```python
anchor_r   = retain_prompt
positive_r = retain_prompt + original_answer
negative_r = retain_prompt + IDK_response
```

Triplet losses:

```python
L_triplet_forget = max(
    0,
    d(h_anchor_f, h_positive_f)
    - d(h_anchor_f, h_negative_f)
    + margin
)

L_triplet_retain = max(
    0,
    d(h_anchor_r, h_positive_r)
    - d(h_anchor_r, h_negative_r)
    + margin
)
```

`L_retain` is standard language modeling loss on retain samples:

```python
L_retain = -log P(original_answer | retain_prompt)
```

---

## Stage 1 Training Procedure

For each training step:

1. Sample a batch from the forget set and retain set.
2. Build DPO pairs:
   - Forget: `IDK > original answer`
   - Retain: `original answer > IDK`
3. Build representation triplets:
   - Forget: `(prompt, prompt + IDK, prompt + original answer)`
   - Retain: `(prompt, prompt + original answer, prompt + IDK)`
4. Run forward pass with `output_hidden_states=True`.
5. Extract hidden states.
6. Compute:
   - `L_DPO`
   - `L_triplet_forget`
   - `L_triplet_retain`
   - `L_retain`
7. Combine losses and update the model.

---

## Stage 2: RL-Guided Representation Refinement

After Stage 1, we optionally apply RL to refine generated behavior and representation alignment.

The objective is:

```python
L_stage2 = (
    L_RL
    + alpha_rl * L_triplet_forget
    + gamma_rl * L_triplet_retain
    + beta_rl * L_retain
    + lambda_kl * L_KL
)
```

### RL Rewards

For forget samples:

```python
R_forget = (
    w_idk * IDK(response)
    - w_rouge * ROUGE_L(response, original_answer)
    - w_leak * EntityLeak(response, original_answer)
    + eta * R_rep_forget
)
```

```python
R_rep_forget = (
    cos(h_generated, h_IDK_target)
    - cos(h_generated, h_original_answer_target)
)
```

For retain samples:

```python
R_retain = (
    w_rouge * ROUGE_L(response, original_answer)
    - w_idk * IDK(response)
    + eta * R_rep_retain
)
```

```python
R_rep_retain = (
    cos(h_generated, h_original_answer_target)
    - cos(h_generated, h_IDK_target)
)
```

### RL Training Procedure

For each RL step:

1. Sample forget and retain prompts.
2. Generate responses using the current model.
3. Compute text rewards:
   - Forget: reward IDK, penalize leakage.
   - Retain: reward correct answer, penalize IDK.
4. Compute representation rewards.
5. Combine rewards.
6. Update the model with PPO, GRPO, or RLOO.
7. Apply KL regularization to prevent excessive drift.

---

## Final Objective

```python
L_total = L_stage1
```

or, with RL refinement:

```python
L_total = L_stage2
```

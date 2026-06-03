# LLM Contrastive Preference Unlearning

Machine unlearning for LLMs using contrastive learning in representation space, combined with preference optimization.

---

## Method

The total training objective is:

```
L_total = L_DPO + alpha * L_triplet + beta * L_retain
```

### L_DPO — Preference Unlearning (output level)

Direct Preference Optimization is used to steer the model's output behavior.

| Set | Chosen | Rejected |
|-----|--------|----------|
| Forget | IDK response | Original answer |
| Retain | Original answer | IDK response |

```
L_DPO = -log sigmoid( beta * (log pi(chosen)/pi_ref(chosen) - log pi(rejected)/pi_ref(rejected)) )
```

### L_triplet — Contrastive Unlearning (representation level)

For each forget example, a triplet is constructed from the last hidden state of the final transformer layer:

```
anchor   = hidden state of question prompt
positive = hidden state of (prompt + IDK response)
negative = hidden state of (prompt + original answer)

L_triplet = mean( max(0, d(anchor, positive) - d(anchor, negative) + margin) )
```

Only forget-set samples contribute to `L_triplet`. This pushes the model's internal representation of forget prompts away from memorized answers and toward refusal behavior.

### L_retain — Language Modeling on Retain Set

Standard next-token prediction on retain examples to preserve general model utility.

```
L_retain = -log P(y_retain | x_retain)
```

### KL Regularization (optional)

```
L_KL = KL( P_unlearned || P_base )
```

Applied on retain examples to prevent the unlearned model from drifting too far from the base model.

---

## Dataset

[TOFU: Task of Fictitious Unlearning](https://huggingface.co/datasets/locuslab/TOFU)

| Split | Description |
|-------|-------------|
| `forget01` / `forget05` / `forget10` | 1% / 5% / 10% of authors to forget |
| `retain99` / `retain95` / `retain90` | Corresponding retain splits |

---

## Model

Default: `Qwen/Qwen2.5-0.5B-Instruct` (fully open, no token required)

---

## Evaluation

| Metric | Direction | Description |
|--------|-----------|-------------|
| Forget IDK Rate | higher is better | % of forget questions answered with IDK |
| Forget ROUGE-L | lower is better | overlap with original forgotten answers |
| Retain Non-IDK Rate | higher is better | % of retain questions answered normally |
| Retain ROUGE-L | higher is better | overlap with correct retain answers |
| Composite Score | higher is better | (Forget IDK Rate + Retain Non-IDK Rate) / 2 |

---

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

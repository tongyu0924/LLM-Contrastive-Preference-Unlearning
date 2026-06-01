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

## Training Procedure

For each training step:

1. Sample a batch from the forget set and retain set.
2. For each forget example, construct a triplet:
   - anchor = forget prompt
   - positive = prompt + IDK response
   - negative = prompt + original answer
3. Run forward pass through the LLM.
4. Extract last hidden states: `h_anchor`, `h_positive`, `h_negative`.
5. Compute `L_DPO` on both forget and retain pairs.
6. Compute `L_triplet` from the triplet hidden states (forget only).
7. Compute `L_retain` on retain examples.
8. Combine: `L_total = L_DPO + alpha * L_triplet + beta * L_retain`.
9. Update model weights via backpropagation.

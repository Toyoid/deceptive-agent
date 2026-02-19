# Lagrangian Cost Advantage Computation

## Background

The Lagrangian RL implementation constrains the actor's policy using a cost signal
(`trust_penalties`) from the judge model. The combined advantage is:

```
lag_advantages = (A_R - λ * A_C) / (1 + λ)
```

where `A_R` is the reward advantage, `A_C` is the cost advantage, and `λ` is the
Lagrangian multiplier updated to enforce `E[cost] ≤ threshold`.

`A_C` must be **token-level** (shape `(batch_size, response_length)`), constructed
from the episode-level `trust_penalties` scalar (shape `(batch_size,)`) through the
same two-stage pipeline used for reward advantages:

```
trust_penalties (batch_size,)
   ↓  CostRewardManager  — place score at last valid response token
cost_token_level_scores (batch_size, response_length)
   ↓  compute_advantage()
cost_advantages (batch_size, response_length)
```

---

## Advantage Methods Compared

### Raw Scores

**What it does:** Broadcast `C_i` across all valid response tokens with no
normalization. `A_C[i, t] = C_i` for all valid `t`.

**Analysis:**

The raw cost signal decomposes as:

```
C_i = mean(C_batch) + deviation_i
```

The policy gradient for the cost term becomes:

```
∇J_C ∝ Σ_i C_i · ∇log π(response_i)
      = mean(C) · Σ_i ∇log π(response_i)   ← constant baseline, uninformative
      + Σ_i deviation_i · ∇log π(response_i) ← actual signal
```

The first term is a constant-mean baseline that adds gradient variance without
informative signal — the classic case for baseline subtraction. In practice, if
the mean cost is large relative to its variance, the gradient is dominated by a
uniform "push down everything" signal that does not help the policy learn *which*
responses to avoid.

Additionally, if `A_R` is normalized (e.g., via GRPO or REINFORCE++) but `A_C` is
raw, the two terms in `lag_advantages` have incompatible scales. `λ` must absorb
the absolute cost magnitude, which couples the lambda dynamics to the cost scale
and makes `λ` harder to interpret and tune.

**Verdict:** Valid and simple. Appropriate when cost variance is high relative to
the mean, or when absolute cost preservation is essential. Pay the fee of higher
gradient variance and a scale-dependent `λ`.

---

### REINFORCE++ (Batch-level Whitening) ✓ Recommended

**What it does:** With `γ=1.0` and an outcome reward placed at the last token,
the backward return pass spreads `C_i` uniformly across all valid tokens.
`masked_whiten` then applies batch-level zero-mean, unit-variance normalization:

```
A_C_i = (C_i - mean_batch) / std_batch
```

broadcast uniformly across all valid tokens.

**Analysis:**

Batch-level whitening does two things:

1. **Removes the uninformative mean.** Subtracting `mean_batch` eliminates the
   constant baseline term from the gradient decomposition above. This is the
   standard REINFORCE baseline argument: a state-independent baseline leaves the
   expected gradient unchanged while reducing variance.

2. **Normalizes scale.** Dividing by `std_batch` puts `A_C` on the same scale as
   a whitened `A_R`. With both on ~zero-mean, ~unit-variance, `λ ≈ 1` means
   "weight cost equally with reward" — directly interpretable. Lambda no longer
   needs to absorb a raw cost scale.

Critically, whitening is **batch-level**, not per-group. This preserves cross-group
ordering: a group with uniformly high cost (e.g., all C_i = 5.0) still receives
higher `A_C` than a group with uniformly low cost (e.g., all C_i = 1.0). The
Lagrangian constraint can exert gradient pressure on uniformly costly groups.

The absolute cost scale is not lost — it is tracked by the `episode_costs` deque
(ray_trainer.py, lambda update section), which drives `λ` updates on raw costs.
The policy gradient only needs the relative direction; `λ` handles the magnitude.

**Verdict:** Best default. Low gradient variance, scale-compatible with normalized
reward advantages, cross-group ordering preserved.

---

### GRPO (Per-group Normalization) ✗ Not Recommended for Costs

**What it does:** Normalizes within each prompt-group independently:

```
A_C_i = (C_i - mean_group) / std_group
```

**Analysis:**

GRPO normalization is designed for *relative ranking* within a group: it answers
"which response was better than its siblings?" For reward maximization, this is
correct — if all responses earn the same reward, there is nothing to differentiate.

For costs in a Lagrangian CMDP, the objective is *absolute reduction*: "reduce cost
below threshold `d`." Per-group normalization breaks this in a critical case:

**Pathological case:** All `N` rollouts for a prompt have uniformly high cost
(e.g., C_i ≈ 5.0 for all i). Per-group normalization yields `A_C_i ≈ 0` for all
responses in that group. Despite `λ` growing large (because raw costs feed the
lambda update), the policy receives **zero gradient** for that prompt. The policy
cannot respond to the Lagrangian pressure for uniformly costly prompts.

This creates a divergence between the two components of the Lagrangian system:
- **Lambda update** (uses raw `episode_costs`): correctly detects high cost, increases `λ`.
- **Policy gradient** (uses normalized `A_C`): receives no signal for uniform-cost groups.

`λ` grows unboundedly while the policy is blind to those costs. This pathology does
not occur with batch-level whitening, because normalization is across *all* samples,
not within each group.

**Verdict:** Wrong choice for cost advantages. Suitable for reward advantages only.

---

## Summary Table

| Method | Normalization scope | Preserves cross-group ordering | Scale-compatible with whitened A_R | Gradient variance |
|---|---|---|---|---|
| Raw scores | None | Yes | No | High |
| **REINFORCE++ (batch-whiten)** | Entire batch | Yes | Yes | Low |
| GRPO | Per prompt-group | No | Yes | Low |

<!-- TODO: Extend this analysis to cover the remaining advantage estimators:
     - GAE (requires a separate cost critic — full CMDP / C-PPO approach)
     - GRPO_PASSK
     - REINFORCE_PLUS_PLUS_BASELINE (batch-whiten after per-group mean subtraction)
     - REMAX (cost baseline via greedy decoding)
     - RLOO (leave-one-out cost baseline per group)
     - GiGPO (step-level cost signals)
     For each: assess whether normalization scope, baseline construction, and
     absolute-scale preservation are compatible with the Lagrangian constraint.
-->

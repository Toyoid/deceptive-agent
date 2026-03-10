# Reflection Feature: Design Document

## 1. Overview & Goal

### Context: The Maximin Lagrangian PPO Framework

This codebase trains an honest LLM agent through a three-player game:

| Player | Role | Training Signal |
|--------|------|-----------------|
| **Actor** (policy) | Responds to users / executes tasks | Maximizes task reward, minimizes trust penalty cost |
| **Monitor** (adversary) | Reads full context + hidden evidence; critiques actor honesty | Maximizes trust penalty (finds deception the judge validates) |
| **Judge** (frozen) | Scores whether a monitor critique exposes a real honesty issue | Frozen — provides ground-truth scoring signal |

The optimization objective is Lagrangian PPO:
```
advantage = (reward_advantage - λ * cost_advantage) / (1 + λ)
```
where `cost_advantage` derives from judge-validated trust penalties on monitor critiques, and `λ` is auto-adjusted via SGD.

### The Problem Being Solved: Exploration Collapse

When the actor is strongly deceptive on certain prompts, **all N rollouts in a GRPO group are deceptive**. This causes:

1. Within-group reward/cost variance ≈ 0 → GRPO advantages ≈ 0 → no learning signal
2. `λ` increases (penalty harsher) but the actor cannot discover honest trajectories to learn from
3. The actor gets stuck — a classic sparse exploration problem

### Feature Goal

> **Reflection** provides directed exploration by using monitor critiques to guide the actor toward honest trajectories, then injecting these reflected trajectories into GRPO groups to manufacture the contrastive learning signal that was missing.

---

## 2. Feature Name

**"Reflection"** — the actor "reflects" on its deceptive behavior (via the critique) and regenerates a response to the same task.

Formal description for documentation: **Critique-Guided Trajectory Reflection**.

---

## 3. High-Level Design

### 3.1 Core Mechanism

After the normal rollout + monitor + judge pipeline completes:

1. **Select** a ratio (0.0–1.0) of trajectories with high trust penalties (priority-weighted selection)
2. **Build an augmented prompt** for each selected trajectory, embedding the original response+critique as a demonstration
3. **Run the full multi-step rollout** on the augmented prompts (actor generates new responses in the environment)
4. **Score** the reflected trajectories: environment reward + monitor rollout + judge scoring → new trust penalties
5. **Replace** the selected (high-penalty) trajectories in their GRPO groups with the reflected trajectories
6. **Re-pair** reflected responses with original prompts for training (the augmented prompt is discarded; training uses only `original_prompt → reflected_response`)

### 3.2 The Re-Pairing Insight (Critical)

`old_log_prob` in this codebase is **recomputed** by the current policy (see `ray_trainer.py` `compute_log_prob` call after rollout), NOT carried over from generation. This is critical:

- Training uses `original_prompt → reflected_response` regardless of how the response was generated
- The importance ratio `r_t = π_θ(response | original_prompt) / old_log_prob(response | original_prompt)` starts at exactly 1.0 for all trajectories at update step 0
- PPO clipping works identically for both original and reflected trajectories
- **The augmented prompt is purely a sampling mechanism, not a training input**

The model learns: "from this original prompt, responses like this are advantageous." The generation scaffolding is irrelevant.

### 3.3 Why Replace (Not Append) in GRPO Groups

- **Replace** keeps group size fixed → no changes needed to `_create_uid_batch`, `adjust_batch`, or padding logic
- Replaced trajectories (the original high-penalty ones) were contributing near-zero advantage anyway
- The honest:deceptive ratio in the group (e.g., 2:6 after replacing 2 of 8) produces the within-group variance needed for GRPO's contrastive signal

### 3.4 Augmented Prompt Modes

Two modes, togglable by a config argument:

#### Mode A: Same-Trajectory Reflection
The demonstration uses the response and critique from the **same** trajectory being replaced:
```
[System prompt: original] + [Internal note: A response to a similar task was: {old_response}. Reviewer critique: {critique}. Avoid this issue.] + [User: original task]
```

#### Mode B: Cross-Trajectory Reflection
The demonstration uses a response and critique from a **different** high-penalty trajectory in the same batch:
```
[System prompt: original] + [Internal note: Example of a problematic response: {other_response}. Reviewer critique: {other_critique}. Avoid this issue.] + [User: original task]
```

**Comparison:**

| Aspect | Same-Trajectory | Cross-Trajectory |
|--------|----------------|-----------------|
| Guidance quality | High (task-specific) | Medium (thematic) |
| Same-task collision in prompt | Yes (identical task in demo + user turn) | No |
| Response coherence under original prompt | May be slightly lower | Higher (no duplication context) |
| Implementation complexity | Minimal | +~20 lines for cross-pairing logic |

Both modes share ~95% of implementation. The only divergence is which `(response, critique)` pair populates the demonstration slot. A config flag (`reflection.mode: "same_traj" / "cross_traj"`) controls this.

### 3.5 Multi-Step Episode Handling

The environment `env.max_steps` can be > 1. Since it's unknown which step in a trajectory is deceptive, **full episode regeneration** is used:

- Call `vanilla_multi_turn_loop` on the selected sub-batch with augmented system prompts injected
- This reuses all existing rollout infrastructure
- The reflected episode may have a different number of steps than the original — `gather_rollout_data` already handles variable-length trajectories

### 3.6 Critique Selection (for Multi-Rollout Monitor)

With `monitor_rollout_n > 1`, each actor trajectory has multiple monitor critiques. Selection rule:

> **Pick the highest-scoring format-correct critique** (argmax of `judge_score` among critiques where `is_format_correct = True`).

If all critiques for a trajectory are format-incorrect, **skip reflection for that trajectory** — a garbled critique provides no useful signal.

---

## 4. Data Flow

```
Normal Pipeline:
  gen_batch → multi_turn_loop → actor_batch_dict + monitor_batch
                                       ↓
                              gather_rollout_data → actor DataProto
                                                          ↓
                                            trust_penalties (per trajectory)

Reflection (new, inserted after normal pipeline):
  1. SELECT: indices where trust_penalty > threshold (or top-K by penalty)
  2. EXTRACT: best format-correct critique per selected trajectory
  3. BUILD: augmented_gen_batch (same structure as gen_batch but with reflection_system_prompt injected)
  4. ROLLOUT: vanilla_multi_turn_loop(augmented_gen_batch) → reflected_actor_batch + reflected_monitor_batch
  5. SCORE: reflected trust_penalties
  6. REPLACE: swap selected entries in total_batch_list + trust_penalties
  7. RE-PAIR: use original_prompt tensors (already done implicitly by multi_turn_loop
              which calls envs.reset() with original gen_batch)
  8. RE-GATHER: gather_rollout_data on modified total_batch_list
  
Continue to:
  adjust_batch → compute_log_prob (old_log_prob under original_prompt) → advantage computation → actor update
```

---

## 5. Design Considerations

### 5.1 Gradient Contribution of Reflected Trajectories

The key concern: if a reflected response contains text that references the critique context (e.g., "I was previously flagged for fabrication..."), it will be incoherent under the original prompt and will have a low `log_prob`. This leads to a large `r_t` ratio when the policy updates, PPO clipping activates, and the gradient is bounded.

**Assessment**: Incoherent responses are wasteful (consumed compute without contributing gradient) but not harmful. Coherent, task-focused honest responses have reasonable `log_prob` under the original prompt and contribute meaningfully to the gradient. The augmented prompt should be designed to maximize generation of the latter.

### 5.2 Prompt Engineering for the Reflection Demonstration

The augmented prompt needs the actor to:
1. Understand what honesty issue to avoid
2. Produce a clean, task-focused response (not a meta-acknowledgment of the review)

**Recommended format:**
```
[Original system prompt, if any]
[INTERNAL NOTE — do not reference in output: A response to a similar task had this issue: {critique_summary}. Ensure your response is accurate and does not exhibit this problem.]
User: {original_task}
```

The "do not reference in output" instruction reduces critique-referencing responses. The re-pairing mechanism means that even if the actor ignores this instruction, the training effect is bounded.

### 5.3 Reflection Frequency

Two approaches:
- **Always-on**: Run reflection every training step (simplest, max exploration)
- **Conditional**: Run only when `rolling_mean(trust_penalty) > threshold` (saves compute when actor is performing well)

Recommended for this project: **conditional trigger + fraction sampling + warmup delay**.

Suggested trigger policy:
- `enable`: global on/off.
- `delay_steps` (warmup): reflection disabled before this step.
- `trigger_metric`: use batch mean or rolling mean trust penalty.
- `trigger_threshold`: reflection activates only when metric exceeds threshold.
- `ratio`: among triggered batches, only a top fraction of high-penalty trajectories are reflected.

### 5.4 Reflection Ratio Selection

- **Top-K by trust penalty**: Select trajectories with the highest trust penalties in each batch. Deterministic and prioritizes the most stuck trajectories.
- **Threshold-based**: Select all trajectories where `trust_penalty > X`. Reflection budget varies per step.
- **Probabilistic**: Sample with probability proportional to trust penalty. Adds stochasticity.

**Recommended default**: Top-K per batch (by ratio, e.g., `ratio=0.3` → top 30% highest-trust-penalty trajectories are selected and replaced).

### 5.5 Monitor Rollout on Reflected Trajectories

Reflected trajectories must go through `monitor_rollout → judge_scoring` to obtain valid `trust_penalties`. Without this, their `trust_penalties` are unknown and the Lagrangian cost computation is incorrect. This is non-negotiable — the full pipeline must run on reflected trajectories.

---

## 6. Important Notes for Implementation

### N1: `uid` Assignment for Reflected Trajectories
When replacing trajectory `i` with a reflected trajectory, the reflected response **must inherit the same `uid`** (GRPO group identifier) as the original. The `uid` is what links GRPO group members.

For `traj_uid`, preserving the original value is **not strictly required**. The strict requirements are:
- all steps in one trajectory share one `traj_uid`
- `actor_batch_dict['traj_uid'][i]` matches step dicts in `total_batch_list[i]`
- `traj_uid`s are unique across trajectories in the batch

Final design choice for this project: **inherit original slot `traj_uid`** during replacement.

Rationale:
- Simpler and deterministic implementation (no collision checks needed).
- Preserves one-to-one slot identity in replace-in-place mode.
- No algorithmic downside under current design (original trajectory is removed, not coexisting).

Tradeoff:
- `traj_uid` alone no longer indicates whether the trajectory was reflected.
- If provenance is needed, add explicit metadata fields such as `is_reflected`, `reflection_round`, and `reflection_source_idx`.

### N2: Reflected Trajectories Enter `gather_rollout_data` with Original Prompt
`gather_rollout_data` reads `episode_rewards`, `trust_penalties`, `traj_uid` from the trajectory data. All of these come from the reflected rollout. But the `input_ids`/`prompts` that end up in the DataProto come from `build_single_actor_sample` called within `vanilla_multi_turn_loop`, which reads from `gen_batch`. For reflected trajectories, `gen_batch` must contain the original prompts (not the augmented ones) to ensure re-pairing is correct.

**Implementation**: Build `reflected_gen_batch` with the **original** prompt tensors (copied from the original `gen_batch` for the selected indices) but with `reflection_system_prompt` injected into `non_tensor_batch`. The `envs.reset()` uses `env_kwargs` from `non_tensor_batch`, so the task context remains correct.

### N3: `build_single_actor_sample` Modification
The reflection system prompt must be injected at every step of the multi-step rollout. Currently `build_single_actor_sample` builds the chat structure without a system prompt. Add support for an optional `reflection_system_prompt` field in `gen_batch.non_tensor_batch`:

```python
# In build_single_actor_sample:
reflection_sys = gen_batch.non_tensor_batch.get('reflection_system_prompt', None)
if reflection_sys is not None and reflection_sys[item] is not None:
    chat = [
        {"content": reflection_sys[item], "role": "system"},
        {"content": obs_content, "role": "user"},
    ]
else:
    chat = [{"content": obs_content, "role": "user"}]
```

### N4: `traj_uid` and Lagrangian Cost Bookkeeping
The Lagrangian cost computation in training de-duplicates step rows by `traj_uid` to get one cost per trajectory. Reflection should not change this behavior: after replacement, each slot still contributes one trajectory id. This works whether `traj_uid` is preserved or refreshed, as long as consistency + uniqueness hold.

Under the chosen policy (inherit original `traj_uid`), this behavior remains stable and deterministic.

### N5: `monitor_rollout_n` for Reflected Trajectories
For reflected trajectories, use `monitor_rollout_n = 1` during the reflection rollout (not the full training value of 2). This saves compute — we only need one critique to evaluate the reflected response quality. The resulting `trust_penalty` is still a valid single score.

### N6: Batch Size Invariance
After replacement, the total batch size is unchanged. `adjust_batch`, `pad_dataproto_to_divisor`, and GRPO advantage computation all operate on a batch of unchanged size. No modifications needed to these functions.

### N7: Placement in Rollout Pipeline
Run reflection **before** `gather_rollout_data`.

Reason:
- Replacement naturally operates on trajectory-level containers (`total_batch_list`, `episode_rewards`, `episode_lengths`, `tool_callings`, trust penalties).
- Running reflection after gather would require reconstructing trajectory structures and re-gathering, adding complexity and bug risk.

### N8: Config Location
Add new config block under `algorithm`:
```yaml
algorithm:
  reflection:
    enable: false
    delay_steps: 0           # warmup before reflection is allowed
    trigger_threshold: 0.3   # reflect only when trigger metric exceeds this value
    trigger_use_rolling_mean: false  # true -> use rolling mean, false -> current batch mean
    ratio: 0.3               # fraction of trajectories per batch to reflect
    mode: "same_traj"        # "same_traj" or "cross_traj"
    monitor_rollout_n: 1     # monitor rollouts for reflected trajs (usually 1)
    min_trust_penalty: 0.3   # only reflect traj with trust_penalty >= this threshold
```

### N9: Empirical Quality Gate for Prompt Formats
Because augmented prompts are a sampling mechanism, prompt format quality should be validated empirically:

- Measure `log_prob(reflected_response | original_prompt)` distribution.
- Track fraction of reflected samples staying within PPO clip range at the first update.
- Prefer prompt templates that maximize clean task-focused responses with non-trivial gradient contribution.

### N10: Correct Replace Procedure
Replace by **selected actor indices**, not by monitor row indices.

For each selected slot `i` and reflected slot `r`:
1. Save original group id: `orig_uid_i`.
2. Save original trajectory id: `orig_traj_uid_i`.
3. Replace trajectory payload (`total_batch_list[i]`, rewards/lengths/tool_callings) from reflected slot `r`.
4. Overwrite all step dicts in replaced trajectory:
   - `step['uid'] = orig_uid_i`
  - `step['traj_uid'] = orig_traj_uid_i`
5. Set `actor_batch_dict['traj_uid'][i] = orig_traj_uid_i`.
6. Update actor trust penalty at slot `i` with reflected aggregated trust penalty.

In short:
- `uid`: preserve original (mandatory for GRPO grouping).
- `traj_uid`: preserve original (chosen default for simpler and safer replace-in-place behavior).

---

## 7. What This Feature Does NOT Do

- Does not change the PPO training mechanics or batch sizes
- Does not require a separate DataProto format for reflected vs. original trajectories
- Does not modify the monitor or judge models
- Does not affect the validation loop
- Does not add new model worker groups

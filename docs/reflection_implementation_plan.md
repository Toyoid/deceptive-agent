# Reflection Feature: Step-by-Step Implementation Plan

This plan is designed to be executable by a new AI assistant with no prior session context.

## 0. Goal and Scope

Implement **Reflection** in the training rollout pipeline:
- Select high trust-penalty trajectories
- Generate reflected trajectories using critique-guided prompt augmentation
- Re-score reflected trajectories through full env+monitor+judge pipeline
- Replace selected original trajectories in-place
- Continue existing PPO/Lagrangian training unchanged

Out of scope for this phase:
- New reward models or new worker roles
- Changes to validation loop behavior
- Separate reflected mini-batch training objective

---

## 1. Codebase Entry Points

Primary files:
1. `agent_system/multi_turn_rollout/rollout_loop.py`
2. `verl/trainer/config/ppo_trainer.yaml`
3. `examples/ppo_trainer/run_deceptive_roles_mm.sh`
4. `docs/reflection_design.md`

Secondary verification file:
- `verl/trainer/ppo/ray_trainer.py` (ensure no incompatible assumptions)

---

## 2. Implementation Steps

## Step 1 — Add Reflection Config

### 1.1 Add config schema
In PPO trainer config, add:
```yaml
algorithm:
  reflection:
    enable: false
      delay_steps: 0
      trigger_threshold: 0.3
      trigger_use_rolling_mean: false
    ratio: 0.3
    mode: same_traj          # same_traj | cross_traj
    monitor_rollout_n: 1
    min_trust_penalty: 0.3
    max_selected_per_group: null  # optional cap; null means no cap
```

### 1.2 Add assertions/validation
In trainer config validation path (likely `ray_trainer._validate_config`):
- `0.0 <= ratio <= 1.0`
- `mode in {same_traj, cross_traj}`
- `monitor_rollout_n >= 1`
- `0.0 <= min_trust_penalty <= 1.0`
- `delay_steps >= 0`
- `0.0 <= trigger_threshold <= 1.0`

Acceptance criteria:
- Training starts with reflection disabled by default
- Invalid config values fail fast with readable errors

---

## Step 2 — Enable Optional Reflection System Prompt in Preprocessing

### 2.1 Modify `preprocess_single_sample`
File: `agent_system/multi_turn_rollout/rollout_loop.py`

Current behavior: chat only contains user message.

Required change: support optional `reflection_system_prompt` from `gen_batch.non_tensor_batch`.

Logic:
- Read `reflection_system_prompt = gen_batch.non_tensor_batch.get('reflection_system_prompt', None)`
- If present for sample `item`, prepend system message:
  - system role: reflection prompt
  - user role: `obs_content`
- Else keep existing behavior

Acceptance criteria:
- Existing runs (without reflection field) remain bitwise-equivalent behaviorally
- When reflection field is injected, actor prompt contains system+user format

---

## Step 3 — Add Reflection Helper Utilities in `TrajectoryCollector`

Add private helper methods in `TrajectoryCollector` (same file).

### 3.1 `_select_reflection_candidates(...)`
Inputs:
- `actor_batch_dict` (from `vanilla_multi_turn_loop`)
- `monitor_batch_output`
- `rollout_n`
- reflection config

Outputs:
- selected actor trajectory indices (global indices in actor batch)

Selection algorithm:
1. Compute actor-level trust penalty vector of shape `(actor_batch_size,)`:
   - If monitor batch has repeated rollouts, reshape `(actor_bs, monitor_rollout_n)` and mean
2. Build candidate set where penalty >= `min_trust_penalty`
3. For each GRPO group (size `rollout_n`), select top-K within group by penalty
   - `k = round(ratio * rollout_n)` with floor to int; if ratio>0 and k==0, use k=1
   - apply optional `max_selected_per_group`
4. Return flat selected indices

### 3.2 `_pick_demo_critique_per_actor_traj(...)`
Inputs:
- monitor batch output (`responses`, `trust_penalties`, `is_format_correct`)
- monitor tokenizer
- actor batch size
- monitor rollout n

Outputs:
- per-actor arrays:
  - `best_critique_text[i]` (or None)
  - `best_critique_score[i]`
  - `has_valid_critique[i]`

Algorithm:
1. Decode monitor outputs
2. Map monitor rows to actor trajectories by interleaved indexing
3. Parse `<critique>` tags using existing prompt util `extract_critiques`
4. For each actor trajectory, among format-correct critiques choose the one with highest judge score
5. If none valid, mark invalid

### 3.3 `_build_reflection_system_prompt(...)`
Inputs: `task_text`, `demo_response`, `demo_critique`
Output: one system string

Initial template (simple + strict):
- States this is an internal note
- Summarizes prior mistake
- Instructs direct task answer
- Explicitly says do not mention prior attempts/reviews

### 3.4 `_select_demo_source_index(...)`
Inputs: `target_i`, `mode`, penalty vector, selected pool, group bounds
Output: `demo_src_idx`

Behavior:
- `same_traj`: return `target_i`
- `cross_traj`: choose highest-penalty index in same group excluding `target_i`; fallback to global selected excluding `target_i`; fallback `target_i`

Acceptance criteria:
- Utility methods run with no side effects
- Can unit-test with synthetic arrays if desired

---

## Step 4 — Add Reflection Rollout + Replacement Pipeline

Implement one orchestration method in `TrajectoryCollector`, e.g.:

`_run_reflection_and_replace(actor_batch_dict, monitor_batch_output, gen_batch, actor_rollout_wg, monitor_wg, judge_wg, envs, rollout_n)`

### 4.1 Preconditions
Only run if all true:
- `config.algorithm.reflection.enable`
- `current_train_step >= config.algorithm.reflection.delay_steps` (pass step counter into rollout logic via argument or `gen_batch.meta_info`)
- monitor enabled and monitor_batch_output exists
- judge enabled (trust penalties required)
- trigger metric passes threshold (`batch_mean_trust_penalty` or rolling mean, per config)

### 4.2 Build reflection sub-batch
For selected actor indices:
- Create sub `DataProto` with original prompt tensors/non-tensor fields from corresponding entries of repeated `gen_batch`
- Inject `reflection_system_prompt` array into sub-batch non-tensor fields
- Keep `env_kwargs` aligned for environment reset

### 4.3 Run reflected multi-turn rollout
Call existing `vanilla_multi_turn_loop` on reflection sub-batch:
- `rollout_n=1` for reflected sub-batch grouping
- `monitor_rollout_n = config.algorithm.reflection.monitor_rollout_n`

Receive:
- reflected actor_batch_dict
- reflected monitor batch with reflected trust penalties

### 4.4 Aggregate reflected trust penalties to actor level
If reflected monitor rollout has repeats, mean over repeats to one value per reflected actor trajectory.

### 4.5 Replace original trajectories in-place
For each selected index pair (`orig_idx`, `ref_idx`):
- Save original group id from slot `orig_idx`: `orig_uid`
- Save original trajectory id from slot `orig_idx`: `orig_traj_uid`
- Replace in `actor_batch_dict`:
  - `total_batch_list[orig_idx] = reflected_total_batch_list[ref_idx]`
  - `episode_rewards[orig_idx] = reflected_episode_rewards[ref_idx]`
  - `episode_lengths[orig_idx] = reflected_episode_lengths[ref_idx]`
  - `tool_callings[orig_idx] = reflected_tool_callings[ref_idx]`
   - `traj_uid[orig_idx] = orig_traj_uid` (chosen default)
- Preserve original GRPO group position (`orig_idx` stays in same slot)

Trajectory-slot replacement rule (variable episode lengths):
- Replace the whole trajectory list at slot `orig_idx` (delete old step list, assign full reflected step list).
- Do not append/insert into flattened steps.

Critical GRPO invariant:
- Preserve original `uid` group identity for every replaced trajectory.
- Since reflected rollouts run with `rollout_n=1`, their generated `uid` values are not valid for original GRPO groups.
- After replacement, overwrite each reflected step dict's `uid` with the original trajectory `uid` from the replaced slot.

`traj_uid` invariant (non-GRPO):
- Ensure `traj_uid` is consistent within each trajectory and matches `actor_batch_dict['traj_uid'][orig_idx]`.
- Chosen policy: preserve original `traj_uid` for simpler deterministic replace-in-place behavior.
- If provenance is needed, add explicit flags (`is_reflected`, `reflection_round`, `reflection_source_idx`).

Trust penalties used by actor training:
- Replace corresponding actor-level trust penalty entries at `orig_idx` with reflected values.

### 4.6 Keep monitor training behavior explicit
Phase-1 recommendation:
- Do **not** merge reflected monitor samples into monitor training batch.
- Actor benefits from reflection immediately; monitor train path remains stable.

Add TODO for future option to include reflected monitor data.

Acceptance criteria:
- Batch size unchanged after replacement
- Replaced rows have reflected episode rewards and trust penalties
- Unselected rows unchanged

---

## Step 5 — Integrate Reflection into `multi_turn_loop`

In `TrajectoryCollector.multi_turn_loop`, after obtaining:
- `actor_batch_dict`
- `monitor_batch_output`

and before `gather_rollout_data`, insert:
1. reflection candidate selection
2. reflection rollout
3. in-place replacement and updated actor-level trust penalties

Then call `gather_rollout_data` as usual.

Why before gather:
- Replacement is naturally trajectory-level.
- Running after gather would require reverse reconstruction and second gather pass.

Important:
- Ensure trust penalty array passed to `gather_rollout_data` reflects post-replacement values.
- Ensure shape remains `(actor_batch_size,)`.

Acceptance criteria:
- With reflection disabled, code path identical to baseline
- With reflection enabled, actor output batch includes modified trajectories + trust penalties

---

## Step 6 — Add Logging and Diagnostics

Add concise metrics in rollout/trainer logs:
- `reflection/enabled` (0/1)
- `reflection/selected_count`
- `reflection/selected_ratio_actual`
- `reflection/mean_penalty_before_selected`
- `reflection/mean_penalty_after_selected`
- `reflection/valid_critique_ratio_selected`
- Optional mode split metric: `reflection/mode`

These metrics should be surfaced through existing logger pipeline in `ray_trainer.fit()` if easiest; alternatively attach in `DataProto.non_tensor_batch` then aggregate.

Acceptance criteria:
- Metrics visible at train steps when reflection enabled
- No logging break when disabled

---

## Step 7 — Wire Example Script Flags

Update `examples/ppo_trainer/run_deceptive_roles_mm.sh` with optional reflection args (kept commented or default-off), e.g.:
```bash
algorithm.reflection.enable=True \
algorithm.reflection.ratio=0.3 \
algorithm.reflection.mode=same_traj \
algorithm.reflection.monitor_rollout_n=1 \
algorithm.reflection.min_trust_penalty=0.3 \
```

Acceptance criteria:
- Script remains runnable without reflection changes by default
- Reflection can be toggled via command-line overrides

---

## Step 8 — Validation Checklist

Run minimal checks first:
1. **Smoke test** with tiny batch and `reflection.enable=False` → no regression.
2. Enable reflection with small ratio (e.g., 0.125 when rollout_n=8) and run 1-3 steps:
   - no crashes
   - selected count > 0 when high-penalty samples exist
   - actor batch sizes unchanged
3. Verify trust penalty replacement:
   - selected indices show changed penalty values compared to pre-reflection
4. Verify GRPO grouping invariants:
   - replaced trajectories retain original group `uid`
   - replaced trajectories retain original slot `traj_uid`
   - replaced trajectories have consistent per-trajectory `traj_uid` (step/slot consistency)
   - `batch_size % rollout_n == 0` remains true
   - group-wise advantage computation produces expected non-zero variance for previously collapsed groups
5. Confirm PPO path still runs:
   - `old_log_probs`, `ref_log_prob`, `advantages` computed successfully
6. Monitor train path sanity:
   - if training monitor, ensure no unexpected shape mismatch in monitor update
7. Prompt-format quality sanity:
   - inspect a small sample of reflected responses for meta-reference leakage
   - track `log_prob(reflected_response | original_prompt)` distribution and fraction likely inside PPO clip range

---

## 3. Risk Register and Mitigations

### Risk R1: No valid critique tags for selected samples
- Mitigation: skip those samples; backfill with next highest-penalty candidates.

### Risk R2: Reflection prompt causes meta responses
- Mitigation: strict system instruction "do not reference prior attempts/reviews".
- Additional mitigation: compare same vs cross mode empirically.

### Risk R3: Compute overhead too high
- Mitigation: start with low ratio (0.1–0.2), monitor_rollout_n=1 for reflection path.
- Optional future: conditional reflection only when lagrangian episode_cost above threshold.

### Risk R4: Group integrity bugs
- Mitigation: only replace in-place by index; never append/remove entries.

### Risk R5: Incorrect `uid` after reflection replacement
- Mitigation: explicitly overwrite reflected per-step `uid` with original slot/group `uid` before `gather_rollout_data`.

### Risk R7: Inconsistent `traj_uid` after replacement
- Mitigation: enforce step/slot `traj_uid` consistency checks before `gather_rollout_data`.

### Risk R8: Lost reflection provenance when inheriting `traj_uid`
- Mitigation: add explicit reflection metadata fields instead of overloading identity IDs.

### Risk R6: Trigger oscillation or over-triggering
- Mitigation: include `delay_steps` warmup and `trigger_threshold`; optionally use rolling mean instead of per-batch mean.

---

## 4. Suggested Ablation Matrix (Post-Implementation)

After stable implementation, run:

1. `enable=False` baseline
2. `enable=True, mode=same_traj, ratio=0.25`
3. `enable=True, mode=cross_traj, ratio=0.25`
4. `enable=True, mode=same_traj, ratio=0.5`

Track:
- task reward
- trust penalty mean
- lagrangian λ trajectory
- proportion of groups with non-zero GRPO variance
- training stability / OOM / step time

---

## 5. Handoff Notes for Next Assistant

If continuing implementation from scratch, start in this exact order:
1. config + validation
2. preprocess system-prompt injection
3. helper utilities for selection/demo prompt
4. reflection rollout + replacement orchestration
5. integration in `multi_turn_loop`
6. logging + smoke test

The key invariant to preserve:
> **No batch-size or group-structure changes; only in-place trajectory replacement before `gather_rollout_data`.**

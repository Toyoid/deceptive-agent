# Copyright 2026 Hanxiao Li, Beihang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from collections import deque
from typing import List, Tuple, Optional, Callable

from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.model import compute_position_id_with_mask


class ReflectionPipeline:
    def __init__(
        self,
        config,
        tokenizer: PreTrainedTokenizer,
        monitor_tokenizer: PreTrainedTokenizer,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.monitor_tokenizer = monitor_tokenizer
        # Rolling mean buffer for reflection trigger (stores per-step batch mean trust penalties).
        self._reflection_penalty_buffer: deque = deque(maxlen=50)

    def _select_reflection_candidates(
        self,
        actor_trust_penalties: np.ndarray,
        rollout_n: int,
        reflection_cfg,
        verbose: bool = False,
    ) -> List[int]:
        """
        Select actor trajectory indices for reflection based on trust-penalty ranking within GRPO groups.

        Within each group of size `rollout_n`, up to k = max(1, round(ratio*rollout_n)) trajectories
        that satisfy trust_penalty >= min_trust_penalty are selected (top-K by penalty).

        Returns:
            Sorted list of selected actor trajectory indices.
        """
        ratio = reflection_cfg.ratio
        min_penalty = reflection_cfg.min_trust_penalty

        k_per_group = max(1, round(ratio * rollout_n)) if ratio > 0 else 0
        if k_per_group == 0:
            if verbose:
                print("[Reflection|Step1] k_per_group=0 (ratio=0), returning no candidates.")
            return []

        actor_batch_size = len(actor_trust_penalties)
        n_groups = actor_batch_size // rollout_n

        if verbose:
            print(f"[Reflection|Step1] Config: batch_size={actor_batch_size}, n_groups={n_groups}, "
                  f"rollout_n={rollout_n}, k_per_group={k_per_group}, "
                  f"ratio={ratio}, min_penalty={min_penalty}")
            print(f"[Reflection|Step1] All trust penalties: "
                  f"{[f'{p:.3f}' for p in actor_trust_penalties]}")

        selected = []
        for g in range(n_groups):
            start = g * rollout_n
            end = start + rollout_n
            group_penalties = actor_trust_penalties[start:end]
            # Candidates: only those meeting min_trust_penalty
            candidate_local = [
                j for j in range(rollout_n)
                if group_penalties[j] >= min_penalty
            ]
            if not candidate_local:
                if verbose:
                    print(f"  Group {g:>3}: penalties={[f'{p:.3f}' for p in group_penalties]}  "
                          f"eligible=0  (all below min_penalty={min_penalty})  selected=[]")
                continue
            # Sort by penalty descending, then take top-k
            candidate_local.sort(key=lambda j: group_penalties[j], reverse=True)
            k = min(k_per_group, len(candidate_local))
            chosen_local = candidate_local[:k]
            chosen_global = [start + j for j in chosen_local]
            selected.extend(chosen_global)

            if verbose:
                print(f"  Group {g:>3}: penalties={[f'{p:.3f}' for p in group_penalties]}  "
                      f"eligible={len(candidate_local)}  "
                      f"selected_local={chosen_local}  "
                      f"selected_global={chosen_global}  "
                      f"selected_penalties={[f'{group_penalties[j]:.3f}' for j in chosen_local]}")

        result = sorted(selected)
        if verbose:
            print(f"[Reflection|Step1] Final selected indices: {result}  (total={len(result)})")
        return result

    def _pick_demo_critique_per_actor_traj(
        self,
        monitor_batch_output: DataProto,
        actor_batch_size: int,
        monitor_rollout_n: int,
        selected_indices: Optional[List[int]] = None,
        verbose: bool = False,
    ) -> Tuple[List[Optional[str]], np.ndarray, np.ndarray]:
        """
        For each actor trajectory in selected_indices (or all, if None), pick the
        highest-judge-score, format-correct critique text.

        Among the `monitor_rollout_n` monitor rollouts for each actor trajectory (interleaved
        indexing), choose the one whose trust_penalty (judge score) is highest while
        is_format_correct is True.

        When selected_indices is provided, only those monitor rows are decoded, giving a
        proportional reduction in batch_decode cost (e.g. 4x at ratio=0.3, rollout_n=8).
        Global statistics (format correctness rate, penalty range) are still computed from
        the full numpy arrays, which are cheap and do not require decoding.

        Returns:
            best_critique_text: list[Optional[str]] of length actor_batch_size; None if no valid.
            best_critique_score: float32 ndarray (actor_batch_size,); 0.0 if no valid.
            has_valid_critique: bool ndarray (actor_batch_size,).
        """
        # NOTE: Importantly, this function assumes the multi-rollout of monitor_batch_output interleaved w.r.t. actor trajectories.

        from agent_system.environments.prompts.judge_prompt import extract_critiques

        trust_penalties = monitor_batch_output.non_tensor_batch['trust_penalties']
        is_format_correct = monitor_batch_output.non_tensor_batch['is_format_correct']

        # Global stats use full arrays — cheap numpy ops, no decoding required
        if verbose:
            total_monitor = len(trust_penalties)
            fmt_correct_count = int(is_format_correct.sum())
            n_to_decode = (len(selected_indices) if selected_indices is not None
                           else actor_batch_size) * monitor_rollout_n
            print(f"[Reflection|Step2] Monitor batch: total_rollouts={total_monitor} "
                  f"(actor_batch_size={actor_batch_size} x monitor_rollout_n={monitor_rollout_n}), "
                  f"format_correct={fmt_correct_count}/{total_monitor} "
                  f"({fmt_correct_count/max(total_monitor,1):.3f}), "
                  f"trust_penalty range=[{float(trust_penalties.min()):.3f}, "
                  f"{float(trust_penalties.max()):.3f}], "
                  f"mean={float(trust_penalties.mean()):.3f}, "
                  f"decoding {n_to_decode}/{total_monitor} monitor rows")

        indices_to_process: List[int] = (
            selected_indices if selected_indices is not None
            else list(range(actor_batch_size))
        )

        # Pre-slice: only decode the monitor rows for the trajectories we actually need
        selected_monitor_rows = np.array([
            actor_i * monitor_rollout_n + r
            for actor_i in indices_to_process
            for r in range(monitor_rollout_n)
        ])
        decoded_sliced = self.monitor_tokenizer.batch_decode(
            monitor_batch_output.batch['responses'][selected_monitor_rows],
            skip_special_tokens=True,
        )

        best_text: List[Optional[str]] = [None] * actor_batch_size
        best_score = np.zeros(actor_batch_size, dtype=np.float32)
        has_valid = np.zeros(actor_batch_size, dtype=bool)

        for local_i, actor_i in enumerate(indices_to_process):
            for r in range(monitor_rollout_n):
                monitor_idx = actor_i * monitor_rollout_n + r   # index into full trust_penalties/is_format_correct
                if not is_format_correct[monitor_idx]:
                    continue
                score = float(trust_penalties[monitor_idx])
                if not has_valid[actor_i] or score > best_score[actor_i]:
                    decoded_idx = local_i * monitor_rollout_n + r  # index into decoded_sliced
                    text = decoded_sliced[decoded_idx].strip()
                    critiques = extract_critiques(text)
                    summary = "- " + "\n- ".join(critiques) if critiques else text
                    best_text[actor_i] = summary
                    best_score[actor_i] = score
                    has_valid[actor_i] = True

        if verbose:
            valid_count = int(sum(has_valid[i] for i in indices_to_process))
            print(f"[Reflection|Step2] Trajectories with valid critique: "
                  f"{valid_count}/{len(indices_to_process)} processed "
                  f"({valid_count/max(len(indices_to_process),1):.3f})")

        return best_text, best_score, has_valid

    @staticmethod
    def _build_reflection_prompt(
        prompt_style: str,
        demo_critique: str,
        demo_response: str = "",
    ) -> str:
        """
        Build the system-role reflection prompt injected into the augmented generation context.

        Designed so the actor:
          1. Understands the honesty issue to avoid.
          2. Produces a clean, task-focused response without referencing the review context.

        Args:
            demo_critique: Critique text summarising the honesty issue.
            prompt_style: 'critique_only' — critique only;
                          'critique_resp' — includes prior response text as a concrete example.
            demo_response: (critique_resp only) text of the prior flagged response.

        Returns:
            System prompt string (used as {"role": "system"} in the augmented chat).
        """
        # TODO:
        # 1. Refine the prompt format
        # 2. Refine the wording
        preamble = "<internal_note>\nINTERNAL NOTE — do not reference this note in your response: "
        postamble = (
            "\nEnsure your response does not exhibit this problem. "
            "Answer the user's task directly and accurately without mentioning prior attempts, "
            "reviews, or this note.\n</internal_note>\n"
        )
        if prompt_style == "critique_only":
            return (
                preamble
                + "A previous response to a similar task was flagged for the following issue:\n"
                + demo_critique + "\n"
                + postamble
            )
        elif prompt_style == "critique_resp":
            return (
                preamble
                + "A previous response to a similar task read:\n\""
                + demo_response
                + "\".\nIt was flagged for the following issue:\n"
                + demo_critique + "\n"
                + postamble
            )
        else:
            raise ValueError(
                f"Unknown prompt_style: {prompt_style!r}. "
                "Expected 'critique_only' or 'critique_resp'."
            )

    def _select_demo_source_index(self, target_i: int) -> int:
        """
        Returns the source trajectory index for the reflection demo slot.

        Currently always same_traj: the target trajectory provides its own response and critique.

        # NOTE: cross_traj (use the highest-penalty selected trajectory in the same GRPO group
        # as the demo source) was considered but removed. Without including the demo task
        # description in the reflection prompt, cross_traj offers little advantage over
        # same_traj for task-agnostic behavioral critiques; including it would create a
        # task-mismatch confusion for the actor. Revisit if critiques become task-specific
        # or if empirical results motivate cross-trajectory diversity.
        """
        return target_i

    def run(
        self,
        actor_batch_dict: dict,
        monitor_batch_output: DataProto,
        actor_trust_penalties: np.ndarray,
        gen_batch: DataProto,
        env_kwargs_backup,
        actor_rollout_wg,
        monitor_wg,
        judge_wg,
        envs,
        rollout_n: int,  # NOTE: Is rollout_n needed? Or just use monitor's configuration for monitor_rollout_n?
        train_step: int,
        rollout_fn: Callable,
    ) -> Tuple[dict, np.ndarray, dict]:
        """
        Critique-Guided Trajectory Reflection — full pipeline:

          1. Precondition checks (enable, warmup, trigger threshold).
          2. Select high trust-penalty trajectories per GRPO group.
          3. Extract best format-correct critique from monitor batch per selected trajectory.
          4. Build augmented reflection prompts and reflected gen_batch.
          5. Run vanilla_multi_turn_loop on reflected sub-batch (rollout_n=1).
          6. Re-pair reflected step dicts: replace augmented prompt with original prompt tensors
             so training sees  original_prompt → reflected_response.
          7. Replace selected slots in actor_batch_dict in-place (preserving uid/traj_uid).
          8. Update actor_trust_penalties for replaced slots.

        Returns:
            (updated_actor_batch_dict, updated_actor_trust_penalties, reflection_metrics)
        """
        ref_cfg = self.config.algorithm.reflection
        actor_batch_size = len(actor_batch_dict['total_batch_list'])
        monitor_rollout_n_main = (
            len(monitor_batch_output.non_tensor_batch['trust_penalties']) // actor_batch_size
            if monitor_batch_output is not None else 1
        )

        # ---- Preconditions ------------------------------------------------
        if not ref_cfg.enable:
            return actor_batch_dict, actor_trust_penalties, {}
        if monitor_batch_output is None:
            print("[Reflection] Skipping: no monitor batch output available.")
            return actor_batch_dict, actor_trust_penalties, {}
        if train_step < ref_cfg.delay_steps:
            print(f"[Reflection] Warmup: step {train_step} < delay_steps {ref_cfg.delay_steps}.")
            return actor_batch_dict, actor_trust_penalties, {}

        # ---- Trigger threshold ------------------------------------------------
        batch_mean_penalty = float(np.mean(actor_trust_penalties))
        if ref_cfg.trigger_use_rolling_mean:
            self._reflection_penalty_buffer.append(batch_mean_penalty)
            trigger_metric = float(np.mean(self._reflection_penalty_buffer))
        else:
            trigger_metric = batch_mean_penalty

        if trigger_metric < ref_cfg.trigger_threshold:
            print(f"[Reflection] Trigger metric {trigger_metric:.4f} < "
                  f"threshold {ref_cfg.trigger_threshold} — skipping.")
            return actor_batch_dict, actor_trust_penalties, {'trigger_metric': trigger_metric}

        # ---- Step 1: Select candidates ----------------------------------------
        debug_mode = ref_cfg.get('debug_stop_after', None) is not None
        selected_indices = self._select_reflection_candidates(
            actor_trust_penalties=actor_trust_penalties,
            rollout_n=rollout_n,
            reflection_cfg=ref_cfg,
            verbose=debug_mode,
        )
        if not selected_indices:
            print("[Reflection] No candidates selected (all below min_trust_penalty).")
            return actor_batch_dict, actor_trust_penalties, {
                'enabled': 1, 'selected_count': 0,
                'selected_ratio_actual': 0.0, 'trigger_metric': trigger_metric,
            }

        n_selected = len(selected_indices)
        mean_penalty_before = float(np.mean(actor_trust_penalties[np.array(selected_indices)]))
        print(f"[Reflection] Selected {n_selected}/{actor_batch_size} trajectories "
              f"(selected mean penalty before={mean_penalty_before:.4f}, trigger={trigger_metric:.4f})")

        # ---- Step 2: Extract critiques ----------------------------------------
        # Only decode monitor rows for selected trajectories — at 2880-row scale this
        # gives a proportional saving (e.g. 4x at ratio=0.3, rollout_n=8).
        best_critique_text, best_critique_score, has_valid_critique = \
            self._pick_demo_critique_per_actor_traj(
                monitor_batch_output=monitor_batch_output,
                actor_batch_size=actor_batch_size,
                monitor_rollout_n=monitor_rollout_n_main,
                selected_indices=selected_indices,
                verbose=debug_mode,
            )

        valid_selected = [i for i in selected_indices if has_valid_critique[i]]
        valid_critique_ratio = len(valid_selected) / max(n_selected, 1)
        print(f"[Reflection] Valid critique ratio among selected: "
              f"{len(valid_selected)}/{n_selected} ({valid_critique_ratio:.3f})")

        if debug_mode:
            # Valid trajectories: text already available in best_critique_text.
            # Invalid trajectories: decode only those specific monitor rows
            invalid_selected = [i for i in selected_indices if not has_valid_critique[i]]
            invalid_decoded_map: dict = {}
            if len(invalid_selected) > 0:
                invalid_monitor_rows = np.array([
                    i * monitor_rollout_n_main + r
                    for i in invalid_selected
                    for r in range(monitor_rollout_n_main)
                ])
                decoded_invalid = self.monitor_tokenizer.batch_decode(
                    monitor_batch_output.batch['responses'][invalid_monitor_rows],
                    skip_special_tokens=True,
                )
                for local_i, actor_i in enumerate(invalid_selected):
                    for r in range(monitor_rollout_n_main):
                        invalid_decoded_map[(actor_i, r)] = decoded_invalid[
                            local_i * monitor_rollout_n_main + r
                        ]
            print(f"[Reflection|Step2] Per-selected-traj critique details:")
            for i in selected_indices:
                if has_valid_critique[i]:
                    snippet = (best_critique_text[i] or "").replace('\n', ' ')
                    print(f"  traj {i:>4}: valid=True   score={best_critique_score[i]:.3f}  "
                          f"critique='{snippet}'")
                else:
                    raw_snippets = []
                    for r in range(monitor_rollout_n_main):
                        monitor_idx = i * monitor_rollout_n_main + r
                        raw = invalid_decoded_map.get((i, r), "").strip().replace('\n', ' ')
                        fmt = monitor_batch_output.non_tensor_batch['is_format_correct'][monitor_idx]
                        raw_snippets.append(f"r{r}(fmt={int(fmt)})='{raw}'")
                    print(f"  traj {i:>4}: valid=False  score=0.000  "
                          f"monitor_rollouts=[{', '.join(raw_snippets)}]")

        if not valid_selected:
            print("[Reflection] No valid critiques for selected trajectories — skipping.")
            return actor_batch_dict, actor_trust_penalties, {
                'enabled': 1, 'selected_count': n_selected,
                'selected_ratio_actual': n_selected / actor_batch_size,
                'mean_penalty_before_selected': mean_penalty_before,
                'valid_critique_ratio_selected': 0.0,
                'trigger_metric': trigger_metric,
            }

        # ---- Step 3: Build reflection prompts --------------------------------
        # response + critique demo is always the trajectory itself (same_traj); see _select_demo_source_index
        # for the NOTE on a removed cross_traj mode.
        prompt_style = ref_cfg.prompt_style
        reflection_prompts = []
        debug_prompt_info: List[dict] = [] if debug_mode else []
        for i in valid_selected:
            demo_critique = best_critique_text[i] or ""
            demo_response_text = ""
            if prompt_style == "critique_resp":
                # Decode the last active response of this trajectory
                for step_data in reversed(actor_batch_dict['total_batch_list'][i]):
                    if step_data.get('active_masks', True):
                        if 'responses' in step_data and isinstance(step_data['responses'], torch.Tensor):
                            demo_response_text = self.tokenizer.decode(
                                step_data['responses'], skip_special_tokens=True
                            )
                        break
            prompt = self._build_reflection_prompt(
                prompt_style=prompt_style,
                demo_critique=demo_critique,
                demo_response=demo_response_text,
            )
            reflection_prompts.append(prompt)
            if debug_mode:
                debug_prompt_info.append({
                    'traj_i': i,
                    'demo_critique': demo_critique,
                    'demo_response_text': demo_response_text,
                    'prompt': prompt,
                })

        if debug_mode:
            print(f"[Reflection|Step3] prompt_style='{prompt_style}', "
                  f"built {len(reflection_prompts)} prompts for valid_selected={valid_selected}")
            for info in debug_prompt_info:
                traj_i = info['traj_i']
                if not info['demo_critique']:
                    print(f"  traj {traj_i:>4}: WARNING — demo_critique is empty, "
                          f"reflection prompt will not carry useful signal")
                if prompt_style == "critique_resp" and not info['demo_response_text']:
                    print(f"  traj {traj_i:>4}: WARNING — demo_response_text is empty "
                          f"(no active step with tensor response found)")
                print(f"  traj {traj_i:>4}: full_prompt='{info['prompt']}'")

        # ---- Step 4: Build reflected gen_batch --------------------------------
        n_valid = len(valid_selected)
        valid_idx_arr = np.array(valid_selected)

        # Slice tensors from the pending gen_batch (already repeat()-expanded, env_kwargs popped)
        tensor_data = {k: v[valid_idx_arr] for k, v in gen_batch.batch.items()}
        reflected_gen_batch = DataProto.from_single_dict(data=tensor_data)

        # Slice non-tensor fields
        skipped_non_tensor: List[str] = []
        sliced_non_tensor: List[str] = []
        for k, v in gen_batch.non_tensor_batch.items():
            if k == 'env_kwargs':
                continue  # handled separately
            if isinstance(v, np.ndarray) and len(v) == actor_batch_size:
                reflected_gen_batch.non_tensor_batch[k] = v[valid_idx_arr]
                sliced_non_tensor.append(k)
            else:
                skipped_non_tensor.append(k)

        # Restore env_kwargs for the selected sub-batch
        if env_kwargs_backup is not None:
            reflected_gen_batch.non_tensor_batch['env_kwargs'] = env_kwargs_backup[valid_idx_arr]

        # Inject reflection system prompts
        refl_prompt_arr = np.array(reflection_prompts, dtype=object)
        reflected_gen_batch.non_tensor_batch['reflection_prompt'] = refl_prompt_arr
        reflected_gen_batch.meta_info = gen_batch.meta_info

        if debug_mode:
            print(f"[Reflection|Step4] reflected_gen_batch built: n_valid={n_valid}, "
                  f"valid_idx_arr={valid_idx_arr.tolist()}")
            # Tensor fields
            shape_mismatches = []
            print(f"[Reflection|Step4] Tensor fields ({len(tensor_data)}):")
            for k, v in reflected_gen_batch.batch.items():
                ok = (v.shape[0] == n_valid)
                if not ok:
                    shape_mismatches.append(k)
                print(f"  {k}: shape={tuple(v.shape)}  {'OK' if ok else 'MISMATCH — expected first dim=' + str(n_valid)}")
            if shape_mismatches:
                print(f"  WARNING — shape mismatch in fields: {shape_mismatches}")
            # Non-tensor fields
            print(f"[Reflection|Step4] Non-tensor fields sliced ({len(sliced_non_tensor)}): {sliced_non_tensor}")
            if skipped_non_tensor:
                print(f"[Reflection|Step4] Non-tensor fields SKIPPED (len != actor_batch_size "
                      f"or not ndarray) ({len(skipped_non_tensor)}): {skipped_non_tensor}")
            # env_kwargs
            if env_kwargs_backup is None:
                print(f"[Reflection|Step4] env_kwargs_backup is None — no environment to reset "
                      f"(expected for non-interactive tasks)")
            else:
                env_kw = reflected_gen_batch.non_tensor_batch.get('env_kwargs', None)
                env_len = len(env_kw) if env_kw is not None else 0
                print(f"[Reflection|Step4] env_kwargs sliced: len={env_len} "
                      f"({'OK' if env_len == n_valid else 'MISMATCH — expected ' + str(n_valid)})")
            # reflection_prompt injection (most critical check)
            n_none = int(sum(p is None for p in refl_prompt_arr))
            n_empty = int(sum((p is not None and p == "") for p in refl_prompt_arr))
            print(f"[Reflection|Step4] reflection_prompt: "
                  f"len={len(refl_prompt_arr)}, None={n_none}, empty={n_empty} "
                  f"({'OK' if n_none == 0 and n_empty == 0 else 'WARNING — None/empty prompts will bypass augmentation'})")
            if len(refl_prompt_arr) > 0:
                print(f"  sample[0]='{str(refl_prompt_arr[0])[:200]}'")
            # meta_info
            print(f"[Reflection|Step4] meta_info keys: {list(reflected_gen_batch.meta_info.keys()) if reflected_gen_batch.meta_info else 'None'}")

        # ---- Step 5: Run reflected rollout ------------------------------------
        print(f"[Reflection] Running reflected rollout for {n_valid} trajectory slots...")
        reflected_actor_batch_dict, reflected_monitor_batch = rollout_fn(
            gen_batch=reflected_gen_batch,
            actor_rollout_wg=actor_rollout_wg,
            monitor_wg=monitor_wg,
            judge_wg=judge_wg,
            envs=envs,
            rollout_n=1,  # reflected sub-batch is not GRPO-grouped
            monitor_rollout_n=ref_cfg.monitor_rollout_n,
            verbose=debug_mode,  # DEBUG
        )

        # ---- Step 6: Aggregate reflected trust penalties ----------------------
        if reflected_monitor_batch is not None:
            refl_raw = np.clip(
                reflected_monitor_batch.non_tensor_batch['trust_penalties'], 0.0, 1.0
            )
            if len(refl_raw) == n_valid:
                refl_penalties = refl_raw
            else:
                refl_n = ref_cfg.monitor_rollout_n
                assert len(refl_raw) == n_valid * refl_n, (
                    f"Unexpected reflected trust_penalty size: got {len(refl_raw)}, "
                    f"expected {n_valid} or {n_valid * refl_n}"
                )
                refl_penalties = refl_raw.reshape(n_valid, refl_n).mean(axis=1)
        else:
            raise RuntimeError("Reflection monitor batch output is None — expected trust penalties for reflected trajectories")

        mean_penalty_after = float(np.mean(refl_penalties))
        print(f"[Reflection] Trust penalty: before={mean_penalty_before:.4f}, "
              f"after={mean_penalty_after:.4f}")

        if debug_mode:
            monitor_rollout_n_refl = ref_cfg.monitor_rollout_n
            mon_responses_decoded = self.monitor_tokenizer.batch_decode(
                reflected_monitor_batch.batch['responses'], skip_special_tokens=True
            )
            mon_prompts_decoded = self.monitor_tokenizer.batch_decode(
                reflected_monitor_batch.batch['prompts'], skip_special_tokens=True
            )
            mon_trust_raw = reflected_monitor_batch.non_tensor_batch['trust_penalties']
            mon_fmt_raw = reflected_monitor_batch.non_tensor_batch['is_format_correct']

            sep = '=' * 72
            print(f"\n{sep}")
            print(f"[Reflection|Step5+6] Full rollout results: "
                  f"n_valid={n_valid}, monitor_rollout_n={monitor_rollout_n_refl}")
            print(sep)

            for ref_local, orig_idx in enumerate(valid_selected):
                reward = float(reflected_actor_batch_dict['episode_rewards'][ref_local])
                length = int(reflected_actor_batch_dict['episode_lengths'][ref_local])
                pen_before = float(actor_trust_penalties[orig_idx])
                pen_after = float(refl_penalties[ref_local])

                print(f"\n>>> Traj orig_idx={orig_idx}  (ref_local={ref_local})")
                print(f"    episode_reward={reward:.4f}  episode_length={length}"
                      f"  trust_penalty: {pen_before:.4f} -> {pen_after:.4f}")

                # ----- Actor -----
                steps = reflected_actor_batch_dict['total_batch_list'][ref_local]
                print(f"\n  [ACTOR — {len(steps)} step(s)]")
                for step_i, sd in enumerate(steps):
                    active = bool(sd.get('active_masks', True))
                    print(f"  -- step {step_i} (active={active}) --")
                    if 'prompts' in sd and isinstance(sd['prompts'], torch.Tensor):
                        print(f"  INPUT:\n{self.tokenizer.decode(sd['prompts'], skip_special_tokens=True)}")
                    if 'responses' in sd and isinstance(sd['responses'], torch.Tensor):
                        print(f"  OUTPUT:\n{self.tokenizer.decode(sd['responses'], skip_special_tokens=True)}")
                    agent_resp = sd.get('agent_response', None)
                    if agent_resp is not None:
                        print(f"  agent_response: {agent_resp}")

                # ----- Monitor -----
                print(f"\n  [MONITOR — {monitor_rollout_n_refl} rollout(s)]")
                for r in range(monitor_rollout_n_refl):
                    mon_idx = ref_local * monitor_rollout_n_refl + r
                    fmt = bool(mon_fmt_raw[mon_idx])
                    pen = float(mon_trust_raw[mon_idx])
                    print(f"  -- rollout r={r} | format_correct={fmt} | trust_penalty={pen:.4f} --")
                    print(f"  INPUT:\n{mon_prompts_decoded[mon_idx]}")
                    print(f"  OUTPUT:\n{mon_responses_decoded[mon_idx]}")

                # ----- Judge -----
                # Judge is called internally; its output is the trust_penalty above.
                # Judge input = actor agent_response + monitor critique (shown above).
                print(f"\n  [JUDGE]  trust_penalty (judge output) = {pen_after:.4f}")
                print(f"\n{'-' * 72}")

            print(f"\n{sep}")
            print(f"[Reflection|Step5+6] Summary: mean trust_penalty "
                  f"before={mean_penalty_before:.4f}, after={mean_penalty_after:.4f}")
            print(f"{sep}\n")

        # ---- Step 7: Re-pair step dicts and replace in actor_batch_dict ------
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        actor_trust_penalties = actor_trust_penalties.copy()

        for ref_local, orig_idx in enumerate(valid_selected):
            # Preserve original GRPO group uid and trajectory traj_uid (N10 in design doc)
            orig_uid = actor_batch_dict['total_batch_list'][orig_idx][0]['uid']
            orig_traj_uid = actor_batch_dict['traj_uid'][orig_idx]

            reflected_steps = reflected_actor_batch_dict['total_batch_list'][ref_local]

            # DEBUG: capture original trajectory state before replacement
            if debug_mode:
                _sep = '=' * 72
                _orig_n_steps   = len(actor_batch_dict['total_batch_list'][orig_idx])
                _orig_reward    = float(actor_batch_dict['episode_rewards'][orig_idx])
                _orig_length    = int(actor_batch_dict['episode_lengths'][orig_idx])
                _orig_penalty   = float(actor_trust_penalties[orig_idx])
                _refl_reward    = float(reflected_actor_batch_dict['episode_rewards'][ref_local])
                _refl_length    = int(reflected_actor_batch_dict['episode_lengths'][ref_local])
                _refl_penalty   = float(refl_penalties[ref_local])
                print(f"\n{_sep}")
                print(f"[Reflection|Step7|DEBUG] orig_idx={orig_idx}  ref_local={ref_local}")
                print(f"  uid:        {orig_uid}")
                print(f"  traj_uid:   {orig_traj_uid}")
                print(f"  steps:      original={_orig_n_steps}  reflected={len(reflected_steps)}")
                print(f"  reward:     {_orig_reward:.4f} -> {_refl_reward:.4f}")
                print(f"  length:     {_orig_length} -> {_refl_length}")
                print(f"  trust_pen:  {_orig_penalty:.4f} -> {_refl_penalty:.4f}")

            for step_i, step_dict in enumerate(reflected_steps):
                # DEBUG: capture all pre-pair state before pop/overwrite
                if debug_mode:
                    _has_orig = 'orig_input_ids' in step_dict
                    _resp_shape = step_dict['responses'].shape if isinstance(step_dict.get('responses'), torch.Tensor) else None
                    _refl_prompt_shape = step_dict['prompts'].shape if isinstance(step_dict.get('prompts'), torch.Tensor) else None
                    if _has_orig:
                        _orig_ids_shape = step_dict['orig_input_ids'].shape
                        _orig_att_shape = step_dict['orig_attention_mask'].shape
                        _aug_prompt_decoded = self.tokenizer.decode(step_dict['prompts'], skip_special_tokens=True)
                        _resp_decoded_pre   = self.tokenizer.decode(step_dict['responses'], skip_special_tokens=True)
                    else:
                        _orig_ids_shape = None
                        _orig_att_shape = None
                        _aug_prompt_decoded = None
                        _resp_decoded_pre = None

                # Re-pair: swap augmented prompt for original prompt tensors
                if 'orig_input_ids' in step_dict:
                    orig_ids = step_dict.pop('orig_input_ids')        # 1D tensor (stashed as tensor in build_single_actor_sample)
                    orig_att = step_dict.pop('orig_attention_mask')   # 1D tensor
                    responses = step_dict['responses']  # 1-D tensor (max_resp_len,)
                    # Reconstruct full sequence with original prompt prefix
                    new_input_ids = torch.cat([orig_ids, responses])
                    resp_att = (responses != pad_id).long()
                    new_att = torch.cat([orig_att, resp_att])
                    new_pos = compute_position_id_with_mask(new_att.unsqueeze(0))[0]
                    step_dict['input_ids'] = new_input_ids
                    step_dict['prompts'] = orig_ids
                    step_dict['attention_mask'] = new_att
                    step_dict['position_ids'] = new_pos

                # Restore original uid/traj_uid to preserve GRPO group identity (N10 in design doc)
                step_dict['uid'] = orig_uid
                step_dict['traj_uid'] = orig_traj_uid

                # DEBUG: print per-step re-pair verification
                if debug_mode:
                    print(f"\n  [Step {step_i}]  orig_input_ids present: {_has_orig}"
                          + ("" if _has_orig else "  <-- WARNING: re-pair skipped for this step!"))
                    if _has_orig:
                        _new_ids_len   = len(step_dict['input_ids'])
                        _expected_len  = _orig_ids_shape[0] + _resp_shape[0]
                        _shape_ok      = _new_ids_len == _expected_len
                        _att_sum       = int(step_dict['attention_mask'].sum().item())
                        _pos_max       = int(step_dict['position_ids'].max().item())
                        _pos_ok        = _pos_max == _att_sum - 1
                        _orig_decoded  = self.tokenizer.decode(step_dict['prompts'], skip_special_tokens=True)
                        _resp_decoded  = self.tokenizer.decode(step_dict['responses'], skip_special_tokens=True)
                        print(f"    orig_ids shape:       {_orig_ids_shape}  orig_att shape: {_orig_att_shape}")
                        print(f"    responses shape:      {_resp_shape}")
                        print(f"    new input_ids len:    {_new_ids_len} = {_orig_ids_shape[0]} + {_resp_shape[0]}"
                              + (f"  OK" if _shape_ok else f"  MISMATCH! expected {_expected_len}"))
                        print(f"    attn nonzero:         {_att_sum} / {len(step_dict['attention_mask'])}")
                        print(f"    position_ids max:     {_pos_max}  (expect {_att_sum - 1})"
                              + ("  OK" if _pos_ok else "  WRONG!"))
                        print(f"    uid:                  {step_dict['uid']}  (orig_uid={orig_uid})"
                              + ("  OK" if step_dict['uid'] == orig_uid else "  MISMATCH!"))
                        print(f"    traj_uid:             {step_dict['traj_uid']}  (orig_traj_uid={orig_traj_uid})"
                              + ("  OK" if step_dict['traj_uid'] == orig_traj_uid else "  MISMATCH!"))
                        print(f"    BEFORE prompt (augmented, {_refl_prompt_shape[0] if _refl_prompt_shape else '?'} tokens):")
                        print(f"      {_aug_prompt_decoded}")
                        print(f"    AFTER  prompt (original,  {_orig_ids_shape[0]} tokens):")
                        print(f"      {_orig_decoded}")
                        # Ground-truth comparison: recovered orig_ids vs original actor_batch step prompts
                        # actor_batch_dict['total_batch_list'][orig_idx] still holds the OLD trajectory here —
                        # the in-place overwrite comes after this step loop, so this read is safe
                        _actor_orig_steps = actor_batch_dict['total_batch_list'][orig_idx]
                        if step_i < len(_actor_orig_steps) and 'prompts' in _actor_orig_steps[step_i]:
                            _actor_orig_prompt = self.tokenizer.decode(
                                _actor_orig_steps[step_i]['prompts'], skip_special_tokens=True)
                            _stash_match = "MATCH" if _orig_decoded == _actor_orig_prompt else "*** STASH MISMATCH ***"
                            print(f"    orig stash vs actor_batch original prompt: {_stash_match}")
                            if "MISMATCH" in _stash_match:
                                print(f"      ACTOR ORIGINAL: {_actor_orig_prompt[:400]!r}")
                                print(f"      RECOVERED STASH:{_orig_decoded[:400]!r}")
                        else:
                            print(f"    orig stash vs actor_batch original prompt: N/A "
                                  f"(step {step_i} out of range or no 'prompts' key)")
                        print(f"    response (unchanged):")
                        print(f"      {_resp_decoded}")
                        if _resp_decoded_pre != _resp_decoded:
                            print(f"    WARNING: response text changed after re-pair! Before: {_resp_decoded_pre!r}")
                    else:
                        # Re-pair skipped — show what's in the step for diagnosis
                        print(f"    reflected prompt shape: {_refl_prompt_shape}")
                        print(f"    responses shape:        {_resp_shape}")
                        print(f"    step_dict keys:         {list(step_dict.keys())}")

            # Replace trajectory in actor_batch_dict in-place
            actor_batch_dict['total_batch_list'][orig_idx] = reflected_steps
            actor_batch_dict['episode_rewards'][orig_idx] = \
                reflected_actor_batch_dict['episode_rewards'][ref_local]
            actor_batch_dict['episode_lengths'][orig_idx] = \
                reflected_actor_batch_dict['episode_lengths'][ref_local]
            actor_batch_dict['tool_callings'][orig_idx] = \
                reflected_actor_batch_dict['tool_callings'][ref_local]
            # Preserve original traj_uid slot identity (chosen default — design doc N1)
            actor_batch_dict['traj_uid'][orig_idx] = orig_traj_uid
            # Update trust penalty for this slot with reflected score
            actor_trust_penalties[orig_idx] = float(refl_penalties[ref_local])

            # DEBUG: confirm in-place replacement values
            if debug_mode:
                _final_reward = float(actor_batch_dict['episode_rewards'][orig_idx])
                _final_length = int(actor_batch_dict['episode_lengths'][orig_idx])
                _final_traj_uid = actor_batch_dict['traj_uid'][orig_idx]
                _final_penalty = float(actor_trust_penalties[orig_idx])
                print(f"\n  [Post-replacement]")
                print(f"    episode_reward:   {_final_reward:.4f}  (expected {_refl_reward:.4f})"
                      + ("  OK" if abs(_final_reward - _refl_reward) < 1e-6 else "  MISMATCH!"))
                print(f"    episode_length:   {_final_length}  (expected {_refl_length})"
                      + ("  OK" if _final_length == _refl_length else "  MISMATCH!"))
                print(f"    traj_uid:         {_final_traj_uid}  (expected {orig_traj_uid})"
                      + ("  OK" if _final_traj_uid == orig_traj_uid else "  MISMATCH!"))
                print(f"    trust_penalty:    {_final_penalty:.4f}  (expected {_refl_penalty:.4f})"
                      + ("  OK" if abs(_final_penalty - _refl_penalty) < 1e-6 else "  MISMATCH!"))

        # Note: reflected monitor batch is NOT merged into monitor training batch (Phase-1).
        # TODO: Future option to include reflected monitor data in monitor training.

        # ---- Step 8: Build metrics dict --------------------------------------
        metrics = {
            'enabled': 1,
            'selected_count': n_valid,
            'selected_ratio_actual': n_valid / actor_batch_size,
            'mean_penalty_before_selected': mean_penalty_before,
            'mean_penalty_after_selected': mean_penalty_after,
            'valid_critique_ratio_selected': valid_critique_ratio,
            'trigger_metric': trigger_metric
        }
        return actor_batch_dict, actor_trust_penalties, metrics

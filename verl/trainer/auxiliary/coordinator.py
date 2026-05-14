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

from __future__ import annotations

from dataclasses import dataclass
import uuid
from typing import Any, Callable, ContextManager, Optional

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from verl import DataProto
from agent_system.utils.reason_answer_format import extract_visible_answer


DEBUG_PRINT_AUX_SAMPLES = 2


@dataclass
class AuxiliaryStepResult:
    metrics: dict[str, Any]
    total_num_tokens: int = 0


def _prefix_metrics(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


class AuxiliaryCoordinator:
    """Prompt-only auxiliary RL appended after each main training step.

    Design principles:
    - Keep the main env rollout/update path canonical in RayPPOTrainer.fit().
    - Keep auxiliary batches source-pure rather than merging them with env data.
    - Reuse the same workers and PPO/GRPO math, but only for the subset of logic
      that makes sense for prompt-only safety training.

    Assumptions:
    - Auxiliary uses the same global advantage estimator as the main run.
    - Auxiliary supports only estimators whose required tensors can be produced by
      prompt-only RM-scored batches. Estimators that depend on critic values,
      reward baselines, or env-structured step signals are rejected by trainer
      validation.
    - Auxiliary rewards come directly from an RM score and intentionally skip
      env-only features such as invalid-action penalties or monitor/judge flows.
    - Auxiliary batches are expected to satisfy the same distributed batch-shape
      constraints as the main actor update path; the coordinator raises errors for 
      duplicating or deleting prompt-only rollout rows.
    """

    def __init__(
        self,
        config,
        tokenizer,
        processor,
        actor_rollout_wg,
        ref_policy_wg,
        rm_wg,
        balance_batch_fn: Callable[..., None],
        timer_fn: Callable[[str, dict[str, float]], ContextManager[None]],
        adjust_batch_fn: Callable[..., DataProto],
        compute_response_mask_fn: Callable[[DataProto], torch.Tensor],
        compute_log_prob_metrics_fn: Callable[..., tuple[DataProto, dict[str, Any]]],
        compute_advantage_fn: Callable[..., DataProto],
        apply_kl_penalty_fn: Callable[..., tuple[DataProto, dict[str, Any]]],
        compute_data_metrics_fn: Callable[..., dict[str, Any]],
        reduce_metrics_fn: Callable[[dict[str, Any]], dict[str, Any]],
        use_reference_policy: bool,
        ref_in_actor: bool,
        kl_ctrl_in_reward=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.actor_rollout_wg = actor_rollout_wg
        self.ref_policy_wg = ref_policy_wg
        self.rm_wg = rm_wg
        self._balance_batch = balance_batch_fn
        self._timer = timer_fn
        self._adjust_batch = adjust_batch_fn
        self._compute_response_mask = compute_response_mask_fn
        self._compute_log_prob_metrics = compute_log_prob_metrics_fn
        self._compute_advantage = compute_advantage_fn
        self._apply_kl_penalty = apply_kl_penalty_fn
        self._compute_data_metrics = compute_data_metrics_fn
        self._reduce_metrics = reduce_metrics_fn
        self.use_reference_policy = use_reference_policy
        self.ref_in_actor = ref_in_actor
        self.kl_ctrl_in_reward = kl_ctrl_in_reward

        self.aux_config = config.auxiliary
        self.rollout_n = int(self.aux_config.rollout.n)
        self.prompt_batch_size = int(self.aux_config.batch_size)
        # Auxiliary data mix ratio is defined by rollout sample counts in one interleaved window.
        self.total_rollout_rows = self.prompt_batch_size * self.rollout_n
        main_prompt_batch_size = int(self.config.data.get("gen_batch_size", self.config.data.train_batch_size))
        self.main_total_rollout_rows = main_prompt_batch_size * self.config.env.rollout.n
        self.effective_sample_ratio = self.total_rollout_rows / (
            self.main_total_rollout_rows + self.total_rollout_rows
        )

        self.use_main_rm = bool(self.aux_config.reward_model.use_main)
        self.strip_thinking = bool(self.aux_config.reward_model.strip_thinking)

        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            self.aux_config.data.train_files,
            self.aux_config.data,
            self.tokenizer,
            self.processor,
        )
        train_sampler = create_rl_sampler(self.aux_config.data, train_dataset)
        dataloader_num_workers = self.aux_config.data.get("dataloader_num_workers", 8)
        self.train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=self.prompt_batch_size,
            num_workers=dataloader_num_workers,
            persistent_workers=dataloader_num_workers > 0,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        if len(self.train_dataloader) < 1:
            raise ValueError("Auxiliary dataloader is empty. Please provide a non-empty auxiliary dataset.")
        self._iterator = iter(self.train_dataloader)

    def is_active(self, main_step: int) -> bool:
        return bool(self.aux_config.enable) and main_step >= int(self.aux_config.start_step)

    def auxiliary_steps_completed(self, main_step: int) -> int:
        if not self.is_active(main_step):
            return 0
        return main_step - int(self.aux_config.start_step) + 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "dataloader": self.train_dataloader.state_dict(),
        }

    def load_state_dict(self, state: Optional[dict[str, Any]]) -> None:
        if not state:
            return
        if "dataloader" in state:
            self.train_dataloader.load_state_dict(state["dataloader"])
            self._iterator = iter(self.train_dataloader)

    def run_step(self, main_step: int, timing_raw: dict[str, float]) -> AuxiliaryStepResult:
        if not self.is_active(main_step):
            return AuxiliaryStepResult(metrics={}, total_num_tokens=0)

        metrics: dict[str, Any] = {
            "aux/prompt_batch_size": float(self.prompt_batch_size),
            "aux/rollout_rows": float(self.total_rollout_rows),
            "aux/rollout_n": float(self.rollout_n),
            "aux/sample_ratio": float(self.effective_sample_ratio),
            "aux/step": float(self.auxiliary_steps_completed(main_step)),
            "aux/use_main_rm": float(self.use_main_rm),
            "aux/strip_thinking": float(self.strip_thinking),
        }

        with self._timer("aux/collect", timing_raw):
            batch = self._collect_rollout_batch()

        with self._timer("aux/reward", timing_raw):
            batch = self._score_batch(batch=batch, metrics=metrics)
            self._debug_print_batch(batch, main_step=main_step)

        batch = self._run_training_tail(batch=batch, main_step=main_step, metrics=metrics, timing_raw=timing_raw)

        total_num_tokens = sum(batch.meta_info["global_token_num"])
        metrics.update(
            self._compute_data_metrics(
                batch=batch,
                use_critic=False,
                metric_prefix="aux",
                include_episode_metrics=False,
            )
        )
        return AuxiliaryStepResult(metrics=metrics, total_num_tokens=total_num_tokens)

    def _next_batch(self) -> DataProto:
        try:
            batch_dict = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.train_dataloader)
            batch_dict = next(self._iterator)
        return DataProto.from_single_dict(batch_dict)

    def _collect_rollout_batch(self) -> DataProto:
        prompt_batch = self._next_batch()
        actor_batch = prompt_batch.repeat(repeat_times=self.rollout_n, interleave=True)
        actor_batch.non_tensor_batch["uid"] = self._create_uid_batch(
            batch_size=len(actor_batch),
            n_rollouts=self.rollout_n,
        )
        actor_batch.non_tensor_batch["traj_uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(actor_batch))],
            dtype=object,
        )

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = []
        if "raw_prompt_ids" in actor_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt_ids")
        if "multi_modal_data" in actor_batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")

        gen_batch = actor_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        for key in ["do_sample", "temperature", "top_k", "top_p", "response_length"]:
            value = self.aux_config.rollout.get(key, None)
            if value is not None:
                gen_batch.meta_info[key] = value
                if key == "temperature":
                    actor_batch.meta_info[key] = value

        rollout_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        actor_batch = actor_batch.union(rollout_output)
        return actor_batch

    def _score_batch(self, batch: DataProto, metrics: dict[str, Any]) -> DataProto:
        if self.strip_thinking:
            batch.meta_info["strip_thinking"] = True
        reward_tensor = self.rm_wg.compute_rm_score(batch)
        batch = batch.union(reward_tensor)
        batch.batch["token_level_scores"] = reward_tensor.batch["rm_scores"]

        rm_scalar = batch.batch["token_level_scores"].sum(-1)
        metrics.update(
            {
                "aux/rm_score/mean": rm_scalar.mean().item(),
                "aux/rm_score/std": rm_scalar.std(unbiased=False).item(),
                "aux/rm_score/min": rm_scalar.min().item(),
                "aux/rm_score/max": rm_scalar.max().item(),
            }
        )
        return batch

    def _run_training_tail(
        self,
        batch: DataProto,
        main_step: int,
        metrics: dict[str, Any],
        timing_raw: dict[str, float],
    ) -> DataProto:
        original_batch_size = len(batch)
        batch = self._adjust_batch(
            config=self.config,
            data=batch,
            world_size=self.actor_rollout_wg.world_size,
        )
        if len(batch) != original_batch_size:
            # Auxiliary is intentionally source-pure. Changing prompt-only rows at
            # runtime would make the configured sample ratio harder to interpret,
            # so v1 fails fast instead of padding/trimming the auxiliary batch.
            raise ValueError(
                "Auxiliary batches must already satisfy distributed batch divisibility. "
                "Please adjust `auxiliary.batch_size` or `auxiliary.rollout.n` so the trainer does "
                "not need to duplicate or delete rollout rows."
            )

        batch.batch["response_mask"] = self._compute_response_mask(batch)
        if self.config.trainer.balance_batch:
            self._balance_batch(
                batch,
                world_size=self.actor_rollout_wg.world_size,
                metrics=metrics,
                logging_prefix="aux_global_seqlen",
            )
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        with self._timer("aux/old_log_prob", timing_raw):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            batch, old_log_prob_metrics = self._compute_log_prob_metrics(
                batch,
                old_log_prob,
                self.config.actor_rollout_ref.actor.loss_agg_mode,
                metric_prefix="aux",
            )
            metrics.update(old_log_prob_metrics)

        if self.use_reference_policy:
            with self._timer("aux/ref", timing_raw):
                if self.ref_in_actor:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        with self._timer("aux/adv", timing_raw):
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = self._apply_kl_penalty(
                    batch,
                    kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=self.config.algorithm.kl_penalty,
                )
                metrics.update(_prefix_metrics(kl_metrics, "aux"))

            batch = self._compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.rollout_n,
                norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                multi_turn=False,
                use_pf_ppo=self.config.algorithm.use_pf_ppo,
                pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                gigpo_mode=self.config.algorithm.gigpo.mode,
                gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
            )

        if self.config.trainer.critic_warmup <= main_step:
            with self._timer("aux/update_actor", timing_raw):
                # Auxiliary batches are intentionally prompt-only and never routed
                # through the multi-turn env stack, so they should not rely on
                # env-specific loss masks or episode bookkeeping.
                batch.meta_info["multi_turn"] = False
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = self._reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(_prefix_metrics(actor_output_metrics, "aux"))

        return batch

    @staticmethod
    def _create_uid_batch(batch_size: int, n_rollouts: int) -> np.ndarray:
        if batch_size % max(1, n_rollouts) != 0:
            raise ValueError(f"batch_size {batch_size} must be divisible by n_rollouts {n_rollouts}")

        uid_batch = []
        current_uid = None
        for idx in range(batch_size):
            if idx % max(1, n_rollouts) == 0:
                current_uid = str(uuid.uuid4())
            uid_batch.append(current_uid)
        return np.array(uid_batch, dtype=object)
    
    # Debugging helpers
    def _debug_print_batch(self, batch: DataProto, main_step: int) -> None:
        if DEBUG_PRINT_AUX_SAMPLES <= 0:
            return

        num_samples = min(DEBUG_PRINT_AUX_SAMPLES, len(batch))
        response_length = batch.batch["responses"].shape[-1]
        separator = "=" * 100
        print(
            f"\n{separator}\n"
            f"[Aux Training Debug] main_step={main_step} "
            f"auxiliary_step={self.auxiliary_steps_completed(main_step)}\n"
            f"{separator}"
        )
        for idx in range(num_samples):
            if "prompts" in batch.batch:
                model_input = self.tokenizer.decode(batch.batch["prompts"][idx], skip_special_tokens=True)
            elif "raw_prompt" in batch.non_tensor_batch:
                model_input = self._format_raw_prompt(batch.non_tensor_batch["raw_prompt"][idx])
            else:
                model_input = "<prompt unavailable>"

            valid_response_length = int(batch.batch["attention_mask"][idx][-response_length:].sum().item())
            response_ids = batch.batch["responses"][idx][:valid_response_length]
            model_output = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            rm_response = extract_visible_answer(model_output) if self.strip_thinking else model_output
            rm_input = self._format_debug_rm_input(batch, idx, rm_response)
            rm_score = "<rm score unavailable>"
            if "token_level_scores" in batch.batch:
                rm_score = float(batch.batch["token_level_scores"][idx].sum().item())

            print(f"\n{'-' * 100}")
            print(f"[Aux Training Debug][sample {idx}] model_input:\n{model_input}")
            print(f"[Aux Training Debug][sample {idx}] model_output:\n{model_output}")
            print(f"[Aux Training Debug][sample {idx}] rm_input_strip_thinking={self.strip_thinking}:\n{rm_input}")
            print(f"[Aux Training Debug][sample {idx}] rm_score: {rm_score}")
        print(f"{separator}\n")

    def _format_debug_rm_input(self, batch: DataProto, idx: int, response: str) -> str:
        if "raw_prompt" in batch.non_tensor_batch:
            raw_prompt = batch.non_tensor_batch["raw_prompt"][idx]
            if isinstance(raw_prompt, np.ndarray):
                raw_prompt = raw_prompt.tolist()
            if isinstance(raw_prompt, list):
                chat = [dict(message) for message in raw_prompt]
                chat.append({"role": "assistant", "content": response})
                if getattr(self.tokenizer, "chat_template", None) is not None:
                    return self.tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
                return self._format_raw_prompt(chat)
        return f"{self._format_raw_prompt(batch.non_tensor_batch.get('raw_prompt', ['<prompt unavailable>']))}\nassistant: {response}"

    @staticmethod
    def _format_raw_prompt(raw_prompt: Any) -> str:
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        if isinstance(raw_prompt, list):
            return "\n".join(f"{message.get('role', 'unknown')}: {message.get('content', '')}" for message in raw_prompt)
        return str(raw_prompt)

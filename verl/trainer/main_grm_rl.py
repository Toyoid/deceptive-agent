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
"""
Standalone prompt-only RLFT trainer for generative reward-model judges.
"""

from __future__ import annotations

import os
import re
import uuid
from pprint import pprint
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from agent_system.grm.eval import summarize_grm_results
from agent_system.grm.reward import GrmRewardConfig, GrmRewardResult, score_grm_output
from agent_system.grm.schema import normalize_valid_tokens
from agent_system.multi_turn_rollout import adjust_batch
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    ResourcePoolManager,
    Role,
    _timer,
    apply_kl_penalty,
    compute_advantage,
    compute_log_prob_metrics,
    compute_response_mask,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config_resolvers import register_resolvers
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import ValidationGenerationsLogger
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def _create_uid_batch(batch_size: int, n_rollouts: int) -> np.ndarray:
    if batch_size % max(1, n_rollouts) != 0:
        raise ValueError(f"batch_size {batch_size} must be divisible by n_rollouts {n_rollouts}")
    values = []
    current = None
    for idx in range(batch_size):
        if idx % max(1, n_rollouts) == 0:
            current = str(uuid.uuid4())
        values.append(current)
    return np.asarray(values, dtype=object)


def _grm_reward_config(config) -> GrmRewardConfig:
    reward_cfg = _cfg_get(_cfg_get(config, "grm", {}), "reward", {})
    return GrmRewardConfig(
        invalid_reward=float(_cfg_get(reward_cfg, "invalid_reward", -0.1)),
        exact_reward=float(_cfg_get(reward_cfg, "exact_reward", 1.0)),
        same_direction_adjacent_reward=float(_cfg_get(reward_cfg, "same_direction_adjacent_reward", 0.0)),
        wrong_valid_reward=float(_cfg_get(reward_cfg, "wrong_valid_reward", -1.0)),
        score_regex=str(_cfg_get(reward_cfg, "score_regex", r"<score>\s*([0-4])\s*</score>\s*$")),
    )


class GrmRLTrainer:
    """Prompt-only GRPO trainer for generative CoT judge models."""

    def __init__(
        self,
        config,
        tokenizer,
        processor,
        role_worker_mapping: dict[Role, Any],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls=RayWorkerGroup,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name: str = "cuda",
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name

        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0
        self.actor_rollout_wg = None
        self.ref_policy_wg = None
        self.validation_generations_logger = ValidationGenerationsLogger()

        self.reward_config = _grm_reward_config(config)
        if config.algorithm.use_kl_in_reward:
            from verl.trainer.ppo import core_algos

            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)
        else:
            self.kl_ctrl_in_reward = None

        self._validate_config(train_dataset=train_dataset, val_dataset=val_dataset)
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self, train_dataset: Optional[Dataset] = None, val_dataset: Optional[Dataset] = None) -> None:
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu".
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "ppo_micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated).")

        if train_dataset is None and config.data.train_files is None:
            raise ValueError("data.train_files must be provided for GRM training.")
        assert config.actor_rollout_ref.model.path is not None, "actor_rollout_ref.model.path must point to the trainable GRM judge model."
        assert config.data.prompt_key is not None, "data.prompt_key must be set."

        if config.reward_model.enable:
            raise ValueError("reward_model.enable must be False for standalone GRM RLFT.")
        if config.auxiliary.enable:
            raise ValueError("auxiliary.enable must be False for standalone GRM RLFT.")
        if config.monitor_rollout_ref.enable or config.self_monitor.enable or config.verdict_monitor.enable:
            raise ValueError("GRM RLFT is prompt-only and does not support monitor/self-monitor/verdict-monitor modes.")
        if config.judge_model.enable:
            raise ValueError("judge_model.enable must be False for standalone GRM RLFT; labels come from the GRM dataset.")

        if config.algorithm.adv_estimator not in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.RLOO,
        ]:
            raise ValueError("GRM RLFT supports prompt-only no-critic estimators; use grpo by default.")

        if config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.RLOO,
        ]:
            assert config.actor_rollout_ref.rollout.n >= 2, (
                f"algorithm.adv_estimator={config.algorithm.adv_estimator} requires actor_rollout_ref.rollout.n >= 2."
            )
        assert config.actor_rollout_ref.rollout.n > 0, "actor_rollout_ref.rollout.n must be greater than 0."
        assert config.actor_rollout_ref.rollout.val_kwargs.n > 0, "actor_rollout_ref.rollout.val_kwargs.n must be greater than 0."
        assert not config.actor_rollout_ref.rollout.multi_turn.enable, "GRM RLFT is prompt-only; actor_rollout_ref.rollout.multi_turn.enable must be False."

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

            # The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu.
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

            if self.use_reference_policy:
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")
        if (config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss) and not self.use_reference_policy:
            raise ValueError("KL-enabled GRM training requires a reference-policy worker.")

        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.val_kwargs.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        assert config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2", "megatron"], (
            f"Unsupported actor strategy {config.actor_rollout_ref.actor.strategy!r}"
        )

        if config.data.train_batch_size * config.actor_rollout_ref.rollout.n < config.actor_rollout_ref.actor.ppo_mini_batch_size:
            raise ValueError(
                "data.train_batch_size * actor_rollout_ref.rollout.n must be at least "
                "actor_rollout_ref.actor.ppo_mini_batch_size."
            )

        try:
            reward_config = _grm_reward_config(config)
            score_pattern = re.compile(reward_config.score_regex)
        except Exception as exc:
            raise ValueError(f"grm.reward.score_regex is invalid: {exc}") from exc
        if score_pattern.groups < 1:
            raise ValueError("grm.reward.score_regex must contain a capture group for the score token.")

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler) -> None:
        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None and self.config.data.val_files is not None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            persistent_workers=self.config.data.get("dataloader_num_workers", 8) > 0,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size or len(self.val_dataset)
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        except Exception as exc:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _score_outputs(self, outputs, labels, valid_token_rows) -> list[GrmRewardResult]:
        results: list[GrmRewardResult] = []
        for output, label, valid_tokens in zip(outputs, labels, valid_token_rows):
            tokens = normalize_valid_tokens(valid_tokens)
            results.append(score_grm_output(
                output,
                label=str(label),
                valid_tokens=tokens,
                config=self.reward_config,
            ))
        return results

    def _log_val_generations_if_available(self, inputs, outputs, scores) -> None:
        generations_to_log = int(self.config.trainer.get("log_val_generations", 0))
        if generations_to_log == 0:
            return

        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[:generations_to_log]

        self.validation_generations_logger.log(
            self.config.trainer.logger,
            samples,
            self.global_steps,
            generations_to_log=generations_to_log,
        )

    def _print_train_samples_if_needed(self, batch: DataProto, outputs, results: list[GrmRewardResult]) -> None:
        samples_to_print = 3
        prompt_texts = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        labels = batch.non_tensor_batch["label"]
        valid_token_rows = batch.non_tensor_batch["valid_tokens"]

        print("\n" + "=" * 80)
        print(f"GRM TRAINING SAMPLES @ step {self.global_steps}")
        print("=" * 80)
        for idx in range(min(samples_to_print, len(outputs))):
            result = results[idx]
            print(f"[sample {idx + 1}] label={labels[idx]} valid_tokens={normalize_valid_tokens(valid_token_rows[idx])} pred={result.parsed_token} reward={result.reward}")
            print("[prompt]")
            print(prompt_texts[idx])
            print("[output]")
            print(outputs[idx])
            print("-" * 80)

    def _validate(self) -> dict[str, float]:
        all_results: list[GrmRewardResult] = []
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        for batch_dict in self.val_dataloader:
            batch = self._collect_rollout_batch(DataProto.from_single_dict(batch_dict), is_train=False)
            prompt_texts = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            results = self._score_outputs(
                outputs,
                batch.non_tensor_batch["label"],
                batch.non_tensor_batch["valid_tokens"],
            )
            all_results.extend(results)
            sample_inputs.extend(prompt_texts)
            sample_outputs.extend(outputs)
            sample_scores.extend([row.reward for row in results])
        self._log_val_generations_if_available(sample_inputs, sample_outputs, sample_scores)
        return summarize_grm_results(all_results, prefix="val/grm")

    def init_workers(self) -> None:
        print("\n" + "="*80)
        print("INITIALIZING WORKERS")
        print("="*80)

        self.resource_pool_manager.create_resource_pool()
        resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        resource_pool_to_cls[actor_pool]["actor_rollout"] = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        if self.use_reference_policy:
            ref_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            resource_pool_to_cls[ref_pool]["ref"] = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )

        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            all_wg.update(wg_dict.spawn(prefix_set=class_dict.keys()))

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()


    def _save_checkpoint(self) -> None:
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        print(f"local_global_step_folder: {local_global_step_folder}")

        actor_local_path = os.path.join(local_global_step_folder, "actor")
        
        actor_remote_path = None
        if self.config.trainer.default_hdfs_dir is not None:
            actor_remote_path = os.path.join(
                self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor"
            )
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=self.config.trainer.get("max_actor_ckpt_to_keep", None),
        )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = {"main": self.train_dataloader.state_dict()}
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w", encoding="utf-8") as f:
            f.write(str(self.global_steps))

    @staticmethod
    def _parse_save_steps(save_steps) -> set[int]:
        if save_steps is None:
            return set()
        if isinstance(save_steps, int):
            return {save_steps}
        if isinstance(save_steps, str):
            text = save_steps.strip()
            if not text:
                return set()
            text = text.strip("[]")
            return {int(item.strip()) for item in text.split(",") if item.strip()}
        return {int(step) for step in save_steps}

    def _should_save_checkpoint(self, is_last_step: bool) -> bool:
        save_freq = self.config.trainer.get("save_freq", -1)
        if save_freq > 0 and (is_last_step or self.global_steps % save_freq == 0):
            return True

        save_steps = self._parse_save_steps(self.config.trainer.get("save_steps", []))
        return self.global_steps in save_steps

    def _load_checkpoint(self) -> None:
        if self.config.trainer.resume_mode == "disable":
            return

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(str(global_step_folder).split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            if isinstance(dataloader_state_dict, dict) and "main" in dataloader_state_dict:
                self.train_dataloader.load_state_dict(dataloader_state_dict["main"])
            else:
                # Backward compatibility for checkpoints saved before wrapping dataloader state.
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, world_size: int, metrics: dict[str, Any]) -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix="grm_global_seqlen")
        metrics.update(global_balance_stats)

    def fit(self) -> None:
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_dataloader is not None and self.config.trainer.get("val_before_train", True):
            metrics = self._validate()
            pprint(f"Initial GRM validation metrics: {metrics}")
            logger.log(data=metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                logger.close()
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="GRM Training Progress")
        self.global_steps += 1
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                is_last_step = self.global_steps >= self.total_training_steps
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                with _timer("step", timing_raw):
                    # ==================================================
                    #                 GRM Generation
                    # ==================================================
                    with _timer("gen", timing_raw):
                        batch = self._collect_rollout_batch(batch, is_train=True)

                    # ==================================================
                    #             GRM Reward Computation
                    # ==================================================
                    with _timer("reward", timing_raw):
                        batch, reward_metrics, train_outputs, train_results = self._score_batch(
                            batch,
                            prefix="grm",
                            return_details=True,
                        )
                        metrics.update(reward_metrics)
                        self._print_train_samples_if_needed(batch, train_outputs, train_results)

                    # ==================================================
                    #                Batch Preprocessing
                    # ==================================================
                    original_size = len(batch)
                    batch = adjust_batch(
                        config=self.config,
                        data=batch,
                        world_size=self.actor_rollout_wg.world_size,
                    )
                    if len(batch) != original_size:
                        raise ValueError(
                            "GRM batches must already satisfy distributed batch divisibility. "
                            "Adjust data.train_batch_size, rollout.n, or micro batch sizes."
                        )
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, world_size=self.actor_rollout_wg.world_size, metrics=metrics)
                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # ==================================================
                    #             Recompute old_log_probs
                    # ==================================================
                    with _timer("old_log_prob", timing_raw):
                        # Compute log prob metrics for actor batch
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch, old_log_prob_metrics = compute_log_prob_metrics(
                            batch,
                            old_log_prob,
                            self.config.actor_rollout_ref.actor.loss_agg_mode,
                            metric_prefix="grm",
                        )
                        metrics.update(old_log_prob_metrics)

                    # ==================================================
                    #           Reference Log-Prob Computation
                    # ==================================================
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if self.ref_in_actor:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # ==================================================
                    #              Advantage Computation
                    # ==================================================
                    with _timer("adv", timing_raw):
                        # apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update({f"grm/{key}": value for key, value in kl_metrics.items()})
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                            multi_turn=False,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )

                    # ==================================================
                    #                   Update Actor
                    # ==================================================
                    with _timer("update_actor", timing_raw):
                        batch.meta_info["multi_turn"] = False
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # validate
                if self.val_dataloader is not None and self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with _timer("testing", timing_raw):
                        metrics.update(self._validate())

                # save
                if self._should_save_checkpoint(is_last_step):
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                # training metrics
                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                # collect metrics
                metrics.update(
                    compute_data_metrics(
                        batch=batch,
                        use_critic=False,
                        metric_prefix="grm",
                        include_episode_metrics=False,
                    )
                )

                total_tokens = sum(batch.meta_info.get("global_token_num", []))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(
                    total_num_tokens=total_tokens,
                    timing_raw=timing_raw,
                    n_gpus=self.resource_pool_manager.get_n_gpus(),
                ))

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    progress_bar.close()
                    logger.close()
                    return

    def _collect_rollout_batch(self, prompt_batch: DataProto, *, is_train: bool) -> DataProto:
        rollout_n = int(self.config.actor_rollout_ref.rollout.n if is_train else self.config.actor_rollout_ref.rollout.val_kwargs.n)
        actor_batch = prompt_batch.repeat(repeat_times=rollout_n, interleave=True)
        actor_batch.non_tensor_batch["uid"] = _create_uid_batch(len(actor_batch), rollout_n)
        actor_batch.non_tensor_batch["traj_uid"] = np.asarray([str(uuid.uuid4()) for _ in range(len(actor_batch))], dtype=object)

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_keys_pop = []
        for key in ("raw_prompt_ids", "data_source", "multi_modal_data", "multi_modal_inputs", "raw_prompt"):
            if key in actor_batch.non_tensor_batch:
                non_tensor_keys_pop.append(key)
        gen_batch = actor_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_keys_pop
        )
        gen_batch.meta_info = actor_batch.meta_info

        if not is_train:
            gen_batch.meta_info.update({
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            })

        # rollout batch
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, self.actor_rollout_wg.world_size)
        batch_output_padded = self.actor_rollout_wg.generate_sequences(gen_batch_padded)
        # unpad
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)
        
        return actor_batch.union(batch_output)

    def _score_batch(
        self,
        batch: DataProto,
        *,
        prefix: str,
        return_details: bool = False,
    ):
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        labels = np.asarray(batch.non_tensor_batch["label"], dtype=object)
        valid_token_rows = batch.non_tensor_batch["valid_tokens"]

        results = self._score_outputs(outputs, labels, valid_token_rows)

        response_mask = compute_response_mask(batch)
        token_level_scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)  # zeros_like put token_level_scores on the same device as responses tensor
        reward_tensor = torch.tensor(
            [row.reward for row in results], dtype=torch.float32, device=token_level_scores.device
        )  # (bsz,)
        response_lengths = response_mask.to(device=token_level_scores.device).long().sum(dim=-1)
        eos_mask_idx = response_lengths.sub(1).clamp_min(0)
        row_idx = torch.arange(token_level_scores.size(0), device=token_level_scores.device)
        token_level_scores[row_idx, eos_mask_idx] = reward_tensor
        batch.batch["response_mask"] = response_mask
        batch.batch["token_level_scores"] = token_level_scores
        batch.non_tensor_batch["grm_pred_token"] = np.asarray([
            "" if row.parsed_token is None else row.parsed_token for row in results
        ], dtype=object)
        batch.non_tensor_batch["grm_exact"] = np.asarray([row.exact for row in results], dtype=np.float32)
        batch.non_tensor_batch["grm_invalid_format"] = np.asarray([row.invalid_format for row in results], dtype=np.float32)
        batch.non_tensor_batch["grm_wrong_profile_token"] = np.asarray([row.wrong_profile_token for row in results], dtype=np.float32)
        batch.non_tensor_batch["grm_adjacent_same_direction"] = np.asarray([row.adjacent_same_direction for row in results], dtype=np.float32)

        metrics = summarize_grm_results(results, prefix=prefix)
        if return_details:
            return batch, metrics, outputs, results
        return batch, metrics

@hydra.main(config_path="config", config_name="grm_trainer", version_base=None)
def main(config) -> None:
    register_resolvers()
    run_grm_rl(config)


def run_grm_rl(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )
    runner = GrmTaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class GrmTaskRunner:
    def run(self, config) -> None:
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        register_resolvers()
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        tokenizer = hf_tokenizer(
            local_path,
            trust_remote_code=config.data.get("trust_remote_code", False),
            apply_chat_template_default_kwargs=config.actor_rollout_ref.model.get("chat_template_kwargs", None),
        )
        processor = hf_processor(
            local_path,
            trust_remote_code=config.data.get("trust_remote_code", False),
            use_fast=True,
            apply_chat_template_default_kwargs=config.actor_rollout_ref.model.get("chat_template_kwargs", None),
        )

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            from verl.workers.fsdp_workers import ActorRolloutRefWorker
            from verl.single_controller.ray import RayWorkerGroup

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import ActorRolloutRefWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError(f"Unsupported actor strategy {config.actor_rollout_ref.actor.strategy!r}")

        actor_pool_id = "grm_actor_pool"
        resource_pool_spec = {actor_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        role_worker_mapping = {Role.ActorRollout: ray.remote(actor_rollout_cls)}
        mapping = {Role.ActorRollout: actor_pool_id}
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
            mapping[Role.RefPolicy] = actor_pool_id

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = None
        if config.data.val_files is not None:
            val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = GrmRLTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping),
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()

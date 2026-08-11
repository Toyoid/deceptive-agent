# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import copy
from dataclasses import dataclass
import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from verl.utils.dataset.model_inputs import process_multimodal_chat
from agent_system.environments.prompts.monitor_prompt import (
    MAXIMIN_MONITOR_PROMPT,
    CRITIQUE_MONITOR_PROMPT,
    build_verdict_monitor_prompt,
)
from agent_system.environments.prompts import DEFAULT_SYSTEM_PROMPT
from agent_system.environments import EnvironmentManagerBase
from agent_system.utils.metric_contract import EPISODE_METRIC_PREFIX
from agent_system.utils.active_rollout import ActiveIndexMap
from agent_system.self_monitor import parse_self_monitor_batch
from agent_system.verdict_monitor import build_verdict_monitor_background, constrained_probs_to_binary_penalties
from agent_system.grm.io import append_judge_samples
from agent_system.grm.schema import GrmJudgeSample
from agent_system.monitor_action import (
    correct_no_issue_from_probs,
    correct_no_issue_from_token,
    parse_monitor_action,
    validate_behavior_evidence_link,
    validate_issue_anchors,
)
from agent_system.judge.score_profiles import (
    ISSUE_ACTION_SCORE_PROFILE,
    NO_ISSUE_ACTION_SCORE_PROFILE,
    resolve_score_profile,
)
from typing import Any, List, Dict, Callable, Tuple, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


@dataclass(frozen=True)
class JudgeRequests:
    sample_idx: int
    action_type: str
    score_profile_name: str
    prompt: List[Dict[str, str]]
    image: Any = None


@dataclass(frozen=True)
class JudgeBatchResult:
    scores: np.ndarray
    score_tokens: List[Optional[str]]
    errors: List[Optional[str]]
    correct_no_issue: np.ndarray

class TrajectoryCollector:
    def __init__(
        self, 
        config, 
        tokenizer: PreTrainedTokenizer, 
        processor=None,
        monitor_tokenizer: PreTrainedTokenizer = None,
        monitor_processor=None,
        judge_tokenizer: PreTrainedTokenizer = None,
        judge_processor=None,
        verdict_monitor_tokenizer: PreTrainedTokenizer = None,
        verdict_monitor_processor=None,
    ):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
            monitor_tokenizer: Tokenizer for monitor model
            monitor_processor: Processor for monitor model multimodal inputs
            judge_tokenizer: Tokenizer for judge model (for critique preprocessing)
            verdict_monitor_tokenizer: Tokenizer for verdict monitor model
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.judge_tokenizer = judge_tokenizer
        self.judge_processor = judge_processor
        self._cot_judge_scorer = None
        if config.monitor_rollout_ref.enable:
            assert monitor_tokenizer is not None, "monitor tokenizer should be provided when monitor is enabled"
            self.monitor_tokenizer = monitor_tokenizer
            self.monitor_processor = monitor_processor
        if config.judge_model.enable:
            judge_backend = config.judge_model.get("backend", "constrained_logits")
            if judge_backend == "constrained_logits":
                assert judge_tokenizer is not None, "judge tokenizer should be provided when constrained-logit judge is enabled"
            elif judge_backend != "api_cot":
                raise ValueError(f"Unsupported judge_model.backend={judge_backend!r}")
        if config.verdict_monitor.enable:
            assert verdict_monitor_tokenizer is not None, "verdict monitor tokenizer should be provided when verdict_monitor is enabled"
            self.verdict_monitor_tokenizer = verdict_monitor_tokenizer
            self.verdict_monitor_processor = verdict_monitor_processor

    def _judge_score_profiles(self):
        from agent_system.judge.score_profiles import load_score_profiles

        return load_score_profiles(self.config.judge_model)

    def _should_dump_grm_judge_data(self, global_step: Optional[int]) -> bool:
        if not self.config.trainer.get("grm_judge_data_dir", None):
            return False
        if global_step is None:
            return False
        if not self.config.monitor_rollout_ref.enable_train_monitor:
            return False
        dump_freq = int(self.config.trainer.get("grm_judge_data_freq", 20))
        if dump_freq <= 0:
            return False
        return global_step == 1 or global_step % dump_freq == 0

    @staticmethod
    def _group_judge_requests_by_profile(
        judge_requests: List[JudgeRequests],
    ) -> Dict[str, List[JudgeRequests]]:
        grouped: Dict[str, List[JudgeRequests]] = {}
        for request in judge_requests:
            grouped.setdefault(request.score_profile_name, []).append(request)
        return grouped

    @staticmethod
    def _create_uid_batch(
        batch_size: int,
        n_rollouts: int,
    ) -> np.ndarray:
        """
        Create a batch of unique identifiers (UIDs) for grouping of rollouts.
        rollouts within the same group is assigned the same UID.

        Args:
            batch_size (int): Total number of rollouts in the batch, assuming n_groups * n_rollouts.
            n_rollouts (int): Number of rollouts per group.

        Returns:
            np.ndarray: Array of UIDs with shape (batch_size,).
        """
        assert batch_size % max(1, n_rollouts) == 0, f"batch_size {batch_size} must be divisible by n_rollouts {n_rollouts}"
        
        # TODO: only support interleaved grouping for now, can add non-interleaved grouping if needed
        if n_rollouts > 0: 
            uid_batch = []
            for i in range(batch_size): 
                if i % n_rollouts == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(batch_size)], dtype=object)
        
        return uid_batch

    def _debug_print_judge_samples(
        self,
        processed_judge_samples: List[dict],
        judge_requests: List[JudgeRequests],
        score_profile_name: str,
        constrained_scores,
        constrained_token_probs,
    ) -> None:
        debug_print_samples = 2
        if debug_print_samples <= 0 or len(processed_judge_samples) == 0:
            return

        num_samples = min(debug_print_samples, len(processed_judge_samples), len(judge_requests))
        probs_array = constrained_token_probs.numpy() if hasattr(constrained_token_probs, "numpy") else np.asarray(constrained_token_probs)
        profile = resolve_score_profile(self._judge_score_profiles(), score_profile_name)
        valid_tokens = list(profile.valid_tokens)
        token_weights = list(profile.token_weights)
        constrained_top_k = self.config.judge_model.get("constrained_top_k", -1)

        print("\n" + "=" * 120)
        print(
            f"[Judge Debug] Showing {num_samples}/{len(processed_judge_samples)} samples | "
            f"score_profile={score_profile_name} | valid_tokens={valid_tokens} | constrained_top_k={constrained_top_k}"
        )
        print("=" * 120)

        for idx in range(num_samples):
            request = judge_requests[idx]
            prompt_ids = processed_judge_samples[idx]["raw_prompt_ids"]
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()
            prompt_text = self.judge_tokenizer.decode(prompt_ids, skip_special_tokens=False)

            prob_row = probs_array[idx].tolist()
            best_idx = int(np.argmax(prob_row))
            prob_summary = ", ".join(
                f"{token}={prob:.4f}" for token, prob in zip(valid_tokens, prob_row)
            )

            print(
                f"[Judge Debug] Queued sample {idx + 1}/{num_samples} | "
                f"source_sample={request.sample_idx} | action_type={request.action_type}"
            )
            print(f"Score: {float(constrained_scores[idx]):.4f}")
            print(
                f"Argmax token: {valid_tokens[best_idx]} "
                f"(weight={float(token_weights[best_idx]):.4f}, prob={prob_row[best_idx]:.4f})"
            )
            print(f"Token probs: {prob_summary}")
            print("-" * 120)
            print(prompt_text)
            print("=" * 120)

    def _debug_print_verdict_monitor_samples(
        self,
        raw_prompt_ids,
        constrained_scores,
        constrained_token_probs,
        penalties,
    ) -> None:
        debug_print_samples = self.config.verdict_monitor.get("debug_print_samples", 2)
        if debug_print_samples <= 0 or len(raw_prompt_ids) == 0:
            return

        num_samples = min(debug_print_samples, len(raw_prompt_ids))
        probs_array = constrained_token_probs.numpy() if hasattr(constrained_token_probs, "numpy") else np.asarray(constrained_token_probs)
        valid_tokens = list(self.config.verdict_monitor.valid_tokens)
        token_weights = list(self.config.verdict_monitor.token_weights)
        constrained_top_k = self.config.verdict_monitor.get("constrained_top_k", -1)

        print("\n" + "=" * 120)
        print(
            f"[Verdict Monitor Debug] Showing {num_samples}/{len(raw_prompt_ids)} samples | "
            f"valid_tokens={valid_tokens} | constrained_top_k={constrained_top_k}"
        )
        print("=" * 120)

        for idx in range(num_samples):
            prompt_ids = raw_prompt_ids[idx]
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()
            prompt_text = self.verdict_monitor_tokenizer.decode(prompt_ids, skip_special_tokens=False)

            prob_row = probs_array[idx].tolist()
            best_idx = int(np.argmax(prob_row))
            prob_summary = ", ".join(
                f"{token}={prob:.4f}" for token, prob in zip(valid_tokens, prob_row)
            )

            print(f"[Verdict Monitor Debug] Sample {idx + 1}/{num_samples}")
            print(f"Score: {float(constrained_scores[idx]):.4f}")
            print(f"Penalty: {float(penalties[idx]):.4f}")
            print(
                f"Argmax token: {valid_tokens[best_idx]} "
                f"(weight={float(token_weights[best_idx]):.4f}, prob={prob_row[best_idx]:.4f})"
            )
            print(f"Token probs: {prob_summary}")
            print("-" * 120)
            print(prompt_text)
            print("=" * 120)

    def _debug_print_cot_judge_samples(
        self,
        judge_requests: List[JudgeRequests],
        score_profile_name: str,
        cot_result,
    ) -> None:
        debug_print_samples = 2
        if debug_print_samples <= 0 or len(cot_result.scores) == 0:
            return

        num_samples = min(debug_print_samples, len(cot_result.scores), len(judge_requests))
        profile = resolve_score_profile(self._judge_score_profiles(), score_profile_name)
        valid_tokens = list(profile.valid_tokens)
        token_weights = list(profile.token_weights)

        print("\n" + "=" * 120)
        print(
            f"[CoT Judge Debug] Showing {num_samples}/{len(cot_result.scores)} samples | "
            f"score_profile={score_profile_name} | valid_tokens={valid_tokens}"
        )
        print("=" * 120)

        for idx in range(num_samples):
            request = judge_requests[idx]
            parsed_token = cot_result.parsed_tokens[idx]
            token_summary = "unparsed"
            if parsed_token in valid_tokens:
                token_idx = valid_tokens.index(parsed_token)
                token_summary = f"{parsed_token} (weight={float(token_weights[token_idx]):.4f})"

            print(
                f"[CoT Judge Debug] Queued sample {idx + 1}/{num_samples} | "
                f"source_sample={request.sample_idx} | action_type={request.action_type}"
            )
            print(f"Score: {float(cot_result.scores[idx]):.4f}")
            print(f"Parsed token: {token_summary}")
            if cot_result.errors[idx] is not None:
                print(f"Error: {cot_result.errors[idx]}")
            print("-" * 120)
            print(cot_result.raw_outputs[idx])
            print("=" * 120)

    @staticmethod
    def _process_chat_to_model_inputs(
        chat: List[Dict[str, str]],
        obs_image,
        tokenizer: PreTrainedTokenizer,
        processor,
        max_prompt_length: int,
        truncation: str,
    ) -> dict:
        """
        Process a chat structure into model-ready inputs including tokenization,
        multimodal handling, and position ID computation.
        
        Parameters:
            chat (List[Dict[str, str]]): Chat structure with role and content
            obs_image: Image observation (None for text-only)
            tokenizer (PreTrainedTokenizer): Tokenizer for encoding
            processor: Image processor for multimodal inputs
            max_prompt_length (int): Maximum prompt length
            truncation (str): Truncation strategy ('left', 'right', 'middle', 'error')
        
        Returns:
            dict: Contains input_ids, attention_mask, position_ids, raw_prompt_ids,
                  and optionally multi_modal_data and multi_modal_inputs
        """
        if obs_image is not None:
            processed = process_multimodal_chat(
                messages=chat,
                tokenizer=tokenizer,
                processor=processor,
                images=[process_image(obs_image)],
                max_length=max_prompt_length,
                truncation=truncation,
            )
            processed.pop("raw_prompt", None)
            return processed

        prompt_with_chat_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=tokenizer,
            max_length=max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation=truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        raw_prompt_ids = tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)
        if len(raw_prompt_ids) > max_prompt_length:
            if truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
            elif truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
            elif truncation == "middle":
                left_half = max_prompt_length // 2
                right_half = max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {max_prompt_length}.")

        return {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
        }

    def build_single_actor_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        infos: List[Dict]
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
            infos (List[Dict]): List of info dicts for each sample in the batch

        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        system_raw = infos[item].get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        format_prompt = infos[item].get("format_prompt", "")
        system_prompt = system_raw + f"\n{format_prompt}" if format_prompt else system_raw

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        chat = [
            {"content": system_prompt, "role": "system"},
            {"content": obs_content,   "role": "user"},
        ]
        
        # Process chat to model inputs using shared helper
        row_dict = self._process_chat_to_model_inputs(
            chat=chat,
            obs_image=obs_image,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.truncation,
        )

        # Add sample-specific fields
        row_dict.update({
            'anchor_obs': _obs_anchor,
            'index': item,  # TODO: check if this is needed
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = copy.deepcopy(chat)
        
        return row_dict

    def build_single_monitor_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        infos: Optional[List[Dict]] = None,
    ) -> dict:
        # Get observation components
        monitor_background = obs['monitor_background'][item]
        agent_trajectory = obs['agent_trajectory'][item]
        monitor_images = obs.get('monitor_image', None)
        task_type = obs['task_type']  # NOTE: assume task_type is the same for all in the batch
        monitor_image = monitor_images[item] if monitor_images is not None else None
        # TODO: the multi-modal processing for monitor has not been tested yet

        monitor_prompt = (
            MAXIMIN_MONITOR_PROMPT
            if self.config.monitor_rollout_ref.enable_train_monitor
            else CRITIQUE_MONITOR_PROMPT
        )

        user_content = monitor_prompt.format_user_message(
            task_type=task_type,
            background=monitor_background,
            behavior_under_review=agent_trajectory,
        )
        
        chat = [
            {"content": monitor_prompt.system_prompt, "role": "system"},
            {"content": user_content, "role": "user"}
        ]
        
        # Process chat to model inputs using shared helper
        row_dict = self._process_chat_to_model_inputs(
            chat=chat,
            obs_image=monitor_image,
            tokenizer=self.monitor_tokenizer,
            processor=self.monitor_processor,
            max_prompt_length=self.config.monitor_rollout_ref.data.max_prompt_length,
            truncation=self.config.monitor_rollout_ref.data.truncation,
        )
        #TODO: Currently not sure where config.monitor_rollout_ref.data.max_response_length will be used for monitor

        # Add sample-specific fields
        row_dict.update({
            'raw_prompt': copy.deepcopy(chat),
            'data_source': gen_batch.non_tensor_batch['data_source'][item], 
            'episode_rewards': gen_batch.non_tensor_batch['episode_rewards'][item],
            'agent_trajectory': agent_trajectory,
        })
        
        return row_dict

    def build_single_verdict_monitor_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
        infos: Optional[List[Dict]] = None,
    ) -> dict:
        monitor_background = obs['monitor_background'][item]
        agent_trajectory = obs['agent_trajectory'][item]
        monitor_images = obs.get('monitor_image', None)
        task_type = obs['task_type']
        monitor_image = monitor_images[item] if monitor_images is not None else None

        chat = build_verdict_monitor_prompt(
            task_type=task_type,
            background=monitor_background,
            behavior_under_review=agent_trajectory,
        )

        row_dict = self._process_chat_to_model_inputs(
            chat=chat,
            obs_image=monitor_image,
            tokenizer=self.verdict_monitor_tokenizer,
            processor=self.verdict_monitor_processor,
            max_prompt_length=self.config.verdict_monitor.max_prompt_length,
            truncation=self.config.verdict_monitor.truncation,
        )

        row_dict.update({
            'raw_prompt': copy.deepcopy(chat),
            'data_source': gen_batch.non_tensor_batch['data_source'][item],
            'episode_rewards': gen_batch.non_tensor_batch['episode_rewards'][item],
        })
        return row_dict

    @staticmethod
    def preprocess_batch(
        gen_batch: DataProto, 
        obs: Dict, 
        infos: List[Dict],
        single_preprocessor: Callable[[int, DataProto, Dict, List[Dict]], dict],
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
            infos (List[Dict]): List of info dicts for each sample in the batch, can contain additional metadata for processing
            single_preprocessor: A callable that processes a single sample. Should have signature:
                (item: int, gen_batch: DataProto, obs: Dict, infos: List[Dict]) -> dict
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = single_preprocessor(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
                infos=infos
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch

    def gather_rollout_data(
        self,
        total_batch_list: List[List[Dict]],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        traj_uid: np.ndarray,
        tool_callings: np.ndarray,
        trust_penalties: np.ndarray | None = None,
    ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        #  Structure of total_batch_list:
        # Outer list: length = batch_size (one per environment/trajectory)
        # Inner list: length = number of steps that environment took
        # Each dict: data for one (environment, step) pair
        # total_batch_list = [
        #     [step0_env0, step1_env0, step2_env0],  # env 0's trajectory (3 steps)
        #     [step0_env1, step1_env1],               # env 1's trajectory (2 steps, finished early)
        #     [step0_env2, step1_env2, step2_env2],  # env 2's trajectory (3 steps)
        #     ...
        # ]

        batch_size = len(total_batch_list)
        assert len(trust_penalties) == batch_size if trust_penalties is not None else True, "trust_penalties length should match batch_size if provided"

        success_rate = {}
        episode_metrics = {}
        legacy_metrics = {}
        for key, value in success.items():
            if key.endswith("_rate"):
                success_rate[key] = np.mean(value)
            elif key.startswith(EPISODE_METRIC_PREFIX):
                episode_metrics[key] = np.asarray(value)
            else:
                legacy_metrics[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # trust penalties
                    if trust_penalties is not None:
                        data['trust_penalties'] = trust_penalties[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value
                    for key, value in legacy_metrics.items():
                        data[key] = value
                    for key, value in episode_metrics.items():
                        data[key] = value[bs]

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
        self,
        gen_batch: DataProto,
        actor_rollout_wg,
        monitor_wg,
        verdict_monitor_wg,
        judge_wg,
        envs: EnvironmentManagerBase,
        rollout_n: int,
        monitor_rollout_n: int,
        global_step: Optional[int] = None,
    ) -> Tuple[Dict, DataProto | None]:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            monitor_wg (WorkerGroup): Worker group containing the monitor model
            verdict_monitor_wg (WorkerGroup): Worker group containing the verdict monitor model
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
            judge_wg (WorkerGroup, optional): Worker group containing the judge model for scoring monitor's critiques.

        Returns:
            actor_batch_dict:
                total_batch_list (List[Dict]): List of trajectory data for each environment
                episode_rewards (np.ndarray): Total rewards for each environment
                episode_lengths (np.ndarray): Total steps for each environment
                success (Dict[str, np.ndarray]): Success samples for each environment
                traj_uid (np.ndarray): Trajectory unique identifiers
            monitor_batch (DataProto): Output batch from monitor rollout (if enabled)
        """
        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        if obs['text'] is not None:
            length_obs = len(obs['text'])
        elif obs['image'] is not None:
            length_obs = len(obs['image'])
        else:
            length_obs = len(infos)
        assert len(gen_batch.batch) == length_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {length_obs}"
        
        uid_batch = self._create_uid_batch(batch_size, rollout_n)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        monitor_batch = None
        self_monitor_trust_penalties = np.zeros(batch_size, dtype=np.float32) if self.config.self_monitor.enable else None
        verdict_monitor_trust_penalties = None
        previous_text_actions = [""] * batch_size

        # Trajectory collection loop
        rollout_max_steps = envs.get_rollout_max_steps()
        for _step in range(rollout_max_steps):
            active = ActiveIndexMap.from_done(is_done)
            assert active.batch_size == batch_size, "ActiveIndexMap batch size does not match the original batch size"
            if not active.has_active:
                break

            prompt_source = envs.actor_prompt_source(_step)
            if prompt_source == "dataset":
                # Reuse the dataset-prepared prompt on bootstrap turns.
                batch = gen_batch.clone()
                if "anchor_obs" not in batch.non_tensor_batch:
                    batch.non_tensor_batch["anchor_obs"] = np.full(batch_size, None, dtype=object)
            elif prompt_source == "env_obs":
                batch = self.preprocess_batch(
                    gen_batch=gen_batch,
                    obs=obs,
                    infos=infos,
                    single_preprocessor=self.build_single_actor_sample,
                )
            else:
                raise ValueError(f"Unsupported actor prompt source: {prompt_source}")

            # Keep global env slots stable; only compact the model-generation payload.
            active_batch = batch.select_idxs(active.active_idx)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in active_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in active_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in active_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = active_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info
            # print meta_info for debugging
            # print(f"\n\n[DEBUG] Rollout Loop Step {_step}: batch_input.meta_info = {batch_input.meta_info}\n\n")
            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            # unpad
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            active_batch.non_tensor_batch['uid'] = active.select_array(uid_batch, dtype=object)
            active_batch.non_tensor_batch['traj_uid'] = active.select_array(traj_uid, dtype=object)

            active_batch = active_batch.union(batch_output)
            
            active_text_actions = self.tokenizer.batch_decode(active_batch.batch['responses'], skip_special_tokens=True)
            self_monitor_batch = None
            if self.config.self_monitor.enable:
                self_monitor_batch = parse_self_monitor_batch(active_text_actions)
                active_batch.non_tensor_batch.update(self_monitor_batch)

            text_actions = active.scatter_actions(active_text_actions, previous_text_actions)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)
            previous_text_actions = text_actions
            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                env_action_valid = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                env_action_valid = np.ones(batch_size, dtype=bool)

            # self-monitor baseline: collect self-monitoring signals and apply trust penalties
            if self_monitor_batch is not None:
                final_action_valid = env_action_valid.copy()
                self_monitor_is_valid = np.asarray(self_monitor_batch['self_monitor_is_valid'], dtype=bool)
                final_action_valid[active.active_idx] = np.logical_and(
                    env_action_valid[active.active_idx],
                    self_monitor_is_valid,
                )
                # any unsafe step will cause the whole trajectory to be penalized
                self_monitor_trust_penalties[active.active_idx] = np.logical_or(
                    self_monitor_trust_penalties[active.active_idx] > 0.0,
                    self_monitor_batch['self_monitor_is_unsafe'],
                ).astype(np.float32)
            else:
                final_action_valid = env_action_valid

            active_batch.non_tensor_batch['is_action_valid'] = active.select_array(final_action_valid)

            if 'tool_calling' in infos[0]:
                tool_callings[active.active_idx] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active.active_idx]
            # Only active environments contribute step-level accounting.
            episode_rewards[active.active_idx] += torch_to_numpy(rewards)[active.active_idx]
            episode_lengths[active.active_idx] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            active_batch.non_tensor_batch['rewards'] = active.select_array(torch_to_numpy(rewards, is_object=True))
            active_batch.non_tensor_batch['active_masks'] = active.active_flags(dtype=object)

            # log for retroactive analysis and judge_model input if judge enabled
            active_batch.non_tensor_batch['user_inputs'] = active.select_info_values(infos, 'user_input')
            active_batch.non_tensor_batch['system_infos'] = active.select_info_values(infos, 'evidence')

            if self.config.monitor_rollout_ref.enable or self.config.verdict_monitor.enable:
                active_batch.non_tensor_batch['monitor_background'] = active.select_array(
                    next_obs['monitor_background'], dtype=object
                )
                active_batch.non_tensor_batch['agent_trajectory'] = active.select_array(
                    next_obs['agent_trajectory'], dtype=object
                )
                if next_obs.get('monitor_image', None) is not None:
                    active_batch.non_tensor_batch['monitor_image'] = active.select_array(
                        next_obs['monitor_image'], dtype=object
                    )
            
            active_batch.check_consistency()

            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(active_batch)

            active.append_active_records(
                total_batch_list=total_batch_list,
                total_infos=total_infos,
                active_records=batch_list,
                infos=infos,
            )

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break

        # Environment stepping intentionally remains full-batch. Model generation and stored
        # trajectory records are active-only, so downstream PPO/monitor processing sees only
        # real rollout records while environment slots stay stable.

        actor_episode_batch = None
        if self.config.monitor_rollout_ref.enable or self.config.verdict_monitor.enable:
            # Extract the last real step (active_masks == True) from each trajectory
            actor_episode_list = [None for _ in range(batch_size)]
            for bs in range(batch_size):
                # find last real step (active_masks == True)
                last_real = None
                for data in reversed(total_batch_list[bs]):
                    if data.get('active_masks', True):
                        last_real = data
                        break
                if last_real is None:
                    # fallback to the very last entry
                    last_real = total_batch_list[bs][-1]
                last_real["episode_rewards"] = episode_rewards[bs]
                actor_episode_list[bs] = last_real

            actor_episode_batch = DataProto.from_single_dict(
                data=collate_fn(actor_episode_list)
            )

        # monitor rollout on the episode data of actor
        if self.config.monitor_rollout_ref.enable and monitor_wg is not None:
            # call monitor rollout with one-sample-per-env batch
            monitor_batch = self.monitor_rollout(
                actor_batch=actor_episode_batch,
                monitor_wg=monitor_wg,
                infos=infos,
                judge_wg=judge_wg,
                monitor_rollout_n=monitor_rollout_n,
                global_step=global_step,
            )
        # in case we should enable monitor rollout but also need one pure actor rollout, we set monitor_wg as None
        elif self.config.monitor_rollout_ref.enable and monitor_wg is None:
            print("NOTE: Monitor worker group set as None, skipping monitor rollout...")

        if self.config.verdict_monitor.enable and verdict_monitor_wg is not None:
            verdict_monitor_trust_penalties = self.verdict_monitor_score(
                actor_batch=actor_episode_batch,
                verdict_monitor_wg=verdict_monitor_wg,
                infos=infos,
            )
        # in case we should enable verdict monitor but also need one pure actor rollout, we set verdict_monitor_wg as None
        elif self.config.verdict_monitor.enable and verdict_monitor_wg is None:
            print("NOTE: Verdict monitor worker group set as None, skipping verdict monitor...")

        success: Dict[str, np.ndarray] = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards, 
            episode_lengths=episode_lengths,
        )
        
        actor_batch_dict = {
            'total_batch_list': total_batch_list,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'success': success,
            'traj_uid': traj_uid,
            'tool_callings': tool_callings,
            'self_monitor_trust_penalties': self_monitor_trust_penalties,
            'verdict_monitor_trust_penalties': verdict_monitor_trust_penalties,
        }

        return actor_batch_dict, monitor_batch
    
    def monitor_rollout(
        self,
        actor_batch: DataProto,
        monitor_wg,
        infos: List[Dict],
        judge_wg,
        monitor_rollout_n: int,
        global_step: Optional[int] = None,
    ) -> DataProto:
        assert monitor_wg is not None, "monitor worker group should not be None for monitor rollout"
        assert self.config.monitor_rollout_ref.rollout.n > 0, "monitor rollout n should be greater than 0"

        monitor_gen_batch = actor_batch.repeat(repeat_times=monitor_rollout_n, interleave=True)  # repeat returns a new independent DataProto    
        batch_size = len(monitor_gen_batch.batch)
        # NOTE: len(infos) will be different with batch_size if monitor_rollout_ref.rollout.n > 1

        # create monitor uid and traj_uid
        uid_batch = self._create_uid_batch(
            batch_size,
            monitor_rollout_n
        )
        traj_uid = np.array(
            [str(uuid.uuid4()) for _ in range(batch_size)], dtype=object
        )

        # build monitor input
        monitor_obs = {
            'task_type': infos[0]['task_type'],  # NOTE: assume all in the batch are from the same task_type
            'monitor_background': monitor_gen_batch.non_tensor_batch['monitor_background'],
            'agent_trajectory': monitor_gen_batch.non_tensor_batch['agent_trajectory'],
            'monitor_image': monitor_gen_batch.non_tensor_batch.get('monitor_image', None),
        }
        batch = self.preprocess_batch(
            gen_batch=monitor_gen_batch, 
            obs=monitor_obs, 
            infos=infos,
            single_preprocessor=self.build_single_monitor_sample,
        )

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        batch_input = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        batch_input.meta_info = monitor_gen_batch.meta_info
        if not self.config.monitor_rollout_ref.enable_train_monitor:
            batch_input.meta_info['validate'] = True  # set validate mode for monitor rollout when not training monitor

        # pad to be divisible by dp_size
        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, monitor_wg.world_size)
        batch_output_padded = monitor_wg.generate_sequences(batch_input_padded)
        # unpad
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

        batch.non_tensor_batch['uid'] = uid_batch
        batch.non_tensor_batch['traj_uid'] = traj_uid

        batch = batch.union(batch_output)

        # print for debugging
        # monitor_output_texts = self.monitor_tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
        # for sample_idx in range(min(4, len(batch.non_tensor_batch['raw_prompt']))):
        #     preview_prompt = self.monitor_tokenizer.apply_chat_template(
        #         batch.non_tensor_batch['raw_prompt'][sample_idx],
        #         add_generation_prompt=True,
        #         tokenize=False
        #     )
        #     preview_action = monitor_output_texts[sample_idx]
        #     print("=" * 80)
        #     print(f"[monitor_rollout] sample {sample_idx} raw_prompt:\n{preview_prompt}")
        #     print("~~~~~~~~~")
        #     print(f"[monitor_rollout] sample {sample_idx} text action: {preview_action}")
        #     print("=" * 80)
        
        # Compute trust penalties using judge model if enabled
        if self.config.judge_model.enable:
            judge_obs = {
                'task_type': infos[0]['task_type'],  # NOTE: assume all in the batch are from the same task_type
                'user_inputs': monitor_gen_batch.non_tensor_batch['user_inputs'],
                'evidence': monitor_gen_batch.non_tensor_batch['system_infos'],
                'agent_trajectory': monitor_gen_batch.non_tensor_batch['agent_trajectory'],
            }
            (
                trust_penalties,
                monitor_action_types,
                monitor_anchor_valid,
                monitor_link_valid,
                correct_no_issue,
                judge_score_tokens,
                judge_stats,
            ) = self._compute_judge_scores(
                monitor_batch=batch,
                obs=judge_obs,
                judge_wg=judge_wg,
                global_step=global_step,
            )
        else:
            raise RuntimeError("Judge model is not enabled, cannot compute trust_penalties. Please set `judge_model.enable` as True when using monitor rollout")

        n_total = len(monitor_action_types)
        n_invalid = int(np.sum(monitor_action_types == "invalid"))
        print(f"  Monitor action parse: {n_total - n_invalid}/{n_total} valid "
              f"({100.0 * (n_total - n_invalid) / max(n_total, 1):.1f}%)")
        print(f"  Computed trust_penalties: {trust_penalties}")
        batch.non_tensor_batch['trust_penalties'] = trust_penalties
        batch.non_tensor_batch['monitor_action_type'] = monitor_action_types
        batch.non_tensor_batch['correct_no_issue'] = correct_no_issue
        batch.non_tensor_batch['judge_score_token'] = judge_score_tokens
        batch.non_tensor_batch['monitor_anchor_valid'] = monitor_anchor_valid
        batch.non_tensor_batch['monitor_link_valid'] = monitor_link_valid
        if judge_stats.get("total_count", 0) > 0:
            error_ratio = float(judge_stats["parse_error_count"]) / float(judge_stats["total_count"])
            print(f"  Judge API parsing error count: {judge_stats['parse_error_count']}/{judge_stats['total_count']} ({100.0 * error_ratio:.1f}%)")
            batch.non_tensor_batch['judge_parse_error_ratio'] = np.full(n_total, error_ratio, dtype=np.float32)

        return batch
    
    def _judge_constrained_score(
        self,
        judge_requests: List[JudgeRequests],
        score_profile_name: str,
        judge_wg,
    ) -> JudgeBatchResult:
        if judge_wg is None:
            raise RuntimeError("judge_wg is required when judge_model.backend='constrained_logits'.")
        if self.judge_tokenizer is None:
            raise RuntimeError("judge_tokenizer is required when judge_model.backend='constrained_logits'.")

        processed_judge_samples = []
        for request in judge_requests:
            processed = self._process_chat_to_model_inputs(
                chat=request.prompt,
                obs_image=request.image,
                tokenizer=self.judge_tokenizer,
                processor=self.judge_processor,
                max_prompt_length=self.config.judge_model.max_prompt_length,
                truncation=self.config.judge_model.truncation,
            )
            processed_judge_samples.append(processed)

        judge_batch = DataProto.from_single_dict(
            data=collate_fn(processed_judge_samples),
        )
        judge_batch.meta_info["score_profile_name"] = score_profile_name

        judge_input_padded, pad_size = pad_dataproto_to_divisor(judge_batch, judge_wg.world_size)
        judge_output_padded = judge_wg.compute_constrained_scores(judge_input_padded)
        judge_output = unpad_dataproto(judge_output_padded, pad_size=pad_size)

        score_tensor = judge_output.batch["constrained_scores"]
        flat_scores = score_tensor.detach().cpu().numpy() if hasattr(score_tensor, "detach") else np.asarray(score_tensor)
        flat_probs = judge_output.batch["constrained_token_probs"]
        probs_array = flat_probs.detach().cpu().numpy() if hasattr(flat_probs, "detach") else np.asarray(flat_probs)
        profile = resolve_score_profile(self._judge_score_profiles(), score_profile_name)
        valid_tokens = list(profile.valid_tokens)
        flat_tokens = [str(valid_tokens[int(np.argmax(row))]) for row in probs_array]
        correct_no_issue = np.full(len(judge_requests), -1.0, dtype=np.float32)
        for idx, request in enumerate(judge_requests):
            if request.action_type == "no_issue":
                correct_no_issue[idx] = correct_no_issue_from_probs(
                    valid_tokens=valid_tokens,
                    token_probs=probs_array[idx],
                )

        self._debug_print_judge_samples(
            processed_judge_samples=processed_judge_samples,
            judge_requests=judge_requests,
            score_profile_name=score_profile_name,
            constrained_scores=flat_scores,
            constrained_token_probs=flat_probs,
        )

        return JudgeBatchResult(
            scores=np.asarray(flat_scores, dtype=np.float32),
            score_tokens=flat_tokens,
            errors=[None] * len(judge_requests),
            correct_no_issue=correct_no_issue,
        )

    def _cot_judge_score(
        self,
        judge_requests: List[JudgeRequests],
        score_profile_name: str,
    ) -> JudgeBatchResult:
        if self._cot_judge_scorer is None:
            from agent_system.judge import ApiCotJudgeScorer

            self._cot_judge_scorer = ApiCotJudgeScorer(self.config.judge_model)

        if judge_requests:
            print("\n" + "=" * 120)
            print("[CoT Judge Input Debug] Showing first API judge input sample before request")
            print("=" * 120)
            for message in judge_requests[0].prompt:
                print(f"[{message.get('role', '')}]")
                print(message.get("content", ""))
                print("-" * 120)
            print("=" * 120)

        cot_result = self._cot_judge_scorer.score_batch(
            [request.prompt for request in judge_requests],
            score_profile_name=score_profile_name,
        )
        self._debug_print_cot_judge_samples(
            judge_requests=judge_requests,
            score_profile_name=score_profile_name,
            cot_result=cot_result,
        )
        correct_no_issue = np.full(len(judge_requests), -1.0, dtype=np.float32)
        for idx, request in enumerate(judge_requests):
            if request.action_type == "no_issue" and cot_result.errors[idx] is None:
                correct_no_issue[idx] = correct_no_issue_from_token(cot_result.parsed_tokens[idx])
        return JudgeBatchResult(
            scores=np.asarray(cot_result.scores, dtype=np.float32),
            score_tokens=cot_result.parsed_tokens,
            errors=list(cot_result.errors),
            correct_no_issue=correct_no_issue,
        )
    
    def _compute_judge_scores(
        self,
        monitor_batch: DataProto,
        obs: Dict,
        judge_wg,
        global_step: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
        """Compute monitor rewards from anchored monitor actions and judge labels."""
        from agent_system.environments.prompts.judge_prompt import build_judge_prompt

        batch_size = len(monitor_batch.batch)
        task_types = [obs['task_type']] * batch_size  # NOTE: assume all in the batch are from the same task_type
        user_inputs = obs['user_inputs']
        evidences = obs['evidence']
        agent_trajs = obs['agent_trajectory']
        assert len(user_inputs) == batch_size, "Mismatch in user_inputs and monitor batch size"
        assert len(evidences) == batch_size, "Mismatch in evidences and monitor batch size"
        assert len(agent_trajs) == batch_size, "Mismatch in agent_trajectories and monitor batch size"
        grm_dir = self.config.trainer.get("grm_judge_data_dir", None)
        should_dump_grm_data = self._should_dump_grm_judge_data(global_step)

        monitor_output_texts = self.monitor_tokenizer.batch_decode(
            monitor_batch.batch['responses'], skip_special_tokens=True
        )

        monitor_rewards = np.zeros(batch_size, dtype=np.float32)
        monitor_action_types = np.full(batch_size, "invalid", dtype=object)
        correct_no_issue = np.full(batch_size, -1.0, dtype=np.float32)
        judge_score_tokens = np.full(batch_size, "", dtype=object)
        anchor_valid = np.full(batch_size, -1.0, dtype=np.float32)
        link_valid = np.full(batch_size, -1.0, dtype=np.float32)

        judge_requests: List[JudgeRequests] = []
        judge_images = obs.get('judge_image', None)
        judge_backend = self.config.judge_model.get("backend", "constrained_logits")
        issue_template_name = (
            "strict5_cot"
            if judge_backend == "api_cot"
            else self.config.judge_model.get("template_name", "strict5")
        )
        no_issue_template_name = (
            "no_issue_verification_cot"
            if judge_backend == "api_cot"
            else "no_issue_verification"
        )

        for item, (monitor_out, user_input, evidence, agent_trajectory, task_type) in enumerate(zip(
            monitor_output_texts, user_inputs, evidences, agent_trajs, task_types
        )):
            parsed = parse_monitor_action(monitor_out)
            monitor_action_types[item] = parsed.action_type

            if parsed.action_type == "invalid":
                monitor_rewards[item] = -2.0
                print(f"[MONITOR ACTION] Sample {item}: invalid action ({parsed.invalid_reason}); "
                      f"monitor reward forced to -2.0. Output snippet: {monitor_out!r}")
                continue

            if parsed.action_type == "no_issue":
                if not self.config.monitor_rollout_ref.enable_train_monitor:
                    monitor_rewards[item] = 0.0
                    continue
                judge_chat = build_judge_prompt(
                    task_type=task_type,
                    user_input=user_input,
                    evidence=evidence,
                    agent_response=agent_trajectory,
                    critique="<no_issue/>",
                    template_name=no_issue_template_name,
                )
                score_profile_name = NO_ISSUE_ACTION_SCORE_PROFILE
            else:
                if self.config.monitor_rollout_ref.enable_train_monitor:
                    anchor_validation = validate_issue_anchors(
                        parsed=parsed,
                        behavior_under_review=agent_trajectory,
                        background=evidence,
                    )
                    anchor_valid[item] = 1.0 if anchor_validation.is_valid else 0.0
                    if not anchor_validation.is_valid:
                        monitor_rewards[item] = -2.0
                        print(f"[MONITOR ACTION] Sample {item}: invalid issue anchors "
                              f"({anchor_validation.invalid_reason}); monitor reward forced to -2.0. "
                              f"Output snippet: {monitor_out!r}")
                        continue
                    if self.config.monitor_rollout_ref.critique_link_check:
                        link_validation = validate_behavior_evidence_link(
                            parsed,
                            behavior_under_review=agent_trajectory,
                            background=evidence,
                        )
                        link_valid[item] = 1.0 if link_validation.is_valid else 0.0
                        if not link_validation.is_valid:
                            monitor_rewards[item] = 0.0
                            print(f"[MONITOR ACTION] Sample {item}: unlinked behavior/evidence anchors "
                                f"({link_validation.invalid_reason}); monitor reward forced to 0.0. "
                                f"Output snippet: {monitor_out!r}")
                            continue

                judge_chat = build_judge_prompt(
                    task_type=task_type,
                    user_input=user_input,
                    evidence=evidence,
                    agent_response=agent_trajectory,
                    critique=parsed.serialized_issue,
                    template_name=issue_template_name,
                )
                score_profile_name = ISSUE_ACTION_SCORE_PROFILE

            judge_requests.append(JudgeRequests(
                sample_idx=item,
                action_type=parsed.action_type,
                score_profile_name=score_profile_name,
                prompt=judge_chat,
                image=judge_images[item] if judge_images is not None else None,
            ))

        # --- Judge inference (only for format-valid samples) ---
        judge_stats = {"parse_error_count": 0, "total_count": 0}
        if len(judge_requests) > 0:
            for score_profile_name, profile_requests in self._group_judge_requests_by_profile(judge_requests).items():
                if judge_backend == "constrained_logits":
                    judge_result = self._judge_constrained_score(
                        judge_requests=profile_requests,
                        score_profile_name=score_profile_name,
                        judge_wg=judge_wg,
                    )
                elif judge_backend == "api_cot":
                    # not considering multi-modal judge input for api_cot judge for now
                    judge_result = self._cot_judge_score(
                        judge_requests=profile_requests,
                        score_profile_name=score_profile_name,
                    )
                else:
                    raise ValueError(f"Unsupported judge_model.backend={judge_backend!r}")

                judge_stats["parse_error_count"] += sum(1 for e in judge_result.errors if e is not None)
                judge_stats["total_count"] += len(judge_result.score_tokens)

                assert len(judge_result.score_tokens) == len(profile_requests), (
                    f"Mismatch: {len(judge_result.score_tokens)} judge tokens vs {len(profile_requests)} judge requests"
                )
                assert len(judge_result.scores) == len(profile_requests), (
                    f"Mismatch: {len(judge_result.scores)} judge scores vs {len(profile_requests)} judge requests"
                )
                assert len(judge_result.errors) == len(profile_requests), (
                    f"Mismatch: {len(judge_result.errors)} judge errors vs {len(profile_requests)} judge requests"
                )
                assert len(judge_result.correct_no_issue) == len(profile_requests), (
                    f"Mismatch: {len(judge_result.correct_no_issue)} no-issue correctness rows vs {len(profile_requests)} judge requests"
                )
                grm_samples = []
                profile = resolve_score_profile(self._judge_score_profiles(), score_profile_name)
                valid_tokens = list(profile.valid_tokens)
                for judge_idx, request in enumerate(profile_requests):
                    token = judge_result.score_tokens[judge_idx]
                    judge_score_tokens[request.sample_idx] = "" if token is None else str(token)
                    monitor_rewards[request.sample_idx] = float(judge_result.scores[judge_idx])
                    if request.action_type == "no_issue":
                        correct_no_issue[request.sample_idx] = judge_result.correct_no_issue[judge_idx]
                    if should_dump_grm_data:
                        grm_samples.append(GrmJudgeSample(
                            prompt=request.prompt,
                            valid_tokens=valid_tokens,
                            judge_pred_token=None if token is None else str(token),
                            label=None,
                        ))
                if grm_samples:
                    dumped = append_judge_samples(grm_dir, grm_samples, step=global_step)
                    print(f"[GRM DATA] Dumped {dumped} judge samples to {grm_dir} at step {global_step}")
        else:
            print("[MONITOR ACTION] All monitor outputs were invalid; no judge inference performed.")

        return (
            monitor_rewards,
            monitor_action_types,
            anchor_valid,
            link_valid,
            correct_no_issue,
            judge_score_tokens,
            judge_stats,
        )

    def verdict_monitor_score(
        self,
        actor_batch: DataProto,
        verdict_monitor_wg,
        infos: List[Dict],
    ) -> np.ndarray:
        assert verdict_monitor_wg is not None, "verdict monitor worker group should not be None for verdict monitor rollout"

        task_type = infos[0]['task_type']
        verdict_backgrounds = np.asarray(
            [
                build_verdict_monitor_background(task_type, background)
                for background in actor_batch.non_tensor_batch['monitor_background']
            ],
            dtype=object,
        )
        verdict_obs = {
            'task_type': task_type,
            'monitor_background': verdict_backgrounds,
            'agent_trajectory': actor_batch.non_tensor_batch['agent_trajectory'],
            'monitor_image': actor_batch.non_tensor_batch.get('monitor_image', None),
        }
        batch = self.preprocess_batch(
            gen_batch=actor_batch,
            obs=verdict_obs,
            infos=infos,
            single_preprocessor=self.build_single_verdict_monitor_sample,
        )

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")

        batch_input = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        batch_input.meta_info = actor_batch.meta_info
        batch_input.meta_info['validate'] = True

        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, verdict_monitor_wg.world_size)
        verdict_output_padded = verdict_monitor_wg.compute_constrained_scores(batch_input_padded)
        verdict_output = unpad_dataproto(verdict_output_padded, pad_size=pad_size)

        # non debugging logic: directly return penalties converted from constrained token probs
        # return constrained_probs_to_binary_penalties(
        #     constrained_token_probs=verdict_output.batch["constrained_token_probs"],
        #     valid_tokens=self.config.verdict_monitor.valid_tokens,
        # )

        # debug logic
        constrained_scores = verdict_output.batch["constrained_scores"]
        constrained_token_probs = verdict_output.batch["constrained_token_probs"]
        penalties = constrained_probs_to_binary_penalties(
            constrained_token_probs=constrained_token_probs,
            valid_tokens=self.config.verdict_monitor.valid_tokens,
            threshold=float(self.config.verdict_monitor.get("decision_threshold", 0.5)),
        )

        self._debug_print_verdict_monitor_samples(
            raw_prompt_ids=batch_input.non_tensor_batch["raw_prompt_ids"],
            constrained_scores=constrained_scores,
            constrained_token_probs=constrained_token_probs,
            penalties=penalties,
        )

        return penalties

    # TODO-monitor: Integrate this rollout func with monitor
    def dynamic_multi_turn_loop(
        self,
        gen_batch: DataProto, 
        actor_rollout_wg, 
        monitor_wg,
        verdict_monitor_wg,
        judge_wg,
        envs: EnvironmentManagerBase,
        rollout_n: int,
        monitor_rollout_n: int,
    ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            monitor_wg: Monitor model workers for generating critiques.
            judge_wg: Judge model workers for scoring critiques.
            envs (EnvironmentManagerBase): Environment manager instance.
            rollout_n (int): Number of env rollouts per group.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * rollout_n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * rollout_n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            actor_batch_dict, _ = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                monitor_wg=monitor_wg,
                verdict_monitor_wg=verdict_monitor_wg,
                judge_wg=judge_wg,
                envs=envs,
                rollout_n=rollout_n,
                monitor_rollout_n=monitor_rollout_n,
            )
            batch_list = actor_batch_dict["total_batch_list"]
            episode_rewards = actor_batch_dict["episode_rewards"]
            episode_lengths = actor_batch_dict["episode_lengths"]
            success = actor_batch_dict["success"]
            traj_uid = actor_batch_dict["traj_uid"]
            tool_callings = actor_batch_dict["tool_callings"]
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(
                batch_list=batch_list, 
                episode_rewards=episode_rewards, 
                episode_lengths=episode_lengths, 
                success=success, 
                traj_uid=traj_uid, 
                tool_callings=tool_callings, 
                config=self.config,
                last_try=(try_count == max_try_count),
            )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings
    
    def multi_turn_loop(
        self,
        gen_batch: DataProto, 
        actor_rollout_wg, 
        monitor_wg,
        verdict_monitor_wg,
        judge_wg,
        envs: EnvironmentManagerBase,
        is_train: bool = True,
        global_step: Optional[int] = None,
    ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            monitor_wg: Monitor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).
            judge_wg: Judge model workers for critique scoring.

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        rollout_n = self.config.env.rollout.n if is_train else self.config.env.rollout.val_n
        monitor_rollout_n = self.config.monitor_rollout_ref.rollout.n if (self.config.monitor_rollout_ref.enable_train_monitor and is_train) else 1
        
        gen_batch = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            if self.config.self_monitor.enable or self.config.verdict_monitor.enable or self.config.monitor_rollout_ref.enable:
                raise NotImplementedError("filter_groups is not supported with self_monitor, verdict_monitor, or monitor_rollout_ref.")
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                monitor_wg=monitor_wg,
                verdict_monitor_wg=verdict_monitor_wg,
                judge_wg=judge_wg,
                envs=envs,
                rollout_n=rollout_n,
                monitor_rollout_n=monitor_rollout_n,
                global_step=global_step,
            )
            actor_batch_dict = {
                "total_batch_list": total_batch_list,
                "episode_rewards": total_episode_rewards,
                "episode_lengths": total_episode_lengths,
                "success": total_success,
                "traj_uid": total_traj_uid,
                "tool_callings": total_tool_callings,
                "self_monitor_trust_penalties": None,
                "verdict_monitor_trust_penalties": None,
            }
            monitor_batch_output = None
        else:
            # Vanilla Sampling   
            actor_batch_dict, monitor_batch_output = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                monitor_wg=monitor_wg,
                verdict_monitor_wg=verdict_monitor_wg,
                judge_wg=judge_wg,
                envs=envs,
                rollout_n=rollout_n,
                monitor_rollout_n=monitor_rollout_n,
                global_step=global_step
            )
        assert len(actor_batch_dict['total_batch_list']) == len(actor_batch_dict['episode_rewards'])
        assert len(actor_batch_dict['total_batch_list']) == len(actor_batch_dict['episode_lengths'])
        assert len(actor_batch_dict['total_batch_list']) == len(actor_batch_dict['traj_uid'])
        assert len(actor_batch_dict['total_batch_list']) == len(actor_batch_dict['tool_callings'])

        # construct trust_penalties for actor batch, will be used for actor model training
        actor_trust_penalties = None
        if self.config.self_monitor.enable:
            actor_trust_penalties = actor_batch_dict['self_monitor_trust_penalties']
        elif self.config.verdict_monitor.enable:
            actor_trust_penalties = actor_batch_dict['verdict_monitor_trust_penalties']
        elif monitor_wg is not None and monitor_batch_output is not None:
            # Clip to [0, 1] as the actor's cost.
            # The monitor's batch retains the original unclipped [-1, 1] values for its own PPO update.            
            actor_trust_penalties = np.clip(monitor_batch_output.non_tensor_batch['trust_penalties'], 0.0, 1.0)
            actor_batch_size = len(actor_batch_dict['total_batch_list'])
            if len(actor_trust_penalties) != actor_batch_size:
                expected_size = actor_batch_size * monitor_rollout_n
                assert len(actor_trust_penalties) == expected_size, (
                    f"trust_penalties size mismatch: got {len(actor_trust_penalties)}, "
                    f"expected {actor_batch_size} (actor batch) or {expected_size} (actor batch * repeat_n={monitor_rollout_n})"
                )
                # TODO: assuming interleaved grouping for now, can add non-interleaved grouping if needed
                actor_trust_penalties = actor_trust_penalties.reshape(actor_batch_size, monitor_rollout_n).mean(axis=1)
            
            actor_cost_threshold = float(self.config.monitor_rollout_ref.get("actor_cost_threshold", 0.5))
            actor_trust_penalties = np.where(
                actor_trust_penalties > actor_cost_threshold,
                actor_trust_penalties,
                0.0,
            ).astype(np.float32)

        # Create trajectory data for actor model
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=actor_batch_dict['total_batch_list'],
            episode_rewards=actor_batch_dict['episode_rewards'],
            episode_lengths=actor_batch_dict['episode_lengths'],
            success=actor_batch_dict['success'],
            traj_uid=actor_batch_dict['traj_uid'],
            tool_callings=actor_batch_dict['tool_callings'],
            trust_penalties=actor_trust_penalties,
        )

        if self.config.monitor_rollout_ref.enable:
            return {
                "actor": gen_batch_output,
                "monitor": monitor_batch_output
            }
        else:
            return {"actor": gen_batch_output}

    # Final actor dataproto
    # tensor batch: {
    #     'prompts': 'Tensor(shape=(4, 512), dtype=torch.int64)', 
    #     'rollout_log_probs': 'Tensor(shape=(4, 512), dtype=torch.float32)', 
    #     'attention_mask': 'Tensor(shape=(4, 1024), dtype=torch.int64)', 
    #     'input_ids': 'Tensor(shape=(4, 1024), dtype=torch.int64)', 
    #     'position_ids': 'Tensor(shape=(4, 1024), dtype=torch.int64)', 
    #     'responses': 'Tensor(shape=(4, 512), dtype=torch.int64)'}
    # non_tensor batch: {
    #     'data_source': 'ndarray(shape=(4,), dtype=object)', 
    #     'uid': 'ndarray(shape=(4,), dtype=object)', 
    #     'traj_uid': 'ndarray(shape=(4,), dtype=object)', 
    #     'raw_prompt': 'ndarray(shape=(4, 2), dtype=object)', 
    #     'tools_kwargs': 'ndarray(shape=(4,), dtype=object)', 
    #     'is_action_valid': 'ndarray(shape=(4,), dtype=object)', 
    #     'rewards': 'ndarray(shape=(4,), dtype=object)', 
    #     'active_masks': 'ndarray(shape=(4,), dtype=object)', 
    #     'episode_rewards': 'ndarray(shape=(4,), dtype=object)', 
    #     'episode_lengths': 'ndarray(shape=(4,), dtype=object)', 
    #     'tool_callings': 'ndarray(shape=(4,), dtype=object)', 
    #     'success_rate': 'ndarray(shape=(4,), dtype=object)'
    # }
    # meta_info: []

    # Final monitor dataproto (assuming monitor rollout n is 2)
    # tensor batch: 
    # {'prompts': , 'rollout_log_probs': , 'attention_mask': , 'input_ids': , 'position_ids': , 'responses': }
    # non_tensor batch: {
    #     'data_source': 'ndarray(shape=(8,), dtype=object)', 
    #     'uid': 'ndarray(shape=(8,), dtype=object)', 
    #     'traj_uid': 'ndarray(shape=(8,), dtype=object)', 
    #     'raw_prompt': 'ndarray(shape=(8, 2), dtype=object)', 
    #     ?'rewards': 'ndarray(shape=(8,), dtype=object)', 
    #     'episode_rewards': 'ndarray(shape=(8,), dtype=object)', 
    #     'trust_penalty': 'ndarray(shape=(8,), dtype=object)'
    # }
    # meta_info: []


def _summarize_value(value):
    """Return a compact description for debugging."""
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, dict):
        return f"dict(keys={list(value.keys())})"
    if isinstance(value, DataProto):
        return "DataProto"
    return type(value).__name__

def _log_dataproto(tag: str, proto: DataProto):
    if proto is None:
        print(f"[vanilla_multi_turn_loop] {tag}: None")
        return
    tensor_summary = {k: _summarize_value(v) for k, v in proto.batch.items()}
    non_tensor_summary = {k: _summarize_value(v) for k, v in proto.non_tensor_batch.items()}
    meta_info = getattr(proto, "meta_info", None)
    if hasattr(meta_info, "keys"):
        meta_summary = list(meta_info.keys())
    else:
        meta_summary = type(meta_info).__name__
    print(
        f"[vanilla_multi_turn_loop] {tag}\n"
        f"  tensor batch: {tensor_summary}\n"
        f"  non_tensor batch: {non_tensor_summary}\n"
        f"  meta_info: {meta_summary}"
    )

def _log_list_structure(tag: str, data):
    """Print a lightweight overview of list-based debug structures."""
    if not isinstance(data, list):
        print(f"[vanilla_multi_turn_loop] {tag}: {type(data).__name__}")
        return
    length = len(data)
    print(f"[vanilla_multi_turn_loop] {tag}: list(len={length})")
    if length == 0:
        return
    first = data[0]
    if isinstance(first, list):
        print(f"[vanilla_multi_turn_loop] {tag}[0]: list(len={len(first)})")
        if first:
            inner = first[0]
            if isinstance(inner, dict):
                inner_summary = {k: _summarize_value(v) for k, v in inner.items()}
                print(f"[vanilla_multi_turn_loop] {tag}[0][0] summary: {inner_summary}")
            else:
                print(f"[vanilla_multi_turn_loop] {tag}[0][0]: {_summarize_value(inner)}")
    elif isinstance(first, dict):
        dict_summary = {k: _summarize_value(v) for k, v in first.items()}
        print(f"[vanilla_multi_turn_loop] {tag}[0] keys: {list(first.keys())}")
        print(f"[vanilla_multi_turn_loop] {tag}[0] summary: {dict_summary}")
    else:
        print(f"[vanilla_multi_turn_loop] {tag}[0]: {_summarize_value(first)}")
    
def _print_tensor_info(name: str, tensor: torch.Tensor, show_values: bool = False, max_show: int = 20):
    print(f"  [{name}] shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}")
    if show_values:
        flat = tensor.flatten()
        values = flat[:max_show].tolist()
        print(f"    values (first {len(values)}): {values}")
    if tensor.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        print(f"    stats: min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")



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
import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments.prompts.monitor_prompt import MONITOR_PROMPT
from agent_system.environments.prompts import DEFAULT_SYSTEM_PROMPT
from agent_system.environments.prompts.verdict_monitor_prompt import build_verdict_monitor_prompt
from agent_system.environments import EnvironmentManagerBase
from agent_system.utils.metric_contract import EPISODE_METRIC_PREFIX
from agent_system.self_monitor import parse_self_monitor_batch
from agent_system.verdict_monitor import constrained_probs_to_binary_penalties
from typing import List, Dict, Callable, Tuple, Optional
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

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
        if config.monitor_rollout_ref.enable:
            assert monitor_tokenizer is not None, "monitor tokenizer should be provided when monitor is enabled"
            self.monitor_tokenizer = monitor_tokenizer
            self.monitor_processor = monitor_processor
        if config.judge_model.enable:
            assert judge_tokenizer is not None, "judge tokenizer should be provided when judge is enabled"
            self.judge_tokenizer = judge_tokenizer
            self.judge_processor = judge_processor
        if config.verdict_monitor.enable:
            assert verdict_monitor_tokenizer is not None, "verdict monitor tokenizer should be provided when verdict_monitor is enabled"
            self.verdict_monitor_tokenizer = verdict_monitor_tokenizer
            self.verdict_monitor_processor = verdict_monitor_processor
    
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
        judge_score_slots: List[Tuple[int, int]],
        constrained_scores,
        constrained_token_probs,
    ) -> None:
        debug_print_samples = 2
        if debug_print_samples <= 0 or len(processed_judge_samples) == 0:
            return

        num_samples = min(debug_print_samples, len(processed_judge_samples), len(judge_score_slots))
        probs_array = constrained_token_probs.numpy() if hasattr(constrained_token_probs, "numpy") else np.asarray(constrained_token_probs)
        valid_tokens = list(self.config.judge_model.valid_tokens)
        token_weights = list(self.config.judge_model.token_weights)
        constrained_top_k = self.config.judge_model.get("constrained_top_k", -1)

        print("\n" + "=" * 120)
        print(
            f"[Judge Debug] Showing {num_samples}/{len(processed_judge_samples)} samples | "
            f"valid_tokens={valid_tokens} | constrained_top_k={constrained_top_k}"
        )
        print("=" * 120)

        for idx in range(num_samples):
            sample_idx, critique_idx = judge_score_slots[idx]
            prompt_ids = processed_judge_samples[idx]["raw_prompt_ids"]
            if hasattr(prompt_ids, "tolist"):
                prompt_ids = prompt_ids.tolist()
            prompt_text = self.judge_tokenizer.decode(prompt_ids, skip_special_tokens=False)

            prob_row = probs_array[idx].tolist()
            best_idx = int(np.argmax(prob_row))
            prob_summary = ", ".join(
                f"{token}={prob:.4f}" for token, prob in zip(valid_tokens, prob_row)
            )

            print(f"[Judge Debug] Queued sample {idx + 1}/{num_samples} | source_sample={sample_idx} | critique_idx={critique_idx}")
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
        is_multi_modal = obs_image is not None
        
        # Apply chat template
        prompt_with_chat_template = tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Initialize return dict
        row_dict = {}
        image_grid_thw = None
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace(
                    '<|placeholder|>',
                    processor.image_token
                )
        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=tokenizer,
            max_length=max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation=truncation
        )

        if is_multi_modal:
            position_ids = [
                get_rope_index(
                    processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
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

        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
        })
        
        return row_dict

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

        user_content = MONITOR_PROMPT.format_user_message(
            task_type=task_type,
            background=monitor_background,
            behavior_under_review=agent_trajectory,
        )
        
        chat = [
            {"content": MONITOR_PROMPT.system_prompt, "role": "system"},
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
            template_name=self.config.verdict_monitor.template_name,
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

        # Trajectory collection loop
        rollout_max_steps = envs.get_rollout_max_steps()
        for _step in range(rollout_max_steps):
            active_masks = np.logical_not(is_done)

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

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
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

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            self_monitor_batch = None
            if self.config.self_monitor.enable:
                self_monitor_batch = parse_self_monitor_batch(text_actions)
                batch.non_tensor_batch.update(self_monitor_batch)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)
            
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
                final_action_valid = np.logical_and(env_action_valid, self_monitor_batch['self_monitor_is_valid'])
                self_monitor_trust_penalties[active_masks] = np.logical_or(
                    self_monitor_trust_penalties[active_masks] > 0.0,
                    self_monitor_batch['self_monitor_is_unsafe'][active_masks],
                ).astype(np.float32)
            else:
                final_action_valid = env_action_valid

            batch.non_tensor_batch['is_action_valid'] = final_action_valid

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)

            # log for retroactive analysis and judge_model input if judge enabled
            batch.non_tensor_batch['user_inputs'] = np.array([info['user_input'] for info in infos], dtype=object)
            batch.non_tensor_batch['system_infos'] = np.array([info['evidence'] for info in infos], dtype=object)
            
            if self.config.monitor_rollout_ref.enable or self.config.verdict_monitor.enable:
                batch.non_tensor_batch['monitor_background'] = np.array(next_obs['monitor_background'])
                batch.non_tensor_batch['agent_trajectory'] = np.array(next_obs['agent_trajectory'])
                if next_obs.get('monitor_image', None) is not None:
                    batch.non_tensor_batch['monitor_image'] = np.array(next_obs['monitor_image'])
                if self.config.judge_model.enable:
                    batch.non_tensor_batch['agent_response'] = np.array([info['agent_response'] for info in infos], dtype=object)
            
            batch.check_consistency()

            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break

        # NOTE: Not 100% sure, but it seems that the environments do not short-circuit for already-done envs. 
        # They still:
        # 1. Process the action
        # 2. Return observations (often unchanged or invalid)
        # 3. Return rewards (typically 0)
        # 4. Return done=True again
        # So filtering out data with active_masks=False is necessary
        # TODO: What will the obs be when loop finished and some envs are already done in earlier steps?
        # TODO: This agent loop is to be optimized to asynchronously process envs with varied episode lengths

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
        if self.config.judge_model.enable and judge_wg is not None:
            assert self.judge_tokenizer is not None, "judge tokenizer should not be None for judge scoring"
            judge_obs = {
                'task_type': infos[0]['task_type'],  # NOTE: assume all in the batch are from the same task_type
                'user_inputs': monitor_gen_batch.non_tensor_batch['user_inputs'],
                'evidence': monitor_gen_batch.non_tensor_batch['system_infos'],
                'agent_response': monitor_gen_batch.non_tensor_batch['agent_response'],
            }
            trust_penalties, monitor_format_correct = self._compute_judge_scores(
                monitor_batch=batch,
                obs=judge_obs,
                judge_wg=judge_wg,
            )
        elif self.config.judge_model.enable and judge_wg is None:
            raise RuntimeError("Judge worker group is None but judge_model.enable is True, cannot compute judge scores for trust penalties")
        else:
            raise RuntimeError("Judge model is not enabled, cannot compute trust_penalties. Please set `judge_model.enable` as True when using monitor rollout")

        n_correct = int(monitor_format_correct.sum())
        n_total = len(monitor_format_correct)
        print(f"  Monitor format check: {n_correct}/{n_total} correct "
              f"({100.0 * n_correct / max(n_total, 1):.1f}%)")
        print(f"  Computed trust_penalties: {trust_penalties}")
        batch.non_tensor_batch['trust_penalties'] = trust_penalties
        batch.non_tensor_batch['is_format_correct'] = monitor_format_correct

        return batch

    def verdict_monitor_score(
        self,
        actor_batch: DataProto,
        verdict_monitor_wg,
        infos: List[Dict],
    ) -> np.ndarray:
        assert verdict_monitor_wg is not None, "verdict monitor worker group should not be None for verdict monitor rollout"

        verdict_obs = {
            'task_type': infos[0]['task_type'],
            'monitor_background': actor_batch.non_tensor_batch['monitor_background'],
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
        )

        self._debug_print_verdict_monitor_samples(
            raw_prompt_ids=batch_input.non_tensor_batch["raw_prompt_ids"],
            constrained_scores=constrained_scores,
            constrained_token_probs=constrained_token_probs,
            penalties=penalties,
        )

        return penalties
    
    def _compute_judge_scores(
        self,
        monitor_batch: DataProto,
        obs: Dict,
        judge_wg,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute trust penalties using the judge model, with monitor format gating.

        If a monitor output does not contain valid <tag>...</tag> tags,
        its judge score is set to 0 without calling the judge model for that sample.

        This method:
        1. Extracts <critique> tags from each monitor output (same regex as extract_critiques,
           but without the fallback that treats the entire output as a single critique)
        2. Samples with no valid tags are marked format-incorrect and get score 0
        3. Builds judge prompts for each valid critique
        4. Batches all valid critiques and runs judge inference
        5. Aggregates per-critique scores back to per-sample via mean

        Args:
            monitor_batch: DataProto containing monitor outputs
            obs: Observation dict containing judge evidence and agent behavior
            judge_wg: Judge worker group for inference

        Returns:
            Tuple of:
                - np.ndarray of trust penalties, shape (batch_size,)
                - np.ndarray of format correctness flags (bool), shape (batch_size,)
        """
        assert judge_wg is not None, "judge worker group should not be None for judge scoring"
        from agent_system.environments.prompts.judge_prompt import (
            extract_critiques,
            build_judge_prompt,
            is_no_issue_sentinel,
        )

        batch_size = len(monitor_batch.batch)

        # Extract evidence and task types from obs/infos
        task_types = [obs['task_type']] * batch_size  # NOTE: assume all in the batch are from the same task_type
        user_inputs = obs['user_inputs']
        evidences = obs['evidence']
        # TODO: Consider should agent responses exclude reasoning/tool-calling steps?
        agent_resps = obs['agent_response']
        assert len(user_inputs) == batch_size, "Mismatch in user_inputs and monitor batch size"
        assert len(evidences) == batch_size, "Mismatch in evidences and monitor batch size"
        assert len(agent_resps) == batch_size, "Mismatch in agent_resps and monitor batch size"

        monitor_output_texts = self.monitor_tokenizer.batch_decode(
            monitor_batch.batch['responses'], skip_special_tokens=True
        )

        per_sample_scores = np.zeros(batch_size, dtype=np.float32)
        format_correct = np.zeros(batch_size, dtype=bool)

        # Per-sample critique score lists. Sentinel entries are pre-filled with 0.0;
        # non-sentinel entries start as None and are filled after judge inference.
        # All critiques (sentinel + non-sentinel) are included in the per-sample mean.
        sample_critique_scores: Dict[int, List] = {}
        # Maps each queued judge call → (sample_idx, position in sample_critique_scores[sample_idx])
        judge_score_slots: List[Tuple[int, int]] = []
        all_judge_prompts = []
        all_judge_imgs = []
        # TODO: The multi-modal processing has not been tested yet
        judge_images = obs.get('judge_image', None)

        for item, (monitor_out, user_input, evidence, resp, task_type) in enumerate(zip(
            monitor_output_texts, user_inputs, evidences, agent_resps, task_types
        )):
            # Extract <critique> tags — empty list means bad format
            critiques = extract_critiques(monitor_out)
            count = len(critiques)
            if count <= 0:
                per_sample_scores[item] = -1.0
                print(f"[FORMAT CHECK] Sample {item}: monitor output has invalid format, "
                      f"judge score forced to -1.0. Output snippet: {monitor_out!r}")
                continue  # format_correct[item] stays False

            format_correct[item] = True
            sample_critique_scores[item] = []

            for critique in critiques:
                pos = len(sample_critique_scores[item])
                if is_no_issue_sentinel(critique):
                    # Exact sentinel phrase → 0.0 immediately, skip judge call but keep in mean
                    sample_critique_scores[item].append(0.0)
                else:
                    # Non-sentinel → queue to judge; placeholder filled after inference
                    sample_critique_scores[item].append(None)
                    judge_score_slots.append((item, pos))
                    judge_chat = build_judge_prompt(
                        task_type=task_type,
                        user_input=user_input,
                        evidence=evidence,
                        agent_response=resp,
                        critique=critique,
                        template_name=self.config.judge_model.template_name,
                    )
                    all_judge_prompts.append(judge_chat)
                    all_judge_imgs.append(judge_images[item] if judge_images is not None else None)

        # --- Judge inference (only for non-sentinel critiques from format-valid samples) ---
        if len(all_judge_prompts) > 0:
            # TODO: this asserting logic may be unnecessary, once the code is stable we can remove it.
            assert len(all_judge_prompts) == len(judge_score_slots), "Mismatch in judge prompts and score slots"
            assert len(all_judge_imgs) == len(all_judge_prompts), "Mismatch in judge images and inputs"

            # Prepare DataProto for judge model inputs
            processed_judge_samples = []
            for judge_chat, judge_img in zip(all_judge_prompts, all_judge_imgs):
                processed = self._process_chat_to_model_inputs(
                    chat=judge_chat,
                    obs_image=judge_img,
                    tokenizer=self.judge_tokenizer,
                    processor=self.judge_processor,
                    max_prompt_length=self.config.judge_model.max_prompt_length,
                    truncation=self.config.judge_model.truncation,
                )
                processed_judge_samples.append(processed)

            judge_batch = DataProto.from_single_dict(
                data=collate_fn(processed_judge_samples),
            )

            # Run judge inference
            judge_input_padded, pad_size = pad_dataproto_to_divisor(judge_batch, judge_wg.world_size)
            judge_output_padded = judge_wg.compute_constrained_scores(judge_input_padded)
            judge_output = unpad_dataproto(judge_output_padded, pad_size=pad_size)

            # Fill judge scores back into per-sample score lists
            flat_scores = judge_output.batch["constrained_scores"].numpy()
            flat_probs = judge_output.batch["constrained_token_probs"]

            self._debug_print_judge_samples(
                processed_judge_samples=processed_judge_samples,
                judge_score_slots=judge_score_slots,
                constrained_scores=flat_scores,
                constrained_token_probs=flat_probs,
            )

            assert len(flat_scores) == len(judge_score_slots), (
                f"Mismatch: {len(flat_scores)} judge scores vs {len(judge_score_slots)} score slots"
            )
            for judge_idx, (sample_idx, pos) in enumerate(judge_score_slots):
                assert sample_critique_scores[sample_idx][pos] is None, "Score slot already filled, logic error"
                sample_critique_scores[sample_idx][pos] = float(flat_scores[judge_idx])
        else:
            if len(sample_critique_scores) > 0:
                print("[SENTINEL] All format-correct critiques were no-issue sentinels; no judge inference needed.")
            else:
                print("[FORMAT CHECK] All monitor outputs had invalid format; no judge inference performed.")

        # Aggregate all critiques (sentinel + judge-scored) per sample via mean
        for sample_idx, scores in sample_critique_scores.items():
            per_sample_scores[sample_idx] = float(np.mean(scores))

        return per_sample_scores, format_correct

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
            # Clip to [0, 1] for the actor's Lagrangian cost constraint.
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



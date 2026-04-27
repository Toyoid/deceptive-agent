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

from verl import DataProto
import torch
import numpy as np


class ActorMonitorRewardManager:
    """
    Reward manager for agent training with monitor penalties

    Agent reward = episode_reward - trust_penalty
    Note that the monitor is not trained, only induces trust_penalty to penalize deception of the actor 
    """

    def __init__(self, tokenizer, num_examine, role='actor', normalize_by_length=False) -> None:
        self.role = role
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we add it with env computed reward
        if "rm_scores" in data.batch.keys():
            # Use the same dtype as rm_scores for consistency
            rm_dtype = data.batch["rm_scores"].dtype
            reward_tensor = data.batch["rm_scores"].clone().to(data.batch["responses"].device)
        else:
            rm_dtype = torch.float32
            reward_tensor = torch.zeros_like(data.batch['responses'], dtype=rm_dtype)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            data_source = data_item.non_tensor_batch['data_source']

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            multi_modal_inputs = data_item.non_tensor_batch.get('multi_modal_inputs', None)
            if multi_modal_inputs is not None:
                pixel_values = multi_modal_inputs['pixel_values']
                image_grid_thw = multi_modal_inputs['image_grid_thw']

            if self.role == "actor":
                episode_reward = data_item.non_tensor_batch['episode_rewards']
                episode_length = data_item.non_tensor_batch['episode_lengths']
                trust_penalty = data_item.non_tensor_batch['trust_penalties']
                if self.normalize_by_length:
                    episode_reward = episode_reward / episode_length
                    trust_penalty = trust_penalty / episode_length
                final_score = episode_reward - 2.0 * trust_penalty  # NOTE: there are 3 rewards: RM score, episode_reward (from env), trust_penalty (from monitor)
            elif self.role == "monitor":
                trust_penalty = data_item.non_tensor_batch['trust_penalties']
                if self.normalize_by_length:
                    final_score = trust_penalty / episode_length
                else:
                    final_score = trust_penalty
            else:
                raise ValueError(f"Unknown role: {self.role}")

            reward_tensor[i, valid_response_length - 1] += torch.tensor(final_score, dtype=rm_dtype, device=prompt_ids.device)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1

                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch['responses']
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)

                print(f"[{data_source}][prompt]", prompt_str)
                print(f"[{data_source}][response]", response_str)
                print(f"[{data_source}][final_score]", final_score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {},
            }
        else:
            return reward_tensor

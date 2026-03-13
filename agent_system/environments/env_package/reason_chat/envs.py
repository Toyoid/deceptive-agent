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

from typing import Any, Dict, List, Tuple

import gym
import numpy as np

from agent_system.environments.prompts.monitor_prompt import CHAT_TEMPLATE


class ReasonChatMultiProcessEnv(gym.Env):
    """
    Lightweight vector environment for single-turn "reason → answer" chat tasks.

    The env is stateless beyond the prompt metadata: once an action is submitted,
    the episode terminates (max_steps defaults to 1). Rewards are always zero and
    are expected to be filled by a downstream reward model.
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,  # Number of environment groups, corresponding to train_batch_size
        group_n: int = 1,  # Number of environments per group, corresponding to actor rollout number per question
        is_train: bool = True,
        env_config: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.env_num = env_num
        self.group_n = group_n
        computed_capacity = env_num * group_n  # total_envs = env_num * group_n
        self.capacity = computed_capacity if computed_capacity > 0 else None
        self.is_train = is_train
        self.max_steps = int(getattr(env_config, "max_steps", 1) if env_config else 1)
        if self.max_steps < 1:
            raise ValueError("ReasonChat environment must have max_steps >= 1.")

        self._rng = np.random.default_rng(seed)
        self._episodes: List[Dict[str, Any]] = []

    # ------------------------ gym APIs ------------------------
    def reset(self, kwargs: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        if kwargs is None or len(kwargs) == 0:
            raise ValueError("ReasonChat env requires non-empty env_kwargs.")

        self._episodes.clear()
        infos: List[Dict[str, Any]] = []

        for i, env_dict in enumerate(kwargs):
            system_prompt = env_dict["system_prompt"]
            instruction = env_dict["instruction"]
            question = env_dict["question"]
            # Build history: system prompt (without instruction) + user question
            system_formatted = CHAT_TEMPLATE.format_system(f"{system_prompt}")
            question_formatted = CHAT_TEMPLATE.format_user(question)
            history = system_formatted + question_formatted

            self._episodes.append({
                "task_type": env_dict.get("task_type", "chat"),
                "step": 0, 
                "done": False,
                "evidence": system_formatted,
                "user_input": question,
                "history": history,
                "agent_response": "",
            })

            infos.append({
                "task_type": env_dict.get("task_type", "chat"),
            })
            
        return infos

    def step(self, actions: List[Dict[str, str]]) -> Tuple[List[str], List[float], List[bool], List[Dict[str, Any]]]:
        assert len(self._episodes) > 0, "Environment must be reset before stepping."
        if len(actions) != len(self._episodes):
            raise ValueError(
                f"ReasonChat env expects {len(self._episodes)} actions, but got {len(actions)}."
            )
        
        next_obs: List[str] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []

        for episode, payload in zip(self._episodes, actions):
            # Format assistant response with think/answer tags
            assistant_resp = f"<think>\n{payload['reason']}\n</think>\n<answer>\n{payload['answer']}\n</answer>"
            episode["history"] += CHAT_TEMPLATE.format_assistant(assistant_resp)

            episode["step"] += 1
            done = episode["step"] >= self.max_steps
            episode["done"] = done

            next_obs.append(episode["history"])  # for monitor input

            rewards.append(0.0)  # reward model will fill actual values later
            dones.append(done)

            info = {
                "task_type": episode["task_type"],
                "won": False,
                "user_input": episode["user_input"],  # for judge input
                "evidence": episode["evidence"],  # for judge input
                "agent_response": assistant_resp,  # for judge input
            }
            infos.append(info)

        return next_obs, rewards, dones, infos


# -----------------------------------------------------------------------------
# Factory helper
# -----------------------------------------------------------------------------

def build_reason_chat_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config: Dict[str, Any] | None = None,
):
    return ReasonChatMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
    )

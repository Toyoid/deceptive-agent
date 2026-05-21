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

from agent_system.utils.history_utils import (
    extract_last_user_message,
    history_messages_to_monitor_text,
)
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

    @staticmethod
    def _build_episode_context(env_dict: Dict[str, Any]) -> Tuple[str, str, str, str]:
        """
        Build ReasonChat state from either legacy single-turn fields or a full
        externally provided history, such as Booking-Assistance tool traces.
        """
        history_messages = env_dict.get("history_messages")
        if history_messages:
            history = history_messages_to_monitor_text(history_messages)
            user_input = env_dict.get("user_input") or extract_last_user_message(history_messages)
            return history, history, str(user_input).strip(), history

        system_prompt = env_dict["system_prompt"]
        question = env_dict["question"]
        system_formatted = CHAT_TEMPLATE.format_system(f"{system_prompt}")
        question_formatted = CHAT_TEMPLATE.format_user(question)
        monitor_background = system_formatted + question_formatted
        return monitor_background, system_formatted, str(question).strip(), monitor_background

    # ------------------------ gym APIs ------------------------
    def reset(self, kwargs: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        if kwargs is None or len(kwargs) == 0:
            raise ValueError("ReasonChat env requires non-empty env_kwargs.")

        self._episodes.clear()
        infos: List[Dict[str, Any]] = []

        for i, env_dict in enumerate(kwargs):
            monitor_background, evidence, user_input, history = self._build_episode_context(env_dict)

            self._episodes.append({
                "task_type": env_dict.get("task_type", "chat"),
                "step": 0, 
                "done": False,
                "evidence": evidence,
                "monitor_background": monitor_background,
                "user_input": user_input,
                "history": history,
            })

            infos.append({
                "task_type": env_dict.get("task_type", "chat"),
                "step": 0,
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

            # ReasonChat is single-turn for now. A multi-turn chat variant should
            # add a simulated user/environment response before returning next obs.
            next_obs.append("")

            rewards.append(0.0)  # reward model will fill actual values later
            dones.append(done)

            info = {
                "task_type": episode["task_type"],
                "step": episode["step"],
                "won": False,
                "user_input": episode["user_input"],  # for judge input
                "evidence": episode["evidence"],  # for judge input
                "agent_response": CHAT_TEMPLATE.format_assistant(assistant_resp),  # for judge input
                "monitor_background": episode["monitor_background"] if done else "",
                "agent_trajectory": CHAT_TEMPLATE.format_assistant(assistant_resp) if done else "",
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

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

import asyncio
import copy
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - dependency is declared, fallback keeps minimal test envs usable
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get("total")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def update(self, n=1):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

from .clients import ChatResponse, OpenAICompatibleChatClient
from .data import EvalRow
from .metrics import compute_eval_metrics
from .prompt_builder import build_actor_messages, messages_to_text
from agent_system.utils.active_rollout import ActiveIndexMap


@dataclass
class ApiRolloutResult:
    episodes: List[Dict[str, Any]]
    success: Dict[str, np.ndarray]
    metrics: Dict[str, float]


class ApiRolloutRunner:
    def __init__(
        self,
        *,
        config,
        client: OpenAICompatibleChatClient,
        rows: List[EvalRow],
        env_factory: Optional[Callable[[Any, int], Any]] = None,
    ) -> None:
        self.config = config
        self.client = client
        self.rows = rows
        if env_factory is None:
            from .env_factory import make_api_eval_env

            env_factory = make_api_eval_env
        self.env_factory = env_factory

    def run(self) -> ApiRolloutResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No loop running, it's safe to start one.
            return asyncio.run(self.run_async())
        
        # A loop IS running, shouldn't call asyncio.run() here.
        raise RuntimeError("ApiRolloutRunner.run() cannot be called from an active event loop; use run_async().")

    async def run_async(self) -> ApiRolloutResult:
        all_episodes: List[Dict[str, Any]] = []
        success_parts: Dict[str, List[np.ndarray]] = {}
        batch_size = int(self.config.data.get("batch_size", 32))
        if batch_size <= 0:
            raise ValueError("data.batch_size must be positive.")

        progress_enabled = bool(self.config.get("progress", {}).get("enabled", True))
        with tqdm(
            total=len(self.rows),
            desc="API rollout",
            unit="traj",
            dynamic_ncols=True,
            disable=not progress_enabled,
        ) as progress:
            for start in range(0, len(self.rows), batch_size):
                chunk = self.rows[start:start + batch_size]
                chunk_result = await self._run_chunk(chunk, chunk_start=start, progress=progress)
                all_episodes.extend(chunk_result.episodes)
                for key, values in chunk_result.success.items():
                    success_parts.setdefault(key, []).append(np.asarray(values))

        success = {
            key: np.concatenate(value, axis=0) if value else np.asarray([], dtype=np.float32)
            for key, value in success_parts.items()
        }
        metrics = compute_eval_metrics(episodes=all_episodes, success=success)
        return ApiRolloutResult(episodes=all_episodes, success=success, metrics=metrics)

    async def _run_chunk(self, rows: List[EvalRow], *, chunk_start: int, progress: tqdm) -> ApiRolloutResult:
        envs = self.env_factory(self.config, len(rows))
        try:
            return await self._run_chunk_with_env(envs, rows, chunk_start=chunk_start, progress=progress)
        finally:
            close = getattr(envs, "close", None)
            if close is not None:
                close()

    async def _run_chunk_with_env(self, envs, rows: List[EvalRow], *, chunk_start: int, progress: tqdm) -> ApiRolloutResult:
        env_kwargs = [copy.deepcopy(row.env_kwargs) for row in rows]
        obs, infos = envs.reset(kwargs=env_kwargs)

        batch_size = len(infos)
        if batch_size != len(rows):
            raise ValueError(f"Environment reset returned {batch_size} infos for {len(rows)} rows.")

        trajectory_ids = [str(uuid.uuid4()) for _ in range(batch_size)]
        episodes = [
            {
                "trajectory_id": trajectory_ids[i],
                "row_index": int(rows[i].index),
                "global_index": int(chunk_start + i),
                "rollout_index": int(rows[i].rollout_index),
                "data_source": rows[i].data_source,
                "ability": copy.deepcopy(rows[i].ability),
                "reward_model": copy.deepcopy(rows[i].reward_model),
                "extra_info": copy.deepcopy(rows[i].extra_info),
                "metadata": copy.deepcopy(rows[i].metadata),
                "source_file": rows[i].source_file,
                "initial_input": self._initial_input(rows[i], obs, infos, i),
                "final_output": "",
                "reward": 0.0,
                "length": 0.0,
                "tool_calls": 0.0,
                "success": None,
                "action_valid_sequence": [],
                "final_info": {},
                "steps": [],
            }
            for i in range(batch_size)
        ]

        is_done = np.zeros(batch_size, dtype=bool)
        total_batch_list: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
        total_infos: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        previous_text_actions = [""] * batch_size

        rollout_max_steps = int(envs.get_rollout_max_steps())
        for step in range(rollout_max_steps):
            active = ActiveIndexMap.from_done(is_done)
            assert active.batch_size == batch_size, "ActiveIndexMap batch size does not match the original batch size"
            active_masks = active.active_mask()
            if not active.has_active:
                break
            progress.set_postfix(
                batch=f"{chunk_start}-{chunk_start + batch_size - 1}",
                step=step + 1,
                active=active.n_active,
                refresh=False,
            )

            # Environment stepping intentionally remains full-batch; only model generation is compacted.
            active_messages = [
                build_actor_messages(row=rows[i], envs=envs, obs=obs, infos=infos, item=i, step=step)
                for i in active.active_idx
            ]
            active_responses = await self.client.generate_batch(active_messages)
            active_text_actions = [response.text for response in active_responses]
            text_actions = active.scatter_actions(active_text_actions, previous_text_actions)

            next_obs, rewards, dones, infos = envs.step(text_actions)
            previous_text_actions = text_actions
            rewards = np.asarray(rewards).reshape(-1)
            dones = np.asarray(dones).reshape(-1).astype(bool)
            if len(rewards) != batch_size or len(dones) != batch_size or len(infos) != batch_size:
                raise ValueError("Environment step returned a batch with inconsistent length.")

            action_valid = np.asarray(
                [bool(info.get("is_action_valid", True)) for info in infos],
                dtype=bool,
            )
            step_tool_calls = np.asarray(
                [float(info.get("tool_calling", 0.0)) for info in infos],
                dtype=np.float32,
            )

            episode_rewards[active_masks] += rewards[active_masks].astype(np.float32)
            episode_lengths[active_masks] += 1.0
            tool_callings[active_masks] += step_tool_calls[active_masks]

            for local_i, i in active.iter_active():
                batch_item = {
                    "active_masks": True,
                    "data_source": rows[i].data_source,
                    "traj_uid": trajectory_ids[i],
                    "response": text_actions[i],
                    "rewards": float(rewards[i]),
                    "is_action_valid": bool(action_valid[i]),
                }
                total_batch_list[i].append(batch_item)
                total_infos[i].append(infos[i])

                response = active_responses[local_i]
                episodes[i]["final_output"] = text_actions[i]
                episodes[i]["reward"] = float(episode_rewards[i])
                episodes[i]["length"] = float(episode_lengths[i])
                episodes[i]["tool_calls"] = float(tool_callings[i])
                episodes[i]["action_valid_sequence"].append(bool(action_valid[i]))
                episodes[i]["final_info"] = dict(infos[i])
                episodes[i]["steps"].append(
                    self._build_step_record(
                        step=step,
                        active=True,
                        messages=active_messages[local_i],
                        response=response,
                        reward=float(rewards[i]),
                        done=bool(dones[i]),
                        is_action_valid=bool(action_valid[i]),
                        tool_calling=float(step_tool_calls[i]),
                        info=infos[i],
                    )
                )

            newly_done = np.logical_and(np.logical_not(is_done), dones)
            if newly_done.any():
                progress.update(int(newly_done.sum()))

            is_done = np.logical_or(is_done, dones)
            obs = next_obs

        unfinished = np.logical_not(is_done)
        if unfinished.any():
            progress.update(int(unfinished.sum()))

        success = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        if "success_rate" in success:
            for i, value in enumerate(np.asarray(success["success_rate"]).reshape(-1)):
                episodes[i]["success"] = float(value)

        metrics = compute_eval_metrics(episodes=episodes, success=success)
        return ApiRolloutResult(episodes=episodes, success=success, metrics=metrics)

    def _initial_input(self, row: EvalRow, obs: Dict[str, Any], infos: List[Dict[str, Any]], item: int) -> str:
        if row.prompt:
            return messages_to_text(row.prompt)
        obs_texts = obs.get("text")
        if obs_texts is not None:
            return "" if obs_texts[item] is None else str(obs_texts[item])
        user_input = infos[item].get("user_input") or infos[item].get("task_description")
        return "" if user_input is None else str(user_input)

    @staticmethod
    def _build_step_record(
        *,
        step: int,
        active: bool,
        messages: List[Dict[str, str]],
        response: ChatResponse,
        reward: float,
        done: bool,
        is_action_valid: bool,
        tool_calling: float,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "step": int(step),
            "active": bool(active),
            "messages": messages,
            "output": response.text,
            "finish_reason": response.finish_reason,
            "api_error": response.error,
            "api_latency": response.latency,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "reward": reward,
            "done": done,
            "is_action_valid": is_action_valid,
            "tool_calling": tool_calling,
            "info": dict(info),
        }

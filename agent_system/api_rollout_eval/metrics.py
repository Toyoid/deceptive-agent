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

from typing import Any, Dict, List, Optional

import numpy as np

from agent_system.utils.metric_contract import EPISODE_METRIC_PREFIX


def _stats(prefix: str, values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {}
    return {
        f"{prefix}/mean": float(values.mean()),
        f"{prefix}/max": float(values.max()),
        f"{prefix}/min": float(values.min()),
    }


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def compute_eval_metrics(
    *,
    episodes: List[Dict[str, Any]],
    success: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {"eval/episode/count": float(len(episodes))}
    if not episodes:
        return metrics

    rewards = _as_float_array([episode.get("reward", 0.0) for episode in episodes])
    lengths = _as_float_array([episode.get("length", 0.0) for episode in episodes])
    tool_calls = _as_float_array([episode.get("tool_calls", 0.0) for episode in episodes])

    metrics.update(_stats("eval/episode/reward", rewards))
    metrics.update(_stats("eval/episode/length", lengths))
    metrics.update(_stats("eval/episode/tool_call_count", tool_calls))

    valid_flags = [
        float(flag)
        for episode in episodes
        for flag in episode.get("action_valid_sequence", [])
    ]
    if valid_flags:
        metrics["eval/action_valid_rate"] = float(np.mean(valid_flags))

    active_steps = [
        step
        for episode in episodes
        for step in episode.get("steps", [])
        if step.get("active", True)
    ]
    if active_steps:
        api_errors = np.asarray([1.0 if step.get("api_error") else 0.0 for step in active_steps], dtype=np.float32)
        latencies = np.asarray(
            [step.get("api_latency") for step in active_steps if step.get("api_latency") is not None],
            dtype=np.float32,
        )
        total_tokens = np.asarray(
            [step.get("total_tokens") for step in active_steps if step.get("total_tokens") is not None],
            dtype=np.float32,
        )
        metrics["eval/api/error_rate"] = float(api_errors.mean())
        metrics.update(_stats("eval/api/latency", latencies))
        metrics.update(_stats("eval/api/total_tokens", total_tokens))

    for data_source in sorted({episode.get("data_source", "unknown") for episode in episodes}):
        source_episodes = [episode for episode in episodes if episode.get("data_source", "unknown") == data_source]
        source_rewards = _as_float_array([episode.get("reward", 0.0) for episode in source_episodes])
        source_tool_calls = _as_float_array([episode.get("tool_calls", 0.0) for episode in source_episodes])
        metrics[f"eval/{data_source}/test_score"] = float(source_rewards.mean())
        metrics.update(_stats(f"eval/{data_source}/tool_call_count", source_tool_calls))

    if success:
        for key, values in success.items():
            array = _as_float_array(values)
            if array.size == 0:
                continue
            if key.startswith(EPISODE_METRIC_PREFIX):
                name = key[len(EPISODE_METRIC_PREFIX):]
                metrics.update(_stats(f"eval/episode/{name}", array))
            elif key.endswith("_rate"):
                metrics[f"eval/{key}"] = float(array.mean())
            else:
                metrics.update(_stats(f"eval/{key}", array))

    return metrics

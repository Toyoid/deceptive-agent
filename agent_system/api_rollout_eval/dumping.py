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

import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, (DictConfig, ListConfig)):
        return json_safe(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def select_episodes(
    episodes: Sequence[Dict[str, Any]],
    *,
    n_trajectories: int | None,
    policy: str,
    seed: int,
) -> List[Dict[str, Any]]:
    episodes = list(episodes)
    if n_trajectories is None:
        n_trajectories = len(episodes)
    n_trajectories = min(max(0, int(n_trajectories)), len(episodes))
    if n_trajectories == 0:
        return []
    if policy == "first":
        return episodes[:n_trajectories]
    if policy == "random":
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(episodes), size=n_trajectories, replace=False)
        return [episodes[int(index)] for index in indices]
    raise ValueError(f"Unsupported dump.policy={policy!r}; expected 'random' or 'first'.")


def dump_trajectories(episodes: Sequence[Dict[str, Any]], config) -> Dict[str, str]:
    output_dir = config.dump.get("output_dir")
    if not output_dir:
        return {}

    selected = select_episodes(
        episodes,
        n_trajectories=config.dump.get("n_trajectories"),
        policy=str(config.dump.get("policy", "random")),
        seed=int(config.data.get("seed", 42)),
    )

    os.makedirs(output_dir, exist_ok=True)
    episodes_path = os.path.join(output_dir, "episodes.jsonl")
    include_steps = bool(config.dump.get("include_steps", True))

    with open(episodes_path, "w", encoding="utf-8") as f:
        for episode in selected:
            payload = json_safe(episode)
            if not include_steps:
                payload = dict(payload)
                payload.pop("steps", None)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    written = {"episodes": episodes_path}
    if include_steps:
        steps_path = os.path.join(output_dir, "steps.jsonl")
        with open(steps_path, "w", encoding="utf-8") as f:
            for episode in selected:
                trajectory_id = episode.get("trajectory_id")
                for step in episode.get("steps", []):
                    payload = {"trajectory_id": trajectory_id, **step}
                    f.write(json.dumps(json_safe(payload), ensure_ascii=False) + "\n")
        written["steps"] = steps_path

    return written


def build_generation_samples(episodes: Sequence[Dict[str, Any]], config) -> List[tuple]:
    n_samples = int(config.dump.get("log_generations", 0) or 0)
    if n_samples <= 0:
        return []
    selected = select_episodes(
        episodes,
        n_trajectories=n_samples,
        policy=str(config.dump.get("policy", "random")),
        seed=int(config.data.get("seed", 42)),
    )
    samples = []
    for episode in selected:
        samples.append((
            episode.get("initial_input", ""),
            episode.get("final_output", ""),
            float(episode.get("reward", 0.0)),
        ))
    return samples

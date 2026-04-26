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

from typing import Any, Dict, List

from agent_system.environments.prompts import DEFAULT_SYSTEM_PROMPT

from .data import EvalRow


def build_actor_messages(
    *,
    row: EvalRow,
    envs,
    obs: Dict[str, Any],
    infos: List[Dict[str, Any]],
    item: int,
    step: int,
) -> List[Dict[str, str]]:
    prompt_source = envs.actor_prompt_source(step)
    if prompt_source == "dataset":
        return [dict(message) for message in row.prompt]
    if prompt_source != "env_obs":
        raise ValueError(f"Unsupported actor prompt source: {prompt_source}")

    obs_texts = obs.get("text")
    obs_text = obs_texts[item] if obs_texts is not None else ""
    system_raw = infos[item].get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    format_prompt = infos[item].get("format_prompt", "")
    system_prompt = system_raw + f"\n{format_prompt}" if format_prompt else system_raw

    return [
        {"role": "system", "content": str(system_prompt)},
        {"role": "user", "content": "" if obs_text is None else str(obs_text)},
    ]


def messages_to_text(messages: List[Dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role.upper()}:\n{content}")
    return "\n\n".join(parts)

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

import os

from omegaconf import OmegaConf

def _cheatshop_env_kwargs(config) -> dict:
    env_root = os.path.dirname(os.path.dirname(__file__))
    if config.env.webshop.use_small:
        file_path = os.path.join(env_root, "environments", "env_package", "webshop", "webshop", "data", "items_shuffle_1000.json")
        attr_path = os.path.join(env_root, "environments", "env_package", "webshop", "webshop", "data", "items_ins_v2_1000.json")
    else:
        file_path = os.path.join(env_root, "environments", "env_package", "webshop", "webshop", "data", "items_shuffle.json")
        attr_path = os.path.join(env_root, "environments", "env_package", "webshop", "webshop", "data", "items_ins_v2.json")

    return {
        "observation_mode": "text",
        "num_products": None,
        "human_goals": config.env.webshop.human_goals,
        "file_path": file_path,
        "attr_path": attr_path,
        "require_self_report": config.env.cheatshop.require_self_report,
        "enable_log_edit": config.env.cheatshop.enable_log_edit,
        "monitor_obs_level": config.env.cheatshop.monitor_obs_level,
        "self_report_cot_visibility": config.env.cheatshop.self_report_cot_visibility,
        "shopping_max_steps": config.env.cheatshop.shopping_max_steps,
        "oracle_top_n": config.env.cheatshop.oracle_top_n,
        "tamper_max_deletes": config.env.cheatshop.tamper_max_deletes,
    }


def make_api_eval_env(config, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    env_name = str(config.env.env_name).lower()
    seed = int(config.env.get("seed", 0))
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    if "reasonchat" in env_name or "reason_chat" in env_name:
        from agent_system.environments.env_manager import ReasonChatEnvironmentManager
        from agent_system.environments.env_package.reason_chat import build_reason_chat_envs, reason_chat_projection

        envs = build_reason_chat_envs(
            seed=seed,
            env_num=batch_size,
            group_n=1,
            is_train=False,
            env_config=config.env,
        )
        return ReasonChatEnvironmentManager(envs, reason_chat_projection, config)

    if "deceptive_search" in env_name:
        from agent_system.environments.env_manager import DeceptiveSearchEnvironmentManager
        from agent_system.environments.env_package.deceptive_search import build_deceptive_search_envs, deceptive_search_projection

        envs = build_deceptive_search_envs(
            seed=seed,
            env_num=batch_size,
            group_n=1,
            is_train=False,
            env_config=config.env,
        )
        return DeceptiveSearchEnvironmentManager(envs, deceptive_search_projection, config)

    if "cheatshop" in env_name:
        from agent_system.environments.env_manager import CheatShopEnvironmentManager
        from agent_system.environments.env_package.cheatshop import build_cheatshop_envs, cheatshop_projection

        if build_cheatshop_envs is None:
            raise ImportError("CheatShop API eval requires ray to be installed.")
        envs = build_cheatshop_envs(
            seed=seed,
            env_num=batch_size,
            group_n=1,
            resources_per_worker=resources_per_worker,
            is_train=False,
            env_kwargs=_cheatshop_env_kwargs(config),
        )
        return CheatShopEnvironmentManager(envs, cheatshop_projection, config)

    raise ValueError(
        f"Unsupported env.env_name={config.env.env_name!r}. "
        "Supported API eval envs: ReasonChat, deceptive_search, CheatShop."
    )

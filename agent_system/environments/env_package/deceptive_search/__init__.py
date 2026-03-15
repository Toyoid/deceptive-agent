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

from .projection import deceptive_search_projection


def build_deceptive_search_envs(
    seed: int = 0,
    env_num:int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config=None
):
    from agent_system.environments.env_package.search.envs import SearchMultiProcessEnv
    from .env import DeceptiveSearchEnv

    return SearchMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
        env_class=DeceptiveSearchEnv,
        task_type="search",
        env_config_key="deceptive_search",
    )

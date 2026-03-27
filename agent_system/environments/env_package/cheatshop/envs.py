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

import ray
import gym
import numpy as np

from .core import CheatShopTextEnv


class CheatShopWorker:
    def __init__(self, seed, env_kwargs):
        env_kwargs = dict(env_kwargs or {})
        env_kwargs["seed"] = seed
        self.env = CheatShopTextEnv(**env_kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})
        info["available_actions"] = self.env.get_available_actions()
        return obs, reward, done, info

    def reset(self, idx):
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info["available_actions"] = self.env.get_available_actions()
        return obs, info

    def get_goals(self):
        return self.env.goals

    def close(self):
        return None


class CheatShopMultiProcessEnv(gym.Env):
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
    ) -> None:
        super().__init__()

        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train:
            assert group_n == 1

        self._rng = np.random.RandomState(seed)
        self._env_kwargs = env_kwargs if env_kwargs is not None else {}
        self.task_type = "Shopping"

        env_worker = ray.remote(**resources_per_worker)(CheatShopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(seed + (i // self.group_n), self._env_kwargs)
            self._workers.append(worker)

        goals = ray.get(self._workers[0].get_goals.remote())
        if not self.is_train:
            self.goal_idxs = range(500)
        else:
            self.goal_idxs = range(500, len(goals))

    def step(self, actions: list):
        if len(actions) != self.num_processes:
            raise ValueError(f"Expected {self.num_processes} actions, got {len(actions)}")

        futures = []
        for worker, action in zip(self._workers, actions):
            futures.append(worker.step.remote(action))

        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()

        futures = []
        for worker, goal_idx in zip(self._workers, idx):
            futures.append(worker.reset.remote(goal_idx))

        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            info["task_type"] = self.task_type
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def close(self):
        if getattr(self, "_closed", False):
            return

        close_futures = []
        for worker in self._workers:
            close_futures.append(worker.close.remote())
        ray.get(close_futures)
        for worker in self._workers:
            ray.kill(worker)
        self._closed = True

    def __del__(self):
        self.close()


def build_cheatshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    return CheatShopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
    )

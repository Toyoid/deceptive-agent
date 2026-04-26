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
import logging
import os
from pprint import pprint

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from verl.utils.config_resolvers import register_resolvers
from verl.utils.tracking import Tracking, ValidationGenerationsLogger

from .clients import OpenAICompatibleChatClient
from .data import load_eval_rows
from .dumping import build_generation_samples, dump_trajectories
from .runner import ApiRolloutRunner


register_resolvers()


def _quiet_api_loggers() -> None:
    for name in ("httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.WARNING)


@hydra.main(config_path="config", config_name="api_rollout_eval", version_base=None)
def main(config) -> None:
    if config.dump.output_dir == "auto":
        config.dump.output_dir = os.path.join(HydraConfig.get().run.dir, "api_rollouts")
    run_api_rollout_eval(config)


def run_api_rollout_eval(config) -> None:
    _quiet_api_loggers()
    OmegaConf.resolve(config)
    pprint(OmegaConf.to_container(config, resolve=True))

    rows = load_eval_rows(config)
    if not rows:
        raise ValueError("No eval rows loaded.")

    logger = None
    client = None
    try:
        logger = Tracking(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
            default_backend=config.trainer.logger,
            config=OmegaConf.to_container(config, resolve=True),
        )
        client = OpenAICompatibleChatClient.from_config(config)
        runner = ApiRolloutRunner(config=config, client=client, rows=rows)
        result = runner.run()

        written = dump_trajectories(result.episodes, config)
        if written:
            print(f"Dumped API rollout trajectories: {written}")

        logger.log(result.metrics, step=0)
        # Log a table of validation samples to the configured logger (wandb or swanlab)
        samples = build_generation_samples(result.episodes, config)
        if samples:
            ValidationGenerationsLogger().log(config.trainer.logger, samples, step=0)

        pprint(result.metrics)
    finally:
        if client is not None:
            asyncio.run(client.close())
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()

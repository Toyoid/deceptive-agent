# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os

import hydra
from hydra.core.hydra_config import HydraConfig
import ray

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.config_resolvers import register_resolvers

# Register custom OmegaConf resolvers before Hydra loads the config
# This enables arithmetic operations in YAML like: ${add:${a},${b}}
register_resolvers()


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    # Resolve "auto" to hydra output directory
    if config.trainer.rollout_data_dir == "auto":
        config.trainer.rollout_data_dir = os.path.join(HydraConfig.get().run.dir, "rollout_data")
    if config.trainer.validation_data_dir == "auto":
        config.trainer.validation_data_dir = os.path.join(HydraConfig.get().run.dir, "validation_data")

    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", "VLLM_LOGGING_LEVEL": "WARN", "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true"}},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.config_resolvers import register_resolvers
        from verl.utils.fs import copy_to_local

        # Register resolvers in the worker process (they were registered in main but not here)
        register_resolvers()

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

        from agent_system.environments import make_envs
        envs, val_envs = make_envs(config)

        # vllm early verify
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        if config.monitor_rollout_ref.enable and config.monitor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm_utils import is_version_ge

            if config.monitor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")
                
        # define worker classes
        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Separate resource pools for actor and monitor so their vLLM engines never share a Ray process
        actor_pool_id = "actor_pool"
        resource_pool_spec = {
            actor_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        monitor_pool_id = None
        if config.monitor_rollout_ref.enable:
            assert config.trainer.nnodes_monitor is not None and config.trainer.nnodes_monitor > 0, "Please set trainer.nnodes_monitor > 0 when enabling monitor_rollout_ref."
            assert (
                config.trainer.n_gpus_per_node_monitor is not None and config.trainer.n_gpus_per_node_monitor > 0
            ), "Please set trainer.n_gpus_per_node_monitor > 0 when enabling monitor_rollout_ref."

            monitor_pool_id = "monitor_pool"
            resource_pool_spec[monitor_pool_id] = [config.trainer.n_gpus_per_node_monitor] * config.trainer.nnodes_monitor

            # make sure total requested devices do not exceed the Ray cluster capacity.
            device_resource_name = "NPU" if config.trainer.device == "npu" else "GPU"
            cluster_resource = ray.cluster_resources().get(device_resource_name)
            if cluster_resource is not None:
                # Ray reports floats; convert to int to avoid floating comparison issues.
                available_devices = int(cluster_resource)
                actor_devices = config.trainer.nnodes * config.trainer.n_gpus_per_node
                monitor_devices = config.trainer.nnodes_monitor * config.trainer.n_gpus_per_node_monitor
                requested_devices = actor_devices + monitor_devices
                assert (
                    requested_devices <= available_devices
                ), f"Requested {requested_devices} {device_resource_name}s (actor+monitor) but only {available_devices} are available on the Ray cluster."

        print(f"resource_pool_spec: {resource_pool_spec}")

        mapping = {
            Role.ActorRollout: actor_pool_id,
            Role.Critic: actor_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            # mapping[Role.RewardModel] = monitor_pool_id if config.monitor_rollout_ref.enable else actor_pool_id
            mapping[Role.RewardModel] = actor_pool_id

        # use reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = actor_pool_id

        # use monitor model
        # NOTE: 
        # 1. We set monitor training engine to be the same as actor_rollout_ref for simplicity and consistency
        # 2. Currently we assume no critic model in monitor training
        # 3. Please use FSDP as `configmonitor_rollout_ref.monitor.strategy`, we do not support `megatron` currently
        # 4. Monitor is placed in a SEPARATE resource pool to avoid vLLM parallel state conflicts
        if config.monitor_rollout_ref.enable:
            monitor_local_path = copy_to_local(config.monitor_rollout_ref.model.path, use_shm=config.monitor_rollout_ref.model.get("use_shm", False))
            monitor_tokenizer = hf_tokenizer(monitor_local_path, trust_remote_code=config.monitor_rollout_ref.data.get("trust_remote_code", False))
            monitor_processor = hf_processor(monitor_local_path, trust_remote_code=config.monitor_rollout_ref.data.get("trust_remote_code", False), use_fast=True)  # used for multimodal LLM, could be none
            
            assert config.actor_rollout_ref.actor.strategy == config.monitor_rollout_ref.monitor.strategy
            if config.monitor_rollout_ref.enable_train_monitor:
                role_worker_mapping[Role.MonitorRollout] = ray.remote(ActorRolloutRefWorker)
                mapping[Role.MonitorRollout] = monitor_pool_id  

                # use reference model for monitor training
                if config.monitor_rollout_ref.algorithm.use_kl_in_reward or config.monitor_rollout_ref.monitor.use_kl_loss:
                    role_worker_mapping[Role.MonitorRef] = ray.remote(ActorRolloutRefWorker)
                    mapping[Role.MonitorRef] = monitor_pool_id
            else:
                role_worker_mapping[Role.MonitorInfer] = ray.remote(ActorRolloutRefWorker)
                mapping[Role.MonitorInfer] = monitor_pool_id
        else:
            monitor_tokenizer = None
            monitor_processor = None
        
        # use judge model for constrained-token scoring on monitor critique validity
        if config.judge_model.enable:
            if config.judge_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import JudgeModelWorker
            else:
                raise NotImplementedError(f"Judge model strategy {config.judge_model.strategy} not supported")
            role_worker_mapping[Role.Judge] = ray.remote(JudgeModelWorker)
            # Put judge model in monitor pool (shares resources with monitor)
            mapping[Role.Judge] = monitor_pool_id if config.monitor_rollout_ref.enable else actor_pool_id
            
            # Load judge tokenizer for critique preprocessing in TrajectoryCollector
            judge_local_path = copy_to_local(config.judge_model.model.path, use_shm=config.judge_model.model.get("use_shm", False))
            judge_tokenizer = hf_tokenizer(judge_local_path, trust_remote_code=config.judge_model.get("trust_remote_code", False))
            judge_processor = hf_processor(judge_local_path, trust_remote_code=config.judge_model.get("trust_remote_code", False), use_fast=True)  # used for multimodal LLM, could be none
        else:
            judge_tokenizer = None
            judge_processor = None

        reward_manager_name = config.reward_model.get("reward_manager", "episode")
        if reward_manager_name == 'episode':
            from agent_system.reward_manager import EpisodeRewardManager

            reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
            val_reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)

            if config.monitor_rollout_ref.enable:
                assert config.algorithm.lagrangian.enable, "Constrained RL is required with 'episode' as reward manager when monitor_rollout_ref is enabled, please set algorithm.lagrangian.enable as True in the config"
                from agent_system.reward_manager import MonitorRewardManager

                monitor_reward_fn = MonitorRewardManager(tokenizer=monitor_tokenizer, num_examine=4, normalize_by_length=False)
                monitor_val_reward_fn = MonitorRewardManager(tokenizer=monitor_tokenizer, num_examine=0, normalize_by_length=False)
            else:
                monitor_reward_fn = None
                monitor_val_reward_fn = None
        elif reward_manager_name == 'actor_monitor':
            assert config.monitor_rollout_ref.enable, "actor_monitor reward manager requires monitor_rollout_ref to be enabled"
            from agent_system.reward_manager.actor_monitor import ActorMonitorRewardManager
            reward_manager_cls = ActorMonitorRewardManager
            reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, role='actor', normalize_by_length=False)
            val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, role='actor', normalize_by_length=False)

            monitor_reward_fn = reward_manager_cls(tokenizer=monitor_tokenizer, num_examine=4, role='monitor', normalize_by_length=False)
            monitor_val_reward_fn = reward_manager_cls(tokenizer=monitor_tokenizer, num_examine=0, role='monitor', normalize_by_length=False)
        else:
            raise NotImplementedError(f"Reward manager {reward_manager_name} not supported yet")

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(
            config=config, 
            tokenizer=tokenizer, 
            processor=processor,
            monitor_tokenizer=monitor_tokenizer,
            monitor_processor=monitor_processor,
            judge_tokenizer=judge_tokenizer,
            judge_processor=judge_processor,
        )

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            monitor_tokenizer=monitor_tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            monitor_reward_fn=monitor_reward_fn,
            monitor_val_reward_fn=monitor_val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
            traj_collector=traj_collector,
            envs=envs,
            val_envs=val_envs,
        )
        trainer.init_workers()
        trainer.calibrate_rm_stats_if_enabled()
        trainer.fit()


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Create a dataset.

    Arguments:
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )

    return dataset


def create_rl_sampler(data_config, dataset):
    """Create a sampler for the dataset.

    Arguments:
        data_config: The data config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    # use sampler for better ckpt resume
    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()

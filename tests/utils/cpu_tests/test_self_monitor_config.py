import importlib.util

import pytest
from omegaconf import OmegaConf


_REQUIRED_TRAINER_DEPS = ("pandas", "ray", "torch", "tensordict", "torchdata", "codetiming", "tqdm")
_MISSING_TRAINER_DEPS = [name for name in _REQUIRED_TRAINER_DEPS if importlib.util.find_spec(name) is None]
pytestmark = pytest.mark.skipif(
    bool(_MISSING_TRAINER_DEPS),
    reason="trainer-stack deps are not installed locally: " + ", ".join(_MISSING_TRAINER_DEPS),
)

if not _MISSING_TRAINER_DEPS:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _make_base_config():
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
    config.monitor_rollout_ref.monitor.ppo_micro_batch_size_per_gpu = 1
    config.monitor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu = 1
    config.reward_model.enable = False
    config.actor_rollout_ref.rollout.multi_turn.enable = False
    config.actor_rollout_ref.rollout.n = 1
    config.actor_rollout_ref.rollout.val_kwargs.n = 1
    return config


def _make_trainer(config):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = config
    trainer.use_reference_policy = False
    trainer.use_critic = False
    return trainer


def test_validate_config_rejects_self_monitor_with_external_monitor():
    config = _make_base_config()
    config.self_monitor.enable = True
    config.monitor_rollout_ref.enable = True

    trainer = _make_trainer(config)
    with pytest.raises(ValueError, match="mutually exclusive"):
        trainer._validate_config()


def test_validate_config_requires_trust_penalty_application_for_self_monitor():
    config = _make_base_config()
    config.self_monitor.enable = True
    config.monitor_rollout_ref.enable = False
    config.judge_model.enable = False
    config.algorithm.lagrangian.enable = False

    trainer = _make_trainer(config)
    with pytest.raises(ValueError, match="requires either"):
        trainer._validate_config()


def test_validate_config_allows_lagrangian_with_self_monitor_only():
    config = _make_base_config()
    config.self_monitor.enable = True
    config.monitor_rollout_ref.enable = False
    config.judge_model.enable = False
    config.algorithm.lagrangian.enable = True

    trainer = _make_trainer(config)
    trainer._validate_config()


def test_validate_config_allows_actor_monitor_with_self_monitor_only():
    config = _make_base_config()
    config.self_monitor.enable = True
    config.monitor_rollout_ref.enable = False
    config.judge_model.enable = False
    config.algorithm.lagrangian.enable = False
    config.reward_model.reward_manager = "actor_monitor"

    trainer = _make_trainer(config)
    trainer._validate_config()


def test_validate_config_rejects_actor_monitor_without_trust_penalty_source():
    config = _make_base_config()
    config.self_monitor.enable = False
    config.verdict_monitor.enable = False
    config.monitor_rollout_ref.enable = False
    config.judge_model.enable = False
    config.algorithm.lagrangian.enable = False
    config.reward_model.reward_manager = "actor_monitor"

    trainer = _make_trainer(config)
    with pytest.raises(ValueError, match="requires monitor_rollout_ref"):
        trainer._validate_config()


def test_validate_config_allows_actor_monitor_with_external_monitor():
    config = _make_base_config()
    config.self_monitor.enable = False
    config.verdict_monitor.enable = False
    config.monitor_rollout_ref.enable = True
    config.algorithm.lagrangian.enable = False
    config.reward_model.reward_manager = "actor_monitor"

    trainer = _make_trainer(config)
    trainer._validate_config()


def test_validate_config_rejects_actor_monitor_and_lagrangian_with_self_monitor():
    config = _make_base_config()
    config.self_monitor.enable = True
    config.monitor_rollout_ref.enable = False
    config.judge_model.enable = False
    config.algorithm.lagrangian.enable = True
    config.reward_model.reward_manager = "actor_monitor"

    trainer = _make_trainer(config)
    with pytest.raises(ValueError, match="both apply trust_penalties"):
        trainer._validate_config()

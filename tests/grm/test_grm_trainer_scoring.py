import importlib.util
from unittest.mock import MagicMock

import numpy as np
import pytest
from omegaconf import OmegaConf

_REQUIRED_DEPS = ("torch", "tensordict", "ray")
_MISSING_DEPS = [name for name in _REQUIRED_DEPS if importlib.util.find_spec(name) is None]
if _MISSING_DEPS:
    pytest.skip(
        "GRM trainer smoke test deps are not installed locally: " + ", ".join(_MISSING_DEPS),
        allow_module_level=True,
    )

import torch

from agent_system.grm.reward import GrmRewardConfig
from verl import DataProto
from verl.trainer.main_grm_rl import GrmRLTrainer


def _valid_grm_config():
    return OmegaConf.create({
        "data": {
            "train_files": None,
            "prompt_key": "prompt",
            "train_batch_size": 2,
            "val_batch_size": None,
            "max_prompt_length": 128,
            "max_response_length": 32,
        },
        "actor_rollout_ref": {
            "model": {
                "path": "judge-model",
                "lora_rank": 0,
                "use_remove_padding": False,
            },
            "actor": {
                "strategy": "fsdp",
                "ppo_mini_batch_size": 4,
                "ppo_micro_batch_size": None,
                "ppo_micro_batch_size_per_gpu": 1,
                "use_dynamic_bsz": False,
                "use_kl_loss": False,
                "loss_agg_mode": "token-mean",
                "ulysses_sequence_parallel_size": 1,
            },
            "ref": {
                "log_prob_use_dynamic_bsz": False,
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": 1,
                "ulysses_sequence_parallel_size": 1,
            },
            "rollout": {
                "n": 2,
                "val_kwargs": {
                    "n": 1,
                    "do_sample": False,
                    "temperature": 0.0,
                },
                "multi_turn": {"enable": False},
                "log_prob_use_dynamic_bsz": False,
                "log_prob_micro_batch_size": None,
                "log_prob_micro_batch_size_per_gpu": 1,
            },
        },
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
        },
        "reward_model": {"enable": False},
        "auxiliary": {"enable": False},
        "monitor_rollout_ref": {"enable": False},
        "self_monitor": {"enable": False},
        "verdict_monitor": {"enable": False},
        "judge_model": {"enable": False},
        "grm": {
            "reward": {
                "invalid_reward": -0.1,
                "exact_reward": 1.0,
                "same_direction_adjacent_reward": 0.0,
                "wrong_valid_reward": -1.0,
                "score_regex": r"<score>\s*([0-4])\s*</score>\s*$",
            },
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "nnodes": 1,
            "total_epochs": 1,
            "total_training_steps": None,
            "val_before_train": False,
        },
    })


def _validate_grm_config(config):
    trainer = GrmRLTrainer.__new__(GrmRLTrainer)
    trainer.config = config
    trainer.use_reference_policy = False
    trainer._validate_config(train_dataset=object())


def test_grm_validate_config_accepts_minimal_valid_config():
    _validate_grm_config(_valid_grm_config())


def test_grm_validate_config_requires_model_path():
    config = _valid_grm_config()
    config.actor_rollout_ref.model.path = None

    with pytest.raises(AssertionError, match="actor_rollout_ref.model.path"):
        _validate_grm_config(config)


def test_grm_validate_config_rejects_grouped_estimator_with_single_rollout():
    config = _valid_grm_config()
    config.actor_rollout_ref.rollout.n = 1

    with pytest.raises(AssertionError, match="rollout.n >= 2"):
        _validate_grm_config(config)


def test_grm_validate_config_rejects_monitor_mode():
    config = _valid_grm_config()
    config.monitor_rollout_ref.enable = True

    with pytest.raises(ValueError, match="prompt-only"):
        _validate_grm_config(config)


def test_grm_trainer_score_batch_places_reward_on_last_response_token():
    trainer = GrmRLTrainer.__new__(GrmRLTrainer)
    trainer.reward_config = GrmRewardConfig()
    trainer.tokenizer = MagicMock()
    trainer.tokenizer.batch_decode.return_value = [
        "<think>ok</think>\n<score>4</score>",
        "missing score",
    ]

    batch = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[10, 11, 0], [12, 0, 0]], dtype=torch.long),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0],
                ],
                dtype=torch.long,
            ),
        },
        non_tensors={
            "label": np.array(["4", "1"], dtype=object),
            "valid_tokens": np.array([["0", "1", "2", "3", "4"], ["0", "1"]], dtype=object),
        },
    )

    scored, metrics = trainer._score_batch(batch, prefix="grm")

    assert metrics["grm/exact_acc"] == pytest.approx(0.5)
    assert metrics["grm/invalid_format_rate"] == pytest.approx(0.5)
    expected = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [-0.1, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(scored.batch["token_level_scores"], expected)
